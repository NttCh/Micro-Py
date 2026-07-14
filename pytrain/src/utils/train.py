#!/usr/bin/env python
import pytrain.src._path_fix  # noqa: F401  — ensures repo root is on sys.path
"""Training utilities and main training functions."""

import os
import re
import json
import sys
import time
import itertools
from pathlib import Path
from typing import List, Optional, Tuple
from torch.utils.data import DataLoader, WeightedRandomSampler
import albumentations as A
import numpy as np
import optuna
import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.loggers.logger import Logger as _PLLoggerBase
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import fbeta_score, matthews_corrcoef


# ─── Freeze/Unfreeze Backbone Callback ───────────────────────────────────────
class FreezeBackboneCallback(pl.Callback):
    """
    Implements cfg.training.freeze_strategy:
      - "none"     : do nothing (all params always trainable)
      - "initial"  : freeze backbone body for the first `freeze_backbone_epochs` epochs,
                     then unfreeze all layers. The classification head (fc / classifier)
                     is ALWAYS kept trainable regardless of freeze state.

    ResNet50 (and most torchvision models) has no separate `.backbone` attribute —
    the entire model IS the backbone plus the head (fc). So we identify backbone
    parameters as everything EXCEPT the head, and always keep the head unfrozen.
    """

    def __init__(self, cfg):
        super().__init__()
        self.strategy        = str(getattr(cfg.training, "freeze_strategy", "none")).lower()
        self.freeze_epochs   = int(getattr(cfg.training, "freeze_backbone_epochs", 5))
        self.final_unfreeze  = bool(getattr(cfg.training, "final_unfreeze", True))
        self._frozen         = False

    def _head_param_ids(self, pl_module) -> set:
        """Return the set of parameter ids belonging to the classification head."""
        model = pl_module.model
        head = getattr(model, "fc", None) or getattr(model, "classifier", None)
        if head is None:
            return set()
        return {id(p) for p in head.parameters()}

    def _set_backbone_grad(self, pl_module, requires_grad: bool):
        """Set requires_grad for backbone params only; head always stays True."""
        head_ids = self._head_param_ids(pl_module)
        n_frozen = 0
        for p in pl_module.model.parameters():
            if id(p) in head_ids:
                p.requires_grad = True   # head always trainable
            else:
                p.requires_grad = requires_grad
                if not requires_grad:
                    n_frozen += 1
        status = "FROZEN" if not requires_grad else "UNFROZEN"
        print(f"[FreezeCallback] Backbone {status} ({n_frozen} non-head params affected) | head always trainable")

    def on_train_start(self, trainer, pl_module):
        if self.strategy == "initial":
            self._set_backbone_grad(pl_module, requires_grad=False)
            self._frozen = True

    def on_train_epoch_start(self, trainer, pl_module):
        if self.strategy != "initial":
            return
        epoch = trainer.current_epoch  # 0-indexed
        if self._frozen and epoch >= self.freeze_epochs and self.final_unfreeze:
            self._set_backbone_grad(pl_module, requires_grad=True)
            self._frozen = False
            print(f"[FreezeCallback] Epoch {epoch}: backbone unfrozen for fine-tuning")


from torch.utils.data import DataLoader

import config as config
from config import cfg
from .callbacks import (
    MasterValidationMetricsCallback,
    OptunaCompositeReportingCallback,
    CleanTQDMProgressBar,
    TrialFoldProgressCallback,
    OverallProgressCallback,
    LocalTrainEvalCallback,
    TrainingResourceLogger,
    append_resource_log
)
from .data import PatchClassificationDataset
from src.model.model import LitClassifier, build_classifier
from .utils import load_obj, set_seed, thai_time, cleanup_cuda


class NullLogger(_PLLoggerBase):
    """No-op logger to satisfy self.log(..., logger=True) without writing files."""

    def __init__(self) -> None:
        super().__init__()
        self._name = "noop"
        self._version = "0"

    @property
    def name(self) -> str:
        return self._name

    @property
    def version(self) -> str:
        return self._version

    @property
    def experiment(self):
        return None

    def log_hyperparams(self, params) -> None:
        pass

    def log_metrics(self, metrics, step: int) -> None:
        pass

    def save(self) -> None:
        pass

    def finalize(self, status: str) -> None:
        pass


# ---------------- logger helper ----------------
def build_logger(cfg, run_id: str) -> TensorBoardLogger | NullLogger:
    is_tune = str(getattr(cfg, "run_mode", "")).lower() == "tune"
    if is_tune:
        return NullLogger()

    save_dir = os.path.join(config.BASE_SAVE_DIR, run_id)
    os.makedirs(save_dir, exist_ok=True)
    return TensorBoardLogger(save_dir=save_dir, name=f"{cfg.general.project_name}")


from omegaconf import OmegaConf

def _ensure_cfg_has_backbone(cfg):
    try:
        if OmegaConf.select(cfg, "model.backbone") is None:
            bb = OmegaConf.to_container(OmegaConf.select(config.cfg, "model.backbone"), resolve=True)
            if OmegaConf.select(cfg, "model") is None:
                OmegaConf.update(cfg, "model", {}, merge=True)
            OmegaConf.update(cfg, "model.backbone", bb, merge=True)
            print("[Guard] cfg.model.backbone was missing -> restored from config.cfg")
    except Exception as e:
        print(f"[Guard] WARNING: could not ensure backbone ({e})")


# ─────────────────────────────────────────────────────────────────────────────
#  Val sweep helpers
# ─────────────────────────────────────────────────────────────────────────────

def _parse_filename_for_sweep(filename: str):
    """
    Parse slide_id, row, col from a filename stem.

    '000043_2_2.jpg' -> ('000043', 2, 2)     grid patch
    '000043.jpg'     -> ('000043', None, None) individual
    'anything_else'  -> ('anything_else', None, None)
    """
    stem = Path(filename).stem
    m = re.match(r"^(\d+)_(\d+)_(\d+)$", stem)
    if m:
        return m.group(1), int(m.group(2)), int(m.group(3))
    m = re.match(r"^(\d+)$", stem)
    if m:
        return m.group(1), None, None
    return stem, None, None


@torch.no_grad()
def _save_val_sweep_csv(
    model,
    valid_df: pd.DataFrame,
    valid_tf,
    cfg,
    fold_number: int,
    int_to_label: dict,
    image_col: str = "filename",
    label_col: str = "label",
):
    """
    Run inference on the val fold (best-epoch weights already loaded) and
    write a patch_results CSV that sweep_thresholds.run_once() can consume.

    Columns written:
        image_name, slide_id, row, col, subfolder, filename,
        gt_label, ml_predicted, ml_conf, ml_prob_G, ml_prob_Gplus

    Output: <BASE_SAVE_DIR>/eval/val_patch_results_fold{N}.csv
    """
    eval_dir = os.path.join(config.BASE_SAVE_DIR, "eval")
    os.makedirs(eval_dir, exist_ok=True)
    out_csv = os.path.join(eval_dir, f"val_patch_results_fold{fold_number}.csv")

    # unwrap LitClassifier -> bare nn.Module if needed
    device    = next(model.parameters()).device if list(model.parameters()) else torch.device("cpu")
    raw_model = getattr(model, "model", model)
    raw_model.eval()

    ds = PatchClassificationDataset(
        valid_df,
        cfg.data.folder_path,
        transforms=valid_tf,
        image_col=image_col,
        label_col=label_col,
    )
    loader = DataLoader(
        ds,
        batch_size=cfg.data.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        pin_memory=False,
        persistent_workers=False,
    )

    filenames     = ds.df[image_col].tolist()
    gt_labels_int = ds.df[label_col].tolist()

    all_probs = []
    for images, _ in loader:
        images = images.to(device)
        logits = raw_model(images)
        probs  = torch.softmax(logits, dim=1).cpu().numpy()
        all_probs.append(probs)

    if not all_probs:
        print(f"[SweepCSV] Fold {fold_number}: no val images — skipping.")
        return

    all_probs = np.concatenate(all_probs, axis=0)   # (N, num_classes)
    n_classes = all_probs.shape[1]

    rows = []
    for i, (fn, gt_int) in enumerate(zip(filenames, gt_labels_int)):
        probs_i      = all_probs[i]
        best_idx     = int(np.argmax(probs_i))
        conf         = float(probs_i[best_idx])
        ml_predicted = int_to_label.get(best_idx, str(best_idx))
        gt_label     = int_to_label.get(int(gt_int), str(gt_int))
        slide_id, row, col = _parse_filename_for_sweep(fn)

        rows.append({
            "image_name":    Path(fn).stem,
            "slide_id":      slide_id,
            "row":           row,
            "col":           col,
            "subfolder":     gt_label,
            "filename":      fn,
            "gt_label":      gt_label,
            "ml_predicted":  ml_predicted,
            "ml_conf":       round(conf, 6),
            "ml_prob_G":     round(float(probs_i[0]), 6) if n_classes > 0 else 0.0,
            "ml_prob_Gplus": round(float(probs_i[1]), 6) if n_classes > 1 else 0.0,
        })

    df_out = pd.DataFrame(rows)
    df_out.to_csv(out_csv, index=False)
    n_grid = int(df_out["row"].notna().sum())
    print(
        f"[SweepCSV] Fold {fold_number}: {len(df_out)} patches "
        f"({n_grid} grid, {len(df_out) - n_grid} individual) → {out_csv}"
    )


def _write_sweep_skipped(eval_dir: str, reason: str):
    """Write a marker so downstream code can detect that sweep was skipped."""
    path = os.path.join(eval_dir, "best_thresholds.json")
    with open(path, "w") as f:
        json.dump({"_skipped": True, "_reason": reason}, f, indent=2)


def run_val_threshold_sweep(eval_dir: str, n_folds: int):
    """
    After all CV folds finish:
      1. Load val_patch_results_fold*.csv files from eval_dir
      2. Combine into one DataFrame
      3. Run the full SWEEP grid (same logic as sweep_thresholds.main)
      4. Apply staged safety filter
      5. Write best_thresholds.json to eval_dir
      6. Write val_threshold_sweep.csv (full ranked results)

    Skips gracefully when:
      - no fold CSVs exist
      - no grid patches (individual-only dataset)
      - sweep_thresholds.py is not importable
    """
    fold_csvs = sorted(Path(eval_dir).glob("val_patch_results_fold*.csv"))
    if not fold_csvs:
        print("[ValSweep] No fold CSVs found — skipping threshold sweep.")
        return

    dfs = []
    for p in fold_csvs:
        try:
            dfs.append(pd.read_csv(p))
        except Exception as e:
            print(f"[ValSweep] Could not read {p}: {e}")

    if not dfs:
        print("[ValSweep] All fold CSVs unreadable — skipping.")
        return

    combined = pd.concat(dfs, ignore_index=True)
    n_grid   = int(combined["row"].notna().sum())
    print(
        f"[ValSweep] Combined {len(fold_csvs)} fold(s): "
        f"{len(combined)} patches, {n_grid} grid-eligible."
    )

    if n_grid == 0:
        print(
            "[ValSweep] No grid patches found.\n"
            "  Filenames need the pattern <slide>_<row>_<col>.jpg "
            "(e.g. 000043_2_2.jpg) for the spatial sweep to work.\n"
            "  Thresholds in config.py will be used unchanged."
        )
        _write_sweep_skipped(eval_dir, reason="no_grid_patches")
        return

    # save combined CSV for transparency / debugging
    combined_csv = os.path.join(eval_dir, "val_patch_results_combined.csv")
    combined.to_csv(combined_csv, index=False)

    # import sweep_thresholds at call-time (not at module load)
    # Try several candidate locations for the repo root where
    # sweep_thresholds.py should live alongside config.py.
    try:
        import importlib
        import importlib.util

        # candidate roots: config module location, this file's parents
        _candidates = []
        try:
            import config as _cm
            _candidates.append(str(Path(_cm.__file__).resolve().parent))
        except Exception:
            pass
        # walk up from this file: pytrain/src/utils/train.py -> repo root is 3 levels up
        _here = Path(__file__).resolve()
        for _n in range(1, 5):
            _candidates.append(str(_here.parents[_n]))

        _sw = None
        for _root in _candidates:
            _sweep_path = os.path.join(_root, "sweep_thresholds.py")
            if os.path.exists(_sweep_path):
                if _root not in sys.path:
                    sys.path.insert(0, _root)
                _spec = importlib.util.spec_from_file_location("sweep_thresholds", _sweep_path)
                _sw   = importlib.util.module_from_spec(_spec)
                _spec.loader.exec_module(_sw)
                print(f"[ValSweep] Loaded sweep_thresholds from: {_sweep_path}")
                break

        if _sw is None:
            print(
                "[ValSweep] sweep_thresholds.py not found in any of:\n"
                + "\n".join(f"  {r}" for r in _candidates)
                + "\n  Skipping threshold sweep."
            )
            return
    except Exception as e:
        print(f"[ValSweep] Could not load sweep_thresholds: {e} — skipping.")
        return

    # ── baseline metrics (CNN only, no rules) ────────────────────
    raw_base           = _sw.rebuild_raw(combined)
    slides, image_info = _sw.rebuild_structures(combined)

    pairs = [
        (image_info[n].get("gt_label"), raw_base[n]["predicted"])
        for n in raw_base
        if image_info[n].get("gt_label") is not None
    ]
    if not pairs:
        print("[ValSweep] No GT labels in combined val data — skipping.")
        return

    baseline = _sw._metrics([p[0] for p in pairs], [p[1] for p in pairs])
    print(
        f"[ValSweep] Val baseline  MCC={baseline['mcc']:.4f}  "
        f"Coverage={baseline['coverage']:.1%}  Deferred={baseline['n_deferred']}"
    )

    # ── full sweep grid ───────────────────────────────────────────
    keys   = list(_sw.SWEEP.keys())
    combos = list(itertools.product(*_sw.SWEEP.values()))
    print(f"[ValSweep] Testing {len(combos)} threshold combinations ...")

    rows_out = []
    for combo in combos:
        params = dict(zip(keys, combo))
        try:
            m = _sw.run_once(
                raw_base, slides, image_info,
                high_thr               = params["HIGH_CONF_THR"],
                nb_min                 = params["NEIGHBOR_AGREE_MIN"],
                section_mode           = params["SECTION_MODE"],
                section_ratio_thr      = params["SECTION_MAJORITY_RATIO_THR"],
                section_min_patches    = params["SECTION_MIN_PATCHES"],
                n_sections             = params["N_SECTIONS"],
                window_rows            = params["WINDOW_ROWS"],
                window_cols            = params["WINDOW_COLS"],
                anchor_conf_thr        = params["ANCHOR_CONF_THR"],
                section_minority_min   = params["SECTION_MINORITY_MIN_PATCHES"],
                section_minority_ratio = params["SECTION_MINORITY_RATIO_THR"],
                section_max_mixed_frac = params["SECTION_MAX_MIXED_FRACTION"],
            )
        except Exception as _run_e:
            # print first failure so we can diagnose, then skip that combo
            if not rows_out:
                print(f"[ValSweep] run_once error (first combo shown): {_run_e}")
            continue
        rows_out.append({
            **params, **m,
            "delta_mcc": m["mcc"] - baseline["mcc"],
        })

    if not rows_out:
        print("[ValSweep] No sweep results produced — skipping.")
        return

    result_df = (
        pd.DataFrame(rows_out)
        .sort_values(
            ["n_tier1_errors", "n_worsened", "n_unflagged",
             "mcc", "commit_accuracy", "coverage", "n_r1b"],
            ascending=[True, True, True, False, False, False, False],
        )
        .reset_index(drop=True)
    )
    result_df.insert(0, "rank", result_df.index + 1)

    # ── staged safety filter (mirrors sweep_thresholds.main) ─────
    safe_df = pd.DataFrame()
    found   = False
    for t1e in _sw.SAFETY_TIER1_ERRORS_RANGE:
        g1 = result_df[result_df["n_tier1_errors"] <= t1e]
        if not len(g1):
            continue
        for wor in _sw.SAFETY_WORSENED_RANGE:
            g2 = g1[g1["n_worsened"] <= wor]
            if not len(g2):
                continue
            for unflg in _sw.SAFETY_UNFLAGGED_RANGE:
                g3 = g2[g2["n_unflagged"] <= unflg]
                if not len(g3):
                    continue
                for cmt in _sw.SAFETY_COMMIT_ACC_RANGE:
                    g4 = g3[g3["commit_accuracy"] >= cmt].copy()
                    if not len(g4):
                        continue
                    safe_df = g4
                    found   = True
                    break
                if found:
                    break
            if found:
                break
        if found:
            break

    if found:
        best_row = safe_df.iloc[0]
    else:
        print("[ValSweep] No safe config found — using best overall.")
        best_row = result_df.iloc[0]

    # ── save full sweep CSV ───────────────────────────────────────
    result_df.to_csv(
        os.path.join(eval_dir, "val_threshold_sweep.csv"), index=False
    )

    # ── build and save best_thresholds.json ──────────────────────
    THRESH_KEYS = [
        "SECTION_MODE",
        "HIGH_CONF_THR",
        "ANCHOR_CONF_THR",
        "SECTION_MAJORITY_RATIO_THR",
        "SECTION_MIN_PATCHES",
        "N_SECTIONS",
        "WINDOW_ROWS",
        "WINDOW_COLS",
        "NEIGHBOR_AGREE_MIN",
        "SECTION_MINORITY_MIN_PATCHES",
        "SECTION_MINORITY_RATIO_THR",
        "SECTION_MAX_MIXED_FRACTION",
    ]
    best_thresholds = {}
    for k in THRESH_KEYS:
        v = best_row.get(k)
        if v is not None:
            best_thresholds[k] = v.item() if hasattr(v, "item") else v

    best_thresholds["_val_mcc"]         = float(best_row["mcc"])
    best_thresholds["_val_n_worsened"]  = int(best_row["n_worsened"])
    best_thresholds["_baseline_mcc"]    = float(baseline["mcc"])
    best_thresholds["_delta_mcc"]       = float(best_row["mcc"] - baseline["mcc"])
    best_thresholds["_n_folds_used"]    = n_folds
    best_thresholds["_n_grid_patches"]  = n_grid

    json_path = os.path.join(eval_dir, "best_thresholds.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(best_thresholds, f, indent=2, ensure_ascii=False)

    print(
        f"[ValSweep] ✓ Saved → {json_path}\n"
        f"  MCC={best_thresholds['_val_mcc']:.4f}  "
        f"Δ={best_thresholds['_delta_mcc']:+.4f}  "
        f"worsened={best_thresholds['_val_n_worsened']}  "
        f"HIGH_CONF={best_thresholds['HIGH_CONF_THR']}  "
        f"ANCHOR={best_thresholds['ANCHOR_CONF_THR']}  "
        f"MINOR_RATIO={best_thresholds['SECTION_MINORITY_RATIO_THR']:.2f}"
    )


# ─────────────────────────────────────────────────────────────────────────────
#  train_stage
# ─────────────────────────────────────────────────────────────────────────────

def train_stage(
    cfg,
    csv_path: str,
    num_classes: int,
    stage_name: str,
    trial: Optional["optuna.trial.Trial"] = None,
    suppress_metrics: bool = False,
    trial_number: Optional[int] = None,
    total_trials: Optional[int] = None,
    fold_number: Optional[int] = None,
    total_folds: Optional[int] = None,
    train_idx: Optional[np.ndarray] = None,
    valid_idx: Optional[np.ndarray] = None,
) -> Tuple[LitClassifier, float]:
    """Train one stage and return (model, score)."""

    res_logger = TrainingResourceLogger()

    if bool(getattr(getattr(cfg, "training", {}), "clear_cuda_before_stage", False)) and torch.cuda.is_available():
        try:
            torch.cuda.synchronize()
        except Exception:
            pass
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass

    is_tune           = str(getattr(cfg, "run_mode", "")).lower() == "tune"
    suppress_artifacts = bool(getattr(cfg.training, "suppress_artifacts", False)) or is_tune
    disable_ckpt      = bool(getattr(getattr(cfg, "training", {}), "disable_checkpointing", False))
    enable_ckpt       = (not suppress_artifacts) and (not disable_ckpt)

    # --- num_classes from config if not explicitly passed ---
    num_classes = int(getattr(getattr(cfg, "model", {}), "num_classes", num_classes))

    # Determine if binary or multiclass for metrics
    is_multiclass = num_classes > 2

    # split
    full_df = pd.read_csv(csv_path)

    label_col = cfg.data.label_col
    y0 = full_df[label_col]

    def _is_int_like(s: pd.Series) -> bool:
        try:
            return np.issubdtype(s.dtype, np.integer)
        except Exception:
            return False

    if not _is_int_like(y0):
        uniq = sorted(pd.Series(y0.astype(str)).unique().tolist())
        mapping = {k: i for i, k in enumerate(uniq)}
        full_df[label_col] = y0.astype(str).map(mapping).astype(int)
        print(f"[LabelMap] {mapping}")

    if train_idx is not None and valid_idx is not None:
        train_df = full_df.iloc[train_idx].reset_index(drop=True)
        valid_df = full_df.iloc[valid_idx].reset_index(drop=True)
        print(f"[INFO] Using provided CV split → Train size: {len(train_df)} | Valid size: {len(valid_df)}")
    else:
        train_df, valid_df = train_test_split(
            full_df,
            test_size=cfg.data.valid_split,
            random_state=cfg.training.seed,
            stratify=full_df[cfg.data.label_col],
        )
        print(f"[INFO] Random split → Train size: {len(train_df)} | Valid size: {len(valid_df)}")

    # Class weights computed after the mixed-cap block below.

    # ── Mixed-class cap ──────────────────────────────────────────────────────
    mixed_cap_ratio = getattr(cfg.data, "mixed_cap_ratio", None)
    MIXED_LABEL = 2
    if mixed_cap_ratio is not None:
        try:
            cap_ratio = float(mixed_cap_ratio)
            if 0.0 < cap_ratio < 1.0:
                total_train = len(train_df)
                max_mixed   = int(total_train * cap_ratio)
                mixed_mask  = train_df[label_col] == MIXED_LABEL
                n_mixed     = int(mixed_mask.sum())
                if n_mixed > max_mixed:
                    keep_mixed   = train_df[mixed_mask].sample(n=max_mixed, random_state=int(cfg.training.seed))
                    non_mixed_df = train_df[~mixed_mask]
                    train_df     = pd.concat([non_mixed_df, keep_mixed]).sample(
                        frac=1, random_state=int(cfg.training.seed)
                    ).reset_index(drop=True)
                    print(f"[MixedCap] Capped mixed from {n_mixed} → {max_mixed} "
                          f"({cap_ratio*100:.0f}% of {total_train} train rows)")
                else:
                    print(f"[MixedCap] mixed={n_mixed} already ≤ cap {max_mixed}; no change")
        except Exception as e:
            print(f"[MixedCap] WARNING: could not apply mixed_cap_ratio ({e})")

    # Recompute class weights after cap
    y_train = train_df[cfg.data.label_col].astype(int).to_numpy()
    counts  = np.bincount(y_train, minlength=int(num_classes)).astype(np.float32)
    counts  = np.maximum(counts, 1.0)
    class_w = (counts.sum() / counts).astype(np.float32)
    class_w = class_w * (len(class_w) / class_w.sum())
    print(f"[Balance] post-cap train counts={counts.tolist()} class_w(auto)={class_w.tolist()}")

    # ── cfg.training.class_weights override ─────────────────────────────────
    cfg_loss_weights = getattr(cfg.training, "class_weights", None)
    if cfg_loss_weights is not None:
        try:
            override_w = np.array(list(cfg_loss_weights), dtype=np.float32)
            if len(override_w) == num_classes:
                override_w = override_w * (num_classes / override_w.sum())
                print(f"[Balance] Using cfg.training.class_weights for LOSS: {override_w.tolist()}")
                loss_class_w = override_w
            else:
                print(f"[Balance] WARNING: cfg.training.class_weights length {len(override_w)} "
                      f"!= num_classes {num_classes}; ignoring override.")
                loss_class_w = class_w
        except Exception as e:
            print(f"[Balance] WARNING: could not apply class_weights override ({e}); using auto.")
            loss_class_w = class_w
    else:
        loss_class_w = class_w

    try:
        setattr(cfg.training, "ce_class_weights", loss_class_w.tolist())
    except Exception:
        pass

    # transforms
    train_tf = A.Compose([load_obj(aug["class_name"])(**aug["params"]) for aug in cfg.augmentation.train.augs])
    valid_tf = A.Compose([load_obj(aug["class_name"])(**aug["params"]) for aug in cfg.augmentation.valid.augs])

    common_cols = dict(
        image_col=getattr(cfg.data, "image_col", "filename"),
        label_col=cfg.data.label_col,
    )
    train_ds       = PatchClassificationDataset(train_df, cfg.data.folder_path, transforms=train_tf, **common_cols)
    valid_ds       = PatchClassificationDataset(valid_df, cfg.data.folder_path, transforms=valid_tf, **common_cols)
    train_eval_ds  = PatchClassificationDataset(train_df, cfg.data.folder_path, transforms=valid_tf, **common_cols)

    base_loader_args = dict(
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        persistent_workers=cfg.data.num_workers > 0,
        pin_memory=True,
    )

    sample_w     = torch.as_tensor(class_w[y_train], dtype=torch.double)
    sampler      = WeightedRandomSampler(weights=sample_w, num_samples=len(sample_w), replacement=True)
    train_loader = DataLoader(train_ds, shuffle=False, sampler=sampler, **base_loader_args)
    valid_loader = DataLoader(valid_ds, shuffle=False, **base_loader_args)
    train_eval_loader = DataLoader(train_eval_ds, shuffle=False, **base_loader_args)

    def _normalize_ckpts(x):
        try:
            from omegaconf import ListConfig
            seq = (list, tuple, ListConfig)
        except Exception:
            seq = (list, tuple)

        def ok(p):
            return isinstance(p, str) and p.strip() and p.strip().lower() != "none"

        if x is None:          return []
        if isinstance(x, seq): return [p for p in x if ok(p)]
        if isinstance(x, str): return [x] if ok(x) else []
        return []

    _ensure_cfg_has_backbone(cfg)
    model = build_classifier(cfg, num_classes=num_classes)

    if not is_tune:
        ckpts = _normalize_ckpts(getattr(cfg, "pretrained_ckpt", None))
        if ckpts:
            try:
                print(f"Loading pretrained checkpoint from {ckpts[0]}")
                raw_sd = torch.load(ckpts[0], map_location="cpu")
                sd = {k.replace("model.", ""): v for k, v in raw_sd.items()}
                model.load_state_dict(sd, strict=False)
            except Exception as e:
                print(f"[WARN] Could not load pretrained_ckpt: {e}")

    # --- label smoothing from cfg ---
    label_smoothing = float(getattr(cfg.training, "label_smoothing", 0.0))
    print(f"[Train] label_smoothing={label_smoothing}  num_classes={num_classes}")

    lit_model = LitClassifier(
        cfg=cfg,
        model=model,
        num_classes=num_classes,
        label_smoothing=label_smoothing,
    )

    run_id   = f"{stage_name}_{thai_time().strftime('%Y%m%d-%H%M%S')}_{int(time.time()*1000)}"
    logger   = build_logger(cfg, run_id)
    save_dir = None
    if not suppress_artifacts:
        save_dir = os.path.join(config.BASE_SAVE_DIR, run_id)
        os.makedirs(save_dir, exist_ok=True)

    max_epochs = getattr(getattr(cfg, "trainer", {}), "max_epochs", None)
    if not max_epochs:
        max_epochs = cfg.training.tuning_epochs_detection

    callbacks: List[pl.Callback] = []
    monitor_metric = getattr(cfg.training, "early_stop_metric", "val_mcc")
    _maximize  = {"val_f2", "val_f1", "val_mcc", "val_recall", "val_precision", "val_acc", "val_auc"}
    monitor_mode = "max" if monitor_metric in _maximize else "min"

    mc = None
    if enable_ckpt:
        mc = ModelCheckpoint(
            dirpath=save_dir,
            monitor=monitor_metric,
            mode=monitor_mode,
            save_top_k=1,
            save_last=False,
            filename=f"{stage_name}-" + "{epoch:02d}-{" + monitor_metric + ":.4f}",
        )

    import math as _math

    def _to_float(x):
        try:
            import torch as _t
            if isinstance(x, _t.Tensor):
                return float(x.detach().cpu().item())
        except Exception:
            pass
        try:
            return float(x)
        except Exception:
            return None

    best_metric = {"value": None}

    class BestMetricCallback(pl.Callback):
        def on_validation_epoch_end(self, trainer, pl_module) -> None:
            m = trainer.callback_metrics.get(monitor_metric, None)
            v = _to_float(m)
            if v is None or not _math.isfinite(v):
                return
            cur = best_metric["value"]
            if cur is None:
                best_metric["value"] = v
                return
            if monitor_mode == "max":
                if v > cur: best_metric["value"] = v
            else:
                if v < cur: best_metric["value"] = v

    es_cfg     = getattr(cfg.training, "early_stopping", None)
    es_enabled = bool(getattr(es_cfg, "enabled", True)) if es_cfg else True

    core_callbacks = [
        FreezeBackboneCallback(cfg),
        OverallProgressCallback(),
        TrialFoldProgressCallback(
            trial_number=trial_number,
            total_trials=total_trials,
            fold_number=fold_number,
            total_folds=total_folds,
        ),
        LocalTrainEvalCallback(train_eval_loader=train_eval_loader),
        BestMetricCallback(),
        CleanTQDMProgressBar(),
    ]

    if not suppress_metrics:
        if es_enabled:
            patience = cfg.training.early_stopping.patience_hpo if trial else cfg.training.early_stopping.patience_final
            callbacks.append(EarlyStopping(monitor=monitor_metric, patience=int(patience), mode=monitor_mode))
        if enable_ckpt and mc is not None:
            callbacks.append(mc)
        callbacks += core_callbacks
    else:
        if enable_ckpt and mc is not None:
            callbacks.append(mc)
        callbacks += core_callbacks

    if trial is not None:
        callbacks.append(OptunaCompositeReportingCallback(trial, cfg, metric_name=monitor_metric))

    if not suppress_artifacts:
        callbacks.append(
            MasterValidationMetricsCallback(
                fold_number=0 if fold_number is None else int(fold_number),
                head_tag="cnn",
            )
        )

    n_train_batches   = len(train_loader)
    log_every_n_steps = max(1, min(50, n_train_batches))
    trainer = Trainer(
        max_epochs=max_epochs,
        devices=cfg.trainer.devices,
        accelerator=cfg.trainer.accelerator,
        precision=cfg.trainer.precision,
        gradient_clip_val=getattr(cfg.trainer, "gradient_clip_val", None),
        logger=logger if logger is not None else False,
        callbacks=callbacks,
        enable_model_summary=False,
        enable_checkpointing=enable_ckpt,
        log_every_n_steps=log_every_n_steps,
        num_sanity_val_steps=0,
    )

    res_logger.start()
    trainer.fit(lit_model, train_dataloaders=train_loader, val_dataloaders=valid_loader)
    train_time_sec = res_logger.end()

    best_path = ""
    if enable_ckpt:
        best_path = getattr(mc, "best_model_path", "") if mc else ""

    if enable_ckpt and best_path and os.path.exists(best_path):
        print(f"[train_stage] Reloading BEST weights from: {best_path}")
        try:
            ckpt = torch.load(best_path, map_location="cpu", weights_only=False)
        except TypeError:
            ckpt = torch.load(best_path, map_location="cpu")

        state_dict = ckpt.get("state_dict", ckpt)
        missing, unexpected = lit_model.load_state_dict(state_dict, strict=False)
        if missing or unexpected:
            print(f"[train_stage] load_state_dict: missing={len(missing)}, unexpected={len(unexpected)}")

        try:
            import shutil
            best_dir = os.path.join(config.BASE_SAVE_DIR, "best_model")
            os.makedirs(best_dir, exist_ok=True)
            dst_ckpt = os.path.join(best_dir, f"best_{stage_name}.ckpt")
            shutil.copy2(best_path, dst_ckpt)
            lit_model.best_ckpt_path = dst_ckpt
            print(f"[train_stage] Copied best checkpoint → {dst_ckpt}")
        except Exception as e:
            print(f"[train_stage] Could not archive best artifacts: {e}")
    else:
        if enable_ckpt:
            print("[train_stage] WARNING: No best_model_path found; returning last-epoch weights.")

    # score
    score = best_metric["value"]
    if (score is None) and enable_ckpt and (mc is not None):
        score = _to_float(getattr(mc, "best_model_score", None))
    if score is None:
        score = _to_float(trainer.callback_metrics.get(monitor_metric))

    try:
        epochs_run = int(getattr(trainer, "current_epoch", 0)) + 1
    except Exception:
        epochs_run = 0

    try:
        num_params = TrainingResourceLogger.count_params(lit_model.model)
    except Exception:
        num_params = 0

    try:
        eval_dir   = os.path.join(config.BASE_SAVE_DIR, "eval")
        model_name = str(getattr(getattr(cfg, "model", {}), "backbone", {}).get("class_name", "model")).split(".")[-1]
        optuna_cfg = getattr(cfg, "optuna", None)

        row = res_logger.to_row(
            run_id=str(run_id),
            is_tune=bool(is_tune),
            optuna_stage=str(getattr(optuna_cfg, "stage", "")) if optuna_cfg is not None else "",
            optuna_study=str(getattr(optuna_cfg, "study_name", "")) if optuna_cfg is not None else "",
            trial_number=None if trial is None else int(trial.number),
            fold_number=None if fold_number is None else int(fold_number),
            cv_run=None if getattr(cfg.training, "cv_run", None) is None else int(cfg.training.cv_run),
            model_name=model_name,
            stage=str(stage_name),
            head_type="cnn",
            num_params=int(num_params),
            epochs_run=int(epochs_run),
            train_time_sec=float(train_time_sec),
            peak_gpu_mb=float(res_logger.peak_gpu_mb or 0.0),
            val_score=None if score is None else float(score),
        )

        append_resource_log(row, eval_dir)
    except Exception as e:
        print(f"[Resource] WARNING: could not log resources ({e})")

    # ── save val predictions in sweep-compatible format ───────────────────────
    # Only during real training (not HPO tune) and only for named CV folds.
    # Best-epoch weights are already loaded above, so inference here uses them.
    if not is_tune and not suppress_artifacts and fold_number is not None:
        try:
            _save_val_sweep_csv(
                model        = lit_model,
                valid_df     = valid_df,
                valid_tf     = valid_tf,
                cfg          = cfg,
                fold_number  = fold_number,
                int_to_label = {0: "G", 1: "Gplus", 2: "Mixed"},
                image_col    = getattr(cfg.data, "image_col", "filename"),
                label_col    = cfg.data.label_col,
            )
        except Exception as _sw_e:
            print(f"[SweepCSV] WARNING: could not save val sweep CSV ({_sw_e})")

    cleanup_cuda(
        logger=logger if logger else None,
        trainer=trainer,
        models=[lit_model],
        dataloaders=[train_loader, valid_loader, train_eval_loader],
    )

    return lit_model, float(score if score is not None else 0.0)


# ─────────────────────────────────────────────────────────────────────────────
#  train_with_cross_validation
# ─────────────────────────────────────────────────────────────────────────────

def train_with_cross_validation(
    cfg,
    csv_path: str,
    num_classes: int,
    stage_name: str,
    cv_run: int,
    total_cv: int,
    verbose: bool = True,
    trial: Optional["optuna.trial.Trial"] = None,
) -> Tuple["LitClassifier", float, List[float]]:

    monitor_metric = getattr(getattr(cfg, "training", {}), "early_stop_metric", "val_mcc")
    full_df: pd.DataFrame = pd.read_csv(csv_path)

    skf = StratifiedKFold(
        n_splits=int(cfg.training.num_folds),
        shuffle=True,
        random_state=int(cfg.training.seed),
    )

    splits = list(skf.split(full_df, full_df[cfg.data.label_col]))

    import hashlib

    def _hash_idx(arr):
        h = hashlib.sha256()
        h.update(np.asarray(arr).astype(np.int64).tobytes())
        return h.hexdigest()[:12]

    print("\n[CV Split Check]")
    for i, (t_idx, v_idx) in enumerate(splits, start=1):
        print(f"  fold{i}: train={len(t_idx)} val={len(v_idx)} | "
              f"train_hash={_hash_idx(t_idx)} val_hash={_hash_idx(v_idx)}")
    print("-" * 60)

    fold_scores: List[float] = []
    fold_models: List["LitClassifier"] = []

    only_fold = getattr(getattr(cfg, "training", {}), "only_fold", None)
    try:
        only_fold = int(only_fold) if only_fold is not None else None
    except Exception:
        only_fold = None

    if only_fold is not None:
        if only_fold < 1 or only_fold > int(cfg.training.num_folds):
            raise ValueError(f"cfg.training.only_fold={only_fold} is out of range (1..{int(cfg.training.num_folds)})")
        print(f"[CV] only_fold enabled → running ONLY fold {only_fold}/{int(cfg.training.num_folds)}")

    _is_tuning      = trial is not None
    _min_fold_score = 0.0
    if _is_tuning:
        try:
            _min_fold_score = float(
                getattr(getattr(cfg, "training", {}), "min_fold_score", 0.0) or 0.0
            )
        except Exception:
            _min_fold_score = 0.0

    for fold, (t_idx, v_idx) in enumerate(splits):
        fold_num = fold + 1
        if only_fold is not None and fold_num != only_fold:
            continue

        fold_seed = int(cfg.training.seed) + fold
        set_seed(fold_seed)

        if verbose:
            print(f"CV {cv_run}/{total_cv} | Fold {fold_num}/{cfg.training.num_folds}: ", end="")

        lit_model, val_metric = train_stage(
            cfg=cfg,
            csv_path=csv_path,
            num_classes=num_classes,
            stage_name=f"{stage_name}_fold{fold_num}",
            trial=trial,
            fold_number=fold_num,
            total_folds=int(cfg.training.num_folds),
            train_idx=t_idx,
            valid_idx=v_idx,
        )

        if verbose:
            print(f"| {monitor_metric}: {val_metric:.4f}")

        fold_scores.append(float(val_metric))
        fold_models.append(lit_model)

        # ── Fold-score gate ───────────────────────────────────────
        if _is_tuning and _min_fold_score > 0.0:
            remaining = int(cfg.training.num_folds) - fold_num
            if val_metric < _min_fold_score and remaining > 0:
                mean_so_far = float(np.mean(fold_scores))
                print(
                    f"[CV Prune] Fold {fold_num} {monitor_metric}={val_metric:.4f} "
                    f"< min_fold_score={_min_fold_score:.2f} — "
                    f"pruning trial (skipping {remaining} remaining fold(s), "
                    f"mean so far={mean_so_far:.4f})"
                )
                try:
                    trial.report(mean_so_far, step=fold_num)
                except Exception:
                    pass
                raise optuna.exceptions.TrialPruned(
                    f"fold{fold_num} {monitor_metric}={val_metric:.4f} "
                    f"< threshold {_min_fold_score:.2f}"
                )

    if len(fold_models) == 0:
        raise RuntimeError("No folds were executed. Check cfg.training.only_fold and cfg.training.num_folds.")

    if len(fold_scores) == 1:
        mean_score = float(fold_scores[0])
        best_idx   = 0
    else:
        best_idx   = int(np.argmax(fold_scores))
        mean_score = float(np.mean(fold_scores))

    std_score = float(np.std(fold_scores, ddof=1)) if len(fold_scores) > 1 else 0.0
    print(f"[CV] {len(fold_scores)} fold(s) mean±std({monitor_metric}) = {mean_score:.4f} ± {std_score:.4f}")

    # ── threshold sweep on accumulated val CSVs (real training only) ─────────
    # trial is None means we are in a real training run, not Optuna HPO.
    # Sweep reads val_patch_results_fold*.csv written by _save_val_sweep_csv
    # inside each train_stage call above.
    if trial is None:
        try:
            run_val_threshold_sweep(
                eval_dir = os.path.join(config.BASE_SAVE_DIR, "eval"),
                n_folds  = len(fold_scores),
            )
        except Exception as _sw_e:
            print(f"[ValSweep] WARNING: threshold sweep failed ({_sw_e})")

    return fold_models[best_idx], mean_score, fold_scores


# ─────────────────────────────────────────────────────────────────────────────
#  repeated_cross_validation
# ─────────────────────────────────────────────────────────────────────────────

def repeated_cross_validation(
    cfg,
    csv_path: str,
    num_classes: int,
    stage_name: str,
    repeats: int,
    trial: Optional["optuna.trial.Trial"] = None,
) -> Tuple["LitClassifier", float]:

    monitor_metric = getattr(getattr(cfg, "training", {}), "early_stop_metric", "val_mcc")
    base_seed      = int(cfg.training.seed)

    repeat_means: List[float]           = []
    repeat_models: List["LitClassifier"] = []

    for r in range(int(repeats)):
        local_seed = base_seed + r
        set_seed(local_seed)

        prev_cfg_seed     = int(cfg.training.seed)
        cfg.training.seed = local_seed
        try:
            print(f"\n=== Repeated CV run {r+1}/{repeats} (seed={local_seed}) — optimizing {monitor_metric} ===")

            lit_model_cv, mean_score, fold_scores = train_with_cross_validation(
                cfg=cfg,
                csv_path=csv_path,
                num_classes=num_classes,
                stage_name=stage_name,
                cv_run=r + 1,
                total_cv=int(repeats),
                verbose=True,
                trial=trial,
            )

            fold_std = float(np.std(fold_scores, ddof=1)) if len(fold_scores) > 1 else 0.0
            print(f"[Repeated CV] run {r+1}: folds mean±std({monitor_metric}) = {mean_score:.4f} ± {fold_std:.4f}")

        finally:
            cfg.training.seed = prev_cfg_seed

        repeat_means.append(float(mean_score))
        repeat_models.append(lit_model_cv)

    mean_over_repeats = float(np.mean(repeat_means))
    std_over_repeats  = float(np.std(repeat_means, ddof=1)) if len(repeat_means) > 1 else 0.0
    print(f"\n[Repeated CV] mean±std({monitor_metric}) over {repeats} runs = {mean_over_repeats:.4f} ± {std_over_repeats:.4f}")

    best_idx = int(np.argmax(repeat_means))
    return repeat_models[best_idx], mean_over_repeats


# ─────────────────────────────────────────────────────────────────────────────
#  optuna progress print
# ─────────────────────────────────────────────────────────────────────────────

def print_trial_thai_callback(study, trial) -> None:
    """Optuna callback: print best value & Thai timestamp on completion."""
    if trial.state == optuna.trial.TrialState.COMPLETE:
        stage = getattr(getattr(cfg, "optuna", {}), "stage", "A")
        metric_name = "macro_mcc" if stage == "B" else getattr(getattr(cfg, "training", {}), "early_stop_metric", "val_mcc")
        print(f"[Optuna] Trial complete with {metric_name}={trial.value:.4f} at {thai_time()}")
