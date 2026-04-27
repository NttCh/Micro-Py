#!/usr/bin/env python
"""
Optuna search and utilities — Stage A: CNN hyperparameter tuning.
"""

from __future__ import annotations

import json
import math
import os
from copy import deepcopy
from typing import Any, Dict, Optional, Tuple

import optuna
import pandas as pd
from omegaconf import OmegaConf

try:
    from config import cfg as global_cfg
    import config
except ModuleNotFoundError:
    import sys, pathlib
    _root = str(pathlib.Path(__file__).resolve().parents[2])
    if _root not in sys.path:
        sys.path.insert(0, _root)
    from config import cfg as global_cfg
    import config

from .train import repeated_cross_validation, train_stage
from .utils import load_obj, set_seed


# ─────────────────────────────────────────────────────────────────────────────
#  PARAM → CFG PATH MAPPING
# ─────────────────────────────────────────────────────────────────────────────

# Stage A: CNN backbone hyperparams
PARAM_PATHS_A = {
    "lr":                    "optimizer.params.lr",
    "weight_decay":          "optimizer.params.weight_decay",
    "batch_size":            "data.batch_size",
    "gradient_clip_val":     "trainer.gradient_clip_val",
    "label_smoothing":       "training.label_smoothing",
    "freeze_backbone_epochs": "training.freeze_backbone_epochs",
    # class_weight_gminus / class_weight_mixed are applied manually in
    # _run_stage_a_objective (they combine into a list, so no single cfg path).
    # They are intentionally excluded here to avoid apply_best_params_to_cfg
    # trying to write them as scalar values.
}


# ─────────────────────────────────────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def record_trial_history(eval_folder: str):
    """Return a callback that appends trial summary + intermediate values to Excel."""

    def _callback(study, trial):
        import datetime

        history_path = os.path.join(eval_folder, "optuna_trials.xlsx")
        tmp_path = history_path + ".writing.xlsx"

        df_trial = study.trials_dataframe(
            attrs=("number", "value", "params", "state",
                   "datetime_start", "datetime_complete", "intermediate_values")
        ).iloc[[trial.number]].copy()
        df_trial["logged_at"] = datetime.datetime.now()

        iv = getattr(trial, "intermediate_values", {}) or {}
        iv_rows = []
        for step, v in iv.items():
            try:
                iv_rows.append({
                    "number": int(trial.number),
                    "step": int(step),
                    "intermediate_value": float(v),
                    "logged_at": df_trial["logged_at"].iloc[0],
                })
            except Exception:
                pass
        df_iv = pd.DataFrame(iv_rows)

        if os.path.exists(history_path):
            try:
                old = pd.read_excel(history_path, sheet_name=None)
                old_trials = old.get("trials", pd.DataFrame())
                old_iv = old.get("intermediate", pd.DataFrame())
            except Exception:
                old_trials = pd.DataFrame()
                old_iv = pd.DataFrame()
        else:
            old_trials = pd.DataFrame()
            old_iv = pd.DataFrame()

        trials_df = pd.concat([old_trials, df_trial], ignore_index=True)

        if len(df_iv) > 0:
            iv_df = pd.concat([old_iv, df_iv], ignore_index=True)
            if all(c in iv_df.columns for c in ["number", "step", "logged_at"]):
                iv_df = iv_df.drop_duplicates(subset=["number", "step", "logged_at"], keep="last")
        else:
            iv_df = old_iv

        try:
            with pd.ExcelWriter(tmp_path, engine="openpyxl", mode="w") as w:
                trials_df.to_excel(w, sheet_name="trials", index=False)
                iv_df.to_excel(w, sheet_name="intermediate", index=False)
            try:
                os.replace(tmp_path, history_path)
            except PermissionError:
                alt_path = os.path.join(eval_folder, "optuna_trials__NEW.xlsx")
                os.replace(tmp_path, alt_path)
                print(f"[Optuna] WARNING: '{history_path}' is locked. Wrote to: {alt_path}")
        finally:
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass

    return _callback


def _suggest_from_space(trial: optuna.Trial, space: Dict[str, Any]) -> Dict[str, Any]:
    """Sample params from a space definition dict."""
    out: Dict[str, Any] = {}
    for name, spec in (space or {}).items():
        t    = str(spec.get("type", "float")).lower()
        low  = spec.get("low",  spec.get("min", None))
        high = spec.get("high", spec.get("max", None))

        if t in ("float", "uniform"):
            step = spec.get("step", None)
            out[name] = trial.suggest_float(name, float(low), float(high), step=step)
        elif t in ("loguniform", "logfloat"):
            out[name] = trial.suggest_float(name, float(low), float(high), log=True)
        elif t in ("int", "integer"):
            step = int(spec.get("step", 1))
            out[name] = trial.suggest_int(name, int(low), int(high), step=step)
        elif t in ("categorical", "choice"):
            out[name] = trial.suggest_categorical(name, spec["choices"])
        else:
            raise ValueError(f"Unknown optuna param type for '{name}': {t}")
    return out


def space_hash(space: Dict[str, Any]) -> str:
    """Short stable hash of the search space dict."""
    blob = json.dumps(space, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    import hashlib as _h
    return _h.md5(blob.encode("utf-8")).hexdigest()[:8]


def _make_study_name(base_name: str, space: Dict[str, Any]) -> str:
    return f"{base_name}_optuna_A_objMCC_{space_hash(space)}"


def _get_space(cfg) -> Dict[str, Any]:
    """Get Stage A search space from cfg."""
    try:
        raw = OmegaConf.to_container(cfg.optuna.params_A, resolve=True)
        return raw if isinstance(raw, dict) else {}
    except Exception:
        return {}


# ─────────────────────────────────────────────────────────────────────────────
#  REUSE HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _choose_reuse_trial(study, trial_id: int | None):
    """
    Pick a trial:
      - trial_id >= 0 → that specific trial (must be COMPLETE)
      - else          → best COMPLETE trial
    Returns None if not found.
    """
    import optuna as _opt

    if study is None:
        return None

    trials = list(getattr(study, "trials", []) or [])
    if not trials:
        return None

    if trial_id is not None and int(trial_id) >= 0:
        matches = [t for t in trials if int(t.number) == int(trial_id)]
        if not matches:
            return None
        t = matches[0]
        if t.state != _opt.trial.TrialState.COMPLETE or t.value is None:
            return None
        return t

    complete = [t for t in trials
                if t.state == _opt.trial.TrialState.COMPLETE and t.value is not None]
    if not complete:
        return None

    try:
        return study.best_trial
    except Exception:
        direction = getattr(study, "direction", None)
        reverse = (direction == _opt.study.StudyDirection.MAXIMIZE)
        return sorted(complete, key=lambda z: z.value, reverse=reverse)[0]


def _apply_trial_params_to_cfg(cfg, params: dict, exclude_keys: Optional[list] = None):
    """Apply a trial's params into cfg using Stage A mapping."""
    exclude   = set([str(k) for k in (exclude_keys or [])])
    params_f  = {k: v for k, v in (params or {}).items() if str(k) not in exclude}
    mapping_f = {k: path for k, path in PARAM_PATHS_A.items() if str(k) not in exclude}

    try:
        apply_best_params_to_cfg(cfg, params_f, target_nodes=mapping_f)
        return
    except Exception:
        pass

    # Fallback manual apply
    if "lr" in params_f:            cfg.optimizer.params.lr           = float(params_f["lr"])
    if "weight_decay" in params_f:  cfg.optimizer.params.weight_decay = float(params_f["weight_decay"])
    if "batch_size" in params_f:    cfg.data.batch_size               = int(params_f["batch_size"])
    if "label_smoothing" in params_f:
        cfg.training.label_smoothing = float(params_f["label_smoothing"])
    if "gradient_clip_val" in params_f:
        if not hasattr(cfg, "trainer") or cfg.trainer is None:
            cfg.trainer = OmegaConf.create({})
        cfg.trainer.gradient_clip_val = float(params_f["gradient_clip_val"])
    if "freeze_backbone_epochs" in params_f:
        cfg.training.freeze_backbone_epochs = int(params_f["freeze_backbone_epochs"])


# ─────────────────────────────────────────────────────────────────────────────
#  STAGE A — CNN hyperparameter objective
# ─────────────────────────────────────────────────────────────────────────────

def _run_stage_a_objective(
    local_cfg,
    trial_params: Dict[str, Any],
    trial: optuna.Trial,
) -> float:
    """Stage A: apply CNN hyperparams, run training, return val_mcc."""
    local_cfg.training.suppress_artifacts = True
    local_cfg.run_mode = "tune"

    if "lr" in trial_params:
        local_cfg.optimizer.params.lr = float(trial_params["lr"])
    if "weight_decay" in trial_params:
        local_cfg.optimizer.params.weight_decay = float(trial_params["weight_decay"])
    if "batch_size" in trial_params:
        local_cfg.data.batch_size = int(trial_params["batch_size"])
    if "gradient_clip_val" in trial_params:
        if not hasattr(local_cfg, "trainer") or local_cfg.trainer is None:
            local_cfg.trainer = OmegaConf.create({})
        local_cfg.trainer.gradient_clip_val = float(trial_params["gradient_clip_val"])
    if "label_smoothing" in trial_params:
        local_cfg.training.label_smoothing = float(trial_params["label_smoothing"])
    if "freeze_backbone_epochs" in trial_params:
        local_cfg.training.freeze_backbone_epochs = int(trial_params["freeze_backbone_epochs"])

    # ── Optional class weights ───────────────────────────────────────────────
    # Gminus weight > 1.0 → penalise misclassifying real Gminus → reduces FP
    # mixed weight  > 1.0 → penalise misclassifying real mixed  → reduces mixed errors
    if "class_weight_gminus" in trial_params or "class_weight_mixed" in trial_params:
        w_gminus = float(trial_params.get("class_weight_gminus", 1.0))
        w_mixed  = float(trial_params.get("class_weight_mixed",  2.5))
        try:
            OmegaConf.update(local_cfg, "training.class_weights",
                             [w_gminus, 1.0, w_mixed], merge=True)
            print(f"[Stage A] class_weights=[{w_gminus:.2f}, 1.00, {w_mixed:.2f}]")
        except Exception as e:
            print(f"[Stage A] WARNING: could not set class_weights ({e})")

    if hasattr(local_cfg.training, "tuning_epochs_detection"):
        if not hasattr(local_cfg, "trainer") or local_cfg.trainer is None:
            local_cfg.trainer = OmegaConf.create({})
        local_cfg.trainer.max_epochs = int(local_cfg.training.tuning_epochs_detection)

    w_gm_str = f"{trial_params['class_weight_gminus']:.2f}" if "class_weight_gminus" in trial_params else "(default)"
    w_mx_str = f"{trial_params['class_weight_mixed']:.2f}"  if "class_weight_mixed"  in trial_params else "(default)"
    print(f"[Stage A] Trial {trial.number} | "
          f"lr={trial_params.get('lr'):.2e}  "
          f"bs={trial_params.get('batch_size')}  "
          f"smooth={trial_params.get('label_smoothing', 0.0):.2f}  "
          f"w_gminus={w_gm_str}  w_mixed={w_mx_str}")

    csv_path = local_cfg.data.detection_csv
    repeats  = int(getattr(local_cfg.training, "repeated_cv", 1))
    use_cv   = bool(getattr(local_cfg, "use_cv", False))
    n_class  = int(getattr(getattr(local_cfg, "model", {}), "num_classes", 3))

    if use_cv:
        model, score = repeated_cross_validation(
            local_cfg, csv_path, num_classes=n_class,
            stage_name="detection", repeats=repeats, trial=trial,
        )
    else:
        model, score = train_stage(
            local_cfg, csv_path, num_classes=n_class,
            stage_name="detection", trial=trial,
        )

    return float(score) if score is not None else 0.0


# ─────────────────────────────────────────────────────────────────────────────
#  PUBLIC API
# ─────────────────────────────────────────────────────────────────────────────

def run_optuna_search(
    cfg=None,
    study_name: Optional[str] = None,
    direction: str = "maximize",
    storage: Optional[str] = None,
    load_if_exists: bool = True,
    show_progress_bar: bool = True,
    callbacks: Optional[list] = None,
) -> Tuple[Dict[str, Any], optuna.Study]:
    """
    Run Optuna Stage A and return (best_params, study).
    Objective: val_MCC from CNN training.
    """
    if cfg is None:
        cfg = global_cfg

    try:
        if hasattr(cfg.training, "seed"):
            set_seed(int(cfg.training.seed))
    except Exception:
        pass

    space = _get_space(cfg)

    n_trials = int(getattr(cfg.optuna, "n_trials_by_stage", {}).get("A", 10))

    # Sampler / pruner
    sampler = None
    pruner  = None
    try:
        if hasattr(cfg, "optuna") and hasattr(cfg.optuna, "sampler") and "class_name" in cfg.optuna.sampler:
            SamplerCls = load_obj(cfg.optuna.sampler["class_name"])
            sampler = SamplerCls(**cfg.optuna.sampler.get("params", {}))
    except Exception as e:
        print(f"[Optuna] Sampler setup failed; using default. Reason: {e}")

    try:
        if hasattr(cfg, "optuna") and hasattr(cfg.optuna, "pruner") and "class_name" in cfg.optuna.pruner:
            PrunerCls = load_obj(cfg.optuna.pruner["class_name"])
            pruner = PrunerCls(**cfg.optuna.pruner.get("params", {}))
    except Exception as e:
        print(f"[Optuna] Pruner setup failed; disabling. Reason: {e}")

    if study_name is None:
        project = getattr(cfg.general, "project_name", "project")
        base = f"{project}_optuna"
    else:
        base = study_name

    derived_name = _make_study_name(base, space)

    def _create_study(name: str) -> optuna.Study:
        return optuna.create_study(
            direction=direction,
            study_name=name,
            storage=storage,
            load_if_exists=load_if_exists,
            sampler=sampler,
            pruner=pruner,
        )

    study = _create_study(derived_name)

    try:
        study.set_user_attr("_search_space_json", json.dumps(space, sort_keys=True, ensure_ascii=False))
        study.set_user_attr("_objective", "val_mcc_max")
        study.set_user_attr("_stage", "A")
    except Exception:
        pass

    print(
        f"[Optuna] Study: {study.study_name}"
        f" | Sampler: {type(study.sampler).__name__}"
        f" | Objective: val_MCC (CNN)"
    )

    def objective(trial: optuna.Trial) -> float:
        trial_params = _suggest_from_space(trial, space)
        local_cfg    = deepcopy(cfg)
        score        = _run_stage_a_objective(local_cfg, trial_params, trial)

        if score is None or not math.isfinite(score):
            trial.set_user_attr("error", "score_missing_or_nan")
            raise optuna.exceptions.TrialPruned()

        trial.set_user_attr("metrics", {"score": score})
        for k, v in trial_params.items():
            trial.set_user_attr(f"param__{k}", v)

        return float(score)

    try:
        study.optimize(
            objective,
            n_trials=n_trials,
            show_progress_bar=show_progress_bar,
            callbacks=callbacks or [],
        )
    except ValueError as e:
        if "does not support dynamic value space" in str(e):
            alt_name = derived_name + "_v2"
            print(f"[Optuna] Dynamic space clash. Creating new study: {alt_name}")
            study = _create_study(alt_name)
            study.optimize(
                objective,
                n_trials=n_trials,
                show_progress_bar=show_progress_bar,
                callbacks=callbacks or [],
            )
        else:
            raise

    best_params = study.best_trial.params

    # Save best params to JSON for reference
    try:
        eval_dir = os.path.join(getattr(cfg.general, "save_dir", "outputs"), "eval")
        os.makedirs(eval_dir, exist_ok=True)
        out_path = os.path.join(eval_dir, "optuna_best_A.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(best_params, f, ensure_ascii=False, indent=2, sort_keys=True)
        print(f"[Optuna] Saved best params → {out_path}")
    except Exception as e:
        print(f"[Optuna] WARNING: could not save best params ({e})")

    return best_params, study


def apply_best_params_to_cfg(
    cfg,
    best_params: Dict[str, Any],
    target_nodes: Optional[Dict[str, str]] = None,
):
    """Merge best_params into specific nodes in cfg."""
    if target_nodes:
        for k, path in target_nodes.items():
            if k in best_params:
                OmegaConf.update(cfg, path, best_params[k], merge=True)
    else:
        OmegaConf.update(cfg, "hparams", best_params, merge=True)

    # ── Special handling: class_weight_gminus / class_weight_mixed ──────────
    if "class_weight_gminus" in best_params or "class_weight_mixed" in best_params:
        w_gminus = float(best_params.get("class_weight_gminus", 1.0))
        w_mixed  = float(best_params.get("class_weight_mixed",  2.5))
        try:
            OmegaConf.update(cfg, "training.class_weights",
                             [w_gminus, 1.0, w_mixed], merge=True)
            print(f"[apply_best_params] class_weights set to "
                  f"[{w_gminus:.2f}, 1.00, {w_mixed:.2f}]")
        except Exception as e:
            print(f"[apply_best_params] WARNING: could not set class_weights ({e})")
