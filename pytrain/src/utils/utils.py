# utils.py
#!/usr/bin/env python
"""Utility functions."""

from __future__ import annotations

import datetime
import gc
import importlib
import os
import random
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple
import glob
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# ---------------- time & randomness ----------------
def thai_time():
    """Return the current time in Thailand (UTC+7)."""
    return datetime.datetime.utcnow() + datetime.timedelta(hours=7)


def set_seed(seed: int = 666) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------- lightweight reflection ----------------
def load_obj(obj_path: str):
    """
    Dynamically load an object by its dotted path.

    Args:
        obj_path: Module path in dot notation.

    Returns:
        The loaded object (class, function, etc.).
    """
    parts = obj_path.split(".")
    module_path = ".".join(parts[:-1])
    obj_name = parts[-1]
    module = importlib.import_module(module_path)
    return getattr(module, obj_name)


# ---------------- dataset bookkeeping ----------------
def generate_detection_csv(
    negative_dir: str,
    positive_dir: str,
    output_csv: str,
    mixed_dir: Optional[str] = None,
) -> None:
    """
    Scan class folders and write a (filename, label) CSV.

    Labels:
        0 = G-negative  (negative_dir)
        1 = G-positive  (positive_dir)
        2 = mixed       (mixed_dir, optional)
    """
    dirs_to_check = [negative_dir, positive_dir]
    if mixed_dir:
        dirs_to_check.append(mixed_dir)

    missing = [d for d in dirs_to_check if d and not os.path.isdir(d)]
    if missing:
        print(f"[CSV] Skipped: some data folders not found:")
        for d in missing:
            print(f"  {d}")
        return

    parent = os.path.dirname(output_csv)
    if parent:
        os.makedirs(parent, exist_ok=True)

    data_rows = []
    exts = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

    class_dirs = [
        (0, negative_dir),
        (1, positive_dir),
    ]
    if mixed_dir and os.path.isdir(mixed_dir):
        class_dirs.append((2, mixed_dir))

    for label, root_dir in class_dirs:
        for root, _, files in os.walk(root_dir):
            for fname in files:
                if fname.lower().endswith(exts):
                    full_path = os.path.join(root, fname)
                    data_rows.append({"filename": full_path, "label": label})

    if not data_rows:
        print("[CSV] No images found, CSV not generated.")
        return

    df = pd.DataFrame(data_rows)
    df.to_csv(output_csv, index=False)

    counts = df["label"].value_counts().sort_index().to_dict()
    label_names = {0: "Gminus", 1: "Gplus", 2: "mixed"}
    count_str = "  ".join(f"{label_names.get(k, k)}={v}" for k, v in counts.items())
    print(f"[CSV] Generated detection CSV: {output_csv}  (rows={len(df)}  {count_str})")


def assert_train_data_available(cfg) -> None:
    """
    Ensure the training folders exist and contain at least one image each.
    Raises FileNotFoundError with a friendly message if not.
    """
    neg, pos = cfg.data.negative_dir, cfg.data.positive_dir
    missing_dirs = [d for d in (neg, pos) if not os.path.isdir(d)]
    if missing_dirs:
        raise FileNotFoundError(
            "[Train] Required data folders are missing:\n"
            + "\n".join(f"  - {d}" for d in missing_dirs)
            + "\n\nCreate them and put images inside, or switch run_mode to 'test'."
        )

    exts = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff")

    def _count_imgs(d: str) -> int:
        return sum(
            len(glob.glob(os.path.join(d, "**", pattern), recursive=True))
            for pattern in exts
        )

    if _count_imgs(neg) == 0 or _count_imgs(pos) == 0:
        raise FileNotFoundError(
            "[Train] No images found in one or both class folders.\n"
            f"  negative_dir: {neg}\n"
            f"  positive_dir: {pos}\n"
            "Drop images into these folders (or set run_mode='test')."
        )

    # Mixed dir is optional — warn but don't crash
    mixed = getattr(cfg.data, "mixed_dir", None)
    if mixed and not os.path.isdir(mixed):
        print(f"[Train] WARNING: mixed_dir not found (will train 2-class only): {mixed}")


def make_run_dirs(root: str):
    """
    Create and return standard subdirectories under `root`:
    - best_model/
    - eval/
    - multi_predictions/
    """
    subdirs = {
        "best_model": os.path.join(root, "best_model"),
        "eval": os.path.join(root, "eval"),
        "multi_predictions": os.path.join(root, "multi_predictions"),
    }
    for p in subdirs.values():
        os.makedirs(p, exist_ok=True)
    return subdirs


# ---------------- resource cleanup ----------------
def cleanup_cuda(
    *,
    logger: Optional[Any] = None,
    trainer: Optional[Any] = None,
    models: Optional[Iterable[Any]] = None,
    dataloaders: Optional[Iterable[Any]] = None,
) -> None:
    """
    Best-effort cleanup of GPU/loader resources between CV folds.
    """
    try:
        from pytorch_lightning.loggers import TensorBoardLogger

        if logger and isinstance(logger, TensorBoardLogger):
            exp = getattr(logger, "experiment", None)
            if hasattr(exp, "flush"):
                exp.flush()
            if hasattr(exp, "close"):
                exp.close()
    except Exception:
        pass

    try:
        if dataloaders:
            for dl in dataloaders:
                if hasattr(dl, "_iterator"):
                    setattr(dl, "_iterator", None)
    except Exception:
        pass

    try:
        if models:
            for m in models:
                del m
        if trainer is not None:
            del trainer
        if dataloaders:
            for dl in dataloaders:
                del dl
    except Exception:
        pass

    gc.collect()

    try:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass


# ---------------- training/reuse helpers ----------------
def _reuse_enabled(cfg) -> bool:
    reuse_cfg = getattr(cfg, "reuse", None)
    if reuse_cfg is None:
        return False

    if not bool(getattr(reuse_cfg, "enable", False)):
        return False

    fixed = getattr(reuse_cfg, "fixed", None)
    if fixed is not None:
        try:
            return len(list(fixed)) > 0
        except Exception:
            return False

    study_name = str(getattr(reuse_cfg, "study_name", "") or "").strip()
    return bool(study_name)


def _resolve_pretrained_ckpt_for_training(cfg) -> str | None:
    """
    Return a checkpoint path only when in TRAIN mode and NOT reusing Optuna.
    Otherwise return None (train from scratch).
    """
    if str(getattr(cfg, "run_mode", "")).lower() != "train":
        return None
    if _reuse_enabled(cfg):
        return None
    ckpt = getattr(cfg, "pretrained_ckpt", None)
    if ckpt and isinstance(ckpt, str) and os.path.exists(ckpt):
        return ckpt
    return None


def _print_training_plan(cfg, ckpt_to_use: str | None):
    # num classes
    num_classes = int(getattr(getattr(cfg, "model", {}), "num_classes", 2))
    print(f"[Plan] Classifier head: Standard CNN | num_classes={num_classes}")

    if _reuse_enabled(cfg):
        print(
            "[Plan] TRAIN + REUSE=TRUE → ignore checkpoints; "
            "retrain using reused Optuna trial params."
        )
    else:
        if ckpt_to_use:
            print(
                "[Plan] TRAIN + REUSE=FALSE → initialize from checkpoint: "
                f"{ckpt_to_use}"
            )
        else:
            print(
                "[Plan] TRAIN + REUSE=FALSE → no checkpoint found/provided; "
                "training from scratch."
            )


# ---------------- checkpoint helpers ----------------
def _ckpt_output_tag(ckpt_path: str) -> str:
    """Build a short readable tag from the checkpoint filename."""
    stem = Path(ckpt_path).stem.lower()

    if stem in {"best_detection", "best_from_best_trial"} or stem.startswith("best_from_trial"):
        return "best"

    import re

    m = re.search(r"fold\s*([0-9]+)", stem) or re.search(r"_fold\s*([0-9]+)", stem)
    if m:
        return f"fold{int(m.group(1))}"

    import re as _re

    s = _re.sub(r"[^a-z0-9_-]+", "_", stem)
    return s[:24] if len(s) > 24 else s


# --------- Reading helpers ---------
def read_cfg_snapshot_from_weights(path: str):
    """Returns a plain cfg snapshot dict if the file contains one, else None."""
    try:
        state = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(state, dict) and "cfg_snapshot" in state:
            return state["cfg_snapshot"]
        return None
    except Exception:
        return None


def _read_state(path: str) -> Dict[str, Any]:
    """Read .pt or .ckpt and normalize to a flat state_dict without 'model.' or 'module.' prefixes."""
    obj = torch.load(path, map_location="cpu", weights_only=False)

    if isinstance(obj, dict) and "state_dict" in obj and isinstance(obj["state_dict"], dict):
        sd = obj["state_dict"]
    elif isinstance(obj, dict):
        if all(hasattr(v, "shape") for v in obj.values()):
            sd = obj
        else:
            sd = obj.get("model", obj)
            if not isinstance(sd, dict):
                raise RuntimeError(f"Unsupported checkpoint structure in: {path}")
    else:
        raise RuntimeError(f"Unsupported checkpoint type {type(obj)} in: {path}")

    def _strip(k: str) -> str:
        if k.startswith("model."):
            k = k[6:]
        if k.startswith("module."):
            k = k[7:]
        return k

    return {_strip(k): v for k, v in sd.items()}


# ---------- Smart loader for .pt/.ckpt into a plain nn.Module ----------
def load_weights_into_model(model: nn.Module, path: str, strict: bool = False) -> Tuple[int, int]:
    """
    Load by intersection (name AND shape), ignore the rest.
    Returns (missing_count, unexpected_count).
    """
    src = _read_state(path)
    dst = model.state_dict()

    filtered = {k: v for k, v in src.items() if k in dst and getattr(v, "shape", None) == dst[k].shape}

    loaded_names = set(filtered.keys())
    dst_names = set(dst.keys())
    src_names = set(src.keys())

    missing_keys = list(dst_names - loaded_names)
    unexpected_keys = list(src_names - loaded_names)

    model.load_state_dict(filtered, strict=False)

    basename = os.path.basename(path)
    print(f"[Load] {basename}  loaded={len(loaded_names)}  missing={len(missing_keys)}  unexpected={len(unexpected_keys)}")
    if missing_keys:
        print("  • missing (first 8):", missing_keys[:8])
    if unexpected_keys:
        print("  • unexpected (first 8):", unexpected_keys[:8])

    return len(missing_keys), len(unexpected_keys)
