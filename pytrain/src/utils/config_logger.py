#!/usr/bin/env python
"""
config_logger.py
────────────────
Saves a full, human-readable snapshot of the run configuration and
system environment to <run_dir>/config_snapshot.txt after training.

Usage (add to main.py after training finishes):
    from src.utils.config_logger import save_config_snapshot
    save_config_snapshot(cfg, dirs["eval"])
"""

from __future__ import annotations

import datetime
import json
import os
import platform
import sys
from typing import Any


# ─────────────────────────────────────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _safe_str(v: Any) -> str:
    """Convert any value to a clean string, handling OmegaConf nodes."""
    try:
        from omegaconf import OmegaConf
        if OmegaConf.is_config(v):
            return json.dumps(OmegaConf.to_container(v, resolve=True), indent=2, ensure_ascii=False)
    except Exception:
        pass
    if isinstance(v, (dict, list)):
        try:
            return json.dumps(v, indent=2, ensure_ascii=False)
        except Exception:
            pass
    return str(v)


def _cfg_to_dict(cfg) -> dict:
    """Convert OmegaConf DictConfig (or plain dict) to a plain Python dict."""
    try:
        from omegaconf import OmegaConf
        if OmegaConf.is_config(cfg):
            return OmegaConf.to_container(cfg, resolve=True)
    except Exception:
        pass
    if isinstance(cfg, dict):
        return cfg
    # fallback: iterate attributes
    try:
        return {k: getattr(cfg, k) for k in dir(cfg) if not k.startswith("_")}
    except Exception:
        return {"raw": str(cfg)}


def _format_section(title: str, content: str, width: int = 72) -> str:
    bar = "─" * width
    return f"\n{bar}\n  {title}\n{bar}\n{content}\n"


def _dict_to_text(d: dict, indent: int = 0) -> str:
    """Recursively format a dict as indented key: value lines."""
    lines = []
    pad = "  " * indent
    for k, v in d.items():
        if isinstance(v, dict):
            lines.append(f"{pad}{k}:")
            lines.append(_dict_to_text(v, indent + 1))
        elif isinstance(v, list):
            lines.append(f"{pad}{k}: {json.dumps(v, ensure_ascii=False)}")
        else:
            lines.append(f"{pad}{k}: {v}")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
#  SYSTEM INFO
# ─────────────────────────────────────────────────────────────────────────────

def _get_system_info() -> dict:
    info = {
        "timestamp":      datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "platform":       platform.platform(),
        "python_version": sys.version.split()[0],
        "hostname":       platform.node(),
    }

    # PyTorch + CUDA
    try:
        import torch
        info["torch_version"]   = torch.__version__
        info["cuda_available"]  = str(torch.cuda.is_available())
        if torch.cuda.is_available():
            info["cuda_version"]    = torch.version.cuda
            info["gpu_count"]       = str(torch.cuda.device_count())
            info["gpu_name"]        = torch.cuda.get_device_name(0)
            info["gpu_memory_gb"]   = f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}"
    except Exception:
        info["torch_version"] = "unavailable"

    # PyTorch Lightning
    try:
        import pytorch_lightning as pl
        info["pytorch_lightning_version"] = pl.__version__
    except Exception:
        pass

    # Albumentations
    try:
        import albumentations
        info["albumentations_version"] = albumentations.__version__
    except Exception:
        pass

    return info


# ─────────────────────────────────────────────────────────────────────────────
#  PUBLIC API
# ─────────────────────────────────────────────────────────────────────────────

def save_config_snapshot(
    cfg,
    out_dir: str,
    filename: str = "config_snapshot.txt",
    extra: dict | None = None,
) -> str:
    """
    Write a human-readable config snapshot to <out_dir>/<filename>.

    Parameters
    ----------
    cfg       : OmegaConf DictConfig or plain dict — the full run config
    out_dir   : directory to write the snapshot (e.g. dirs["eval"])
    filename  : output filename (default: config_snapshot.txt)
    extra     : optional dict of extra key-value pairs to append
                (e.g. {"best_ckpt": path, "val_mcc": 0.72})

    Returns
    -------
    str : absolute path to the written file
    """
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, filename)

    cfg_dict    = _cfg_to_dict(cfg)
    system_info = _get_system_info()

    lines = []
    lines.append("=" * 72)
    lines.append("  RUN CONFIGURATION SNAPSHOT")
    lines.append("=" * 72)

    # ── System info ──────────────────────────────────────────────────────────
    lines.append(_format_section("SYSTEM / ENVIRONMENT", _dict_to_text(system_info)))

    # ── Top-level run settings ────────────────────────────────────────────────
    top_keys = ["run_mode", "use_cv", "pretrained_ckpt"]
    top = {k: cfg_dict.get(k, "N/A") for k in top_keys}
    lines.append(_format_section("RUN MODE", _dict_to_text(top)))

    # ── Model ────────────────────────────────────────────────────────────────
    lines.append(_format_section("MODEL", _dict_to_text(cfg_dict.get("model", {}))))

    # ── Training ─────────────────────────────────────────────────────────────
    lines.append(_format_section("TRAINING", _dict_to_text(cfg_dict.get("training", {}))))

    # ── Optimizer & Scheduler ────────────────────────────────────────────────
    opt_sched = {
        "optimizer": cfg_dict.get("optimizer", {}),
        "scheduler": cfg_dict.get("scheduler", {}),
    }
    lines.append(_format_section("OPTIMIZER & SCHEDULER", _dict_to_text(opt_sched)))

    # ── Data ─────────────────────────────────────────────────────────────────
    lines.append(_format_section("DATA", _dict_to_text(cfg_dict.get("data", {}))))

    # ── Augmentation ─────────────────────────────────────────────────────────
    lines.append(_format_section("AUGMENTATION", _dict_to_text(cfg_dict.get("augmentation", {}))))

    # ── PyReason thresholds ──────────────────────────────────────────────────
    lines.append(_format_section("PYREASON THRESHOLDS", _dict_to_text(cfg_dict.get("pyreason", {}))))

    # ── Optuna ───────────────────────────────────────────────────────────────
    lines.append(_format_section("OPTUNA", _dict_to_text(cfg_dict.get("optuna", {}))))

    # ── Trainer ──────────────────────────────────────────────────────────────
    lines.append(_format_section("TRAINER (Lightning)", _dict_to_text(cfg_dict.get("trainer", {}))))

    # ── Extra (e.g. best ckpt path, final val_mcc) ───────────────────────────
    if extra:
        lines.append(_format_section("EXTRA / RESULTS", _dict_to_text(extra)))

    # ── Full raw JSON dump (machine-readable) ────────────────────────────────
    lines.append(_format_section(
        "FULL CONFIG (JSON)",
        json.dumps(cfg_dict, indent=2, ensure_ascii=False, default=str)
    ))

    lines.append("\n" + "=" * 72 + "\n")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"[ConfigLogger] Config snapshot saved → {out_path}")
    return out_path
