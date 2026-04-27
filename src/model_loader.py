"""
src/model_loader.py
===================
Build a bare ResNet classifier and load weights from a Lightning checkpoint.
No KAN, no attention heads, no training code.

Key fix: Lightning saves weights as "model.fc.weight" / "model.fc.bias".
We strip the "model." prefix, then replace the fc head AFTER inspecting
the checkpoint so num_classes matches whatever the ckpt was trained with.
"""

import importlib

import torch
import torch.nn as nn


def _load_obj(dotted: str):
    parts = dotted.split(".")
    mod = importlib.import_module(".".join(parts[:-1]))
    return getattr(mod, parts[-1])


def _strip_prefixes(sd: dict) -> dict:
    """Remove 'model.' and 'module.' prefixes added by Lightning / DDP."""
    out = {}
    for k, v in sd.items():
        key = k
        for prefix in ("model.", "module."):
            if key.startswith(prefix):
                key = key[len(prefix):]
        out[key] = v
    return out


def _infer_num_classes(clean_sd: dict) -> int:
    """Read num_classes from fc.weight or classifier weight in checkpoint."""
    for key in ("fc.weight", "classifier.weight", "head.weight"):
        if key in clean_sd:
            return int(clean_sd[key].shape[0])
    return 2   # fallback


def build_and_load(
    ckpt_path: str,
    backbone: str,
    backbone_weights: str,
    num_classes: int = None,   # None = auto-detect from checkpoint
) -> nn.Module:
    """
    Build ResNet and load weights from a Lightning .ckpt.

    num_classes=None  →  detect from checkpoint (recommended for test mode)
    num_classes=N     →  force N classes (use when training fresh)
    """
    # ── Read checkpoint first ────────────────────────────────────
    print(f"  Reading checkpoint: {ckpt_path}")
    raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    if isinstance(raw, dict) and "state_dict" in raw:
        sd = raw["state_dict"]
    elif isinstance(raw, dict):
        sd = raw
    else:
        raise ValueError(f"Unexpected checkpoint format: {type(raw)}")

    clean = _strip_prefixes(sd)

    # Debug: show fc keys found in checkpoint
    fc_keys = [k for k in clean if "fc" in k or "classifier" in k or "head" in k]
    print(f"  Checkpoint head keys: {fc_keys}")

    # Auto-detect num_classes from checkpoint
    ckpt_classes = _infer_num_classes(clean)
    if num_classes is None:
        num_classes = ckpt_classes
        print(f"  Auto-detected num_classes={num_classes} from checkpoint")
    elif num_classes != ckpt_classes:
        print(f"  [WARN] num_classes={num_classes} but checkpoint has {ckpt_classes} classes")
        print(f"         fc head will be randomly initialised for {num_classes} classes")

    # ── Build backbone ───────────────────────────────────────────
    backbone_cls = _load_obj(backbone)
    model = backbone_cls(weights=backbone_weights)

    # Replace head to match detected num_classes
    if hasattr(model, "fc"):
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    elif hasattr(model, "classifier"):
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)

    # ── Load weights (shape-matched) ─────────────────────────────
    dst = model.state_dict()
    filtered = {k: v for k, v in clean.items()
                if k in dst and getattr(v, "shape", None) == dst[k].shape}

    missing    = sorted(set(dst.keys()) - set(filtered.keys()))
    unexpected = sorted(set(clean.keys()) - set(filtered.keys()))

    model.load_state_dict(filtered, strict=False)

    print(f"  [Load] loaded={len(filtered)}  missing={len(missing)}  unexpected={len(unexpected)}")
    if missing:
        print(f"  missing  : {missing}")
    if unexpected:
        print(f"  unexpected (first 5): {unexpected[:5]}")

    if missing:
        critical = [k for k in missing if "fc" in k or "classifier" in k]
        if critical:
            print()
            print("  *** WARNING: head weights not loaded — model will give random predictions ***")
            print(f"  *** Missing: {critical} ***")
            print("  *** Train a new model with run_train.py first ***")
            print()

    model.eval()
    return model, num_classes
