"""
src/ml_inference.py
===================
Run the trained model on all images in a folder.
Returns raw predictions: {image_name: {prob_G, prob_Gplus, predicted, ...}}

No training code. No augmentation beyond resize + normalize.
"""

import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import albumentations as A
import cv2
import numpy as np
import torch
import torch.nn as nn
from albumentations.pytorch import ToTensorV2

import config

IMG_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")


# ─────────────────────────────────────────────────────────────────
#  Transform (same as val transform used during training)
# ─────────────────────────────────────────────────────────────────

def build_transform(img_size: int = 224) -> A.Compose:
    return A.Compose([
        A.Resize(height=img_size, width=img_size, p=1.0),
        A.Normalize(p=1.0),
        ToTensorV2(p=1.0),
    ])


# ─────────────────────────────────────────────────────────────────
#  Filename parser  →  (slide_id, is_patch, row, col)
# ─────────────────────────────────────────────────────────────────

def parse_filename(filename: str) -> Tuple[str, bool, Optional[int], Optional[int]]:
    """
    000043_2_2.jpg  →  slide='000043', is_patch=True,  row=2, col=2
    000043.jpg      →  slide='000043', is_patch=False, row=None, col=None
    """
    stem = Path(filename).stem
    m = re.match(r"^(\d+)_(\d+)_(\d+)$", stem)
    if m:
        return m.group(1), True, int(m.group(2)), int(m.group(3))
    m = re.match(r"^(\d+)$", stem)
    if m:
        return m.group(1), False, None, None
    return stem, False, None, None


# ─────────────────────────────────────────────────────────────────
#  Build image index
# ─────────────────────────────────────────────────────────────────

def build_image_index(
    test_folder: str,
    subfolder_names: Optional[List[str]] = None,
) -> Tuple[Dict, Dict]:
    """
    Returns:
        slides      {slide_id: [image_name, ...]}
        image_info  {image_name: {full_path, slide_id, subfolder,
                                  row, col, is_patch, gt_label}}
    gt_label is the ground-truth class string ('G' or 'Gplus') if subfolders
    are labelled, otherwise None.
    """
    from collections import defaultdict
    slides = defaultdict(list)
    image_info: Dict = {}

    def _add(folder_path: str, subfolder_tag: str, gt_label: Optional[str]):
        if not os.path.isdir(folder_path):
            print(f"  [Index] folder not found: {folder_path}")
            return
        files = [f for f in os.listdir(folder_path) if f.lower().endswith(IMG_EXTS)]
        print(f"  [Index] {subfolder_tag or 'root'}: {len(files)} images")
        for f in files:
            sid, is_patch, row, col = parse_filename(f)
            prefix = f"{subfolder_tag}_" if subfolder_tag else ""
            iname = f"{prefix}{Path(f).stem}"
            # Use raw sid (no subfolder prefix) so patches from pos/neg/mixed
            # subfolders that share the same slide number are grouped together.
            # e.g. pos/000299_1_4.jpg and mixed/000299_0_1.jpg → slide "000299"
            slide_id = sid
            slides[slide_id].append(iname)
            image_info[iname] = {
                "filename":   f,
                "full_path":  os.path.join(folder_path, f),
                "slide_id":   slide_id,
                "subfolder":  subfolder_tag,
                "is_patch":   is_patch,
                "row":        row,
                "col":        col,
                "gt_label":   gt_label,
            }

    if subfolder_names:
        for sf in subfolder_names:
            gt = config.SUBFOLDER_TO_LABEL.get(sf)
            _add(os.path.join(test_folder, sf), sf, gt)
    else:
        _add(test_folder, "", None)

    return dict(slides), image_info


# ─────────────────────────────────────────────────────────────────
#  Run inference
# ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_inference(
    model: nn.Module,
    image_info: Dict,
    transform: A.Compose,
    device: torch.device,
    num_classes: int = 2,
) -> Dict:
    """
    Returns raw_predictions:
        {image_name: {probs, predicted, conf, conf_tier, prob_G, prob_Gplus}}

    Supports 2-class (G / Gplus) and 3-class (G / Gplus / Mix).
    INT_TO_LABEL from config maps class index → label string.
    """
    model.eval()
    raw: Dict = {}
    total = len(image_info)

    for i, (iname, info) in enumerate(image_info.items(), 1):
        if i % 50 == 0 or i == total:
            print(f"  [{i}/{total}] inference…", end="\r")

        img_bgr = cv2.imread(info["full_path"], cv2.IMREAD_COLOR)
        if img_bgr is None:
            print(f"\n  [WARN] Cannot read: {info['full_path']}")
            continue

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
        x = transform(image=img_rgb)["image"].unsqueeze(0).to(device)

        logits = model(x)
        probs  = torch.softmax(logits, dim=1)[0].cpu().tolist()

        # Best class
        best_idx   = int(np.argmax(probs))
        conf       = float(probs[best_idx])
        predicted  = config.INT_TO_LABEL.get(best_idx, str(best_idx))

        if conf >= config.HIGH_CONF_THR:
            tier = "HIGH"
        elif conf >= config.MEDIUM_CONF_THR:
            tier = "MEDIUM"
        else:
            tier = "LOW"

        # Always expose prob_G and prob_Gplus for PyReason compatibility
        entry = {
            "predicted":  predicted,
            "conf":       conf,
            "conf_tier":  tier,
            "prob_G":     float(probs[0]) if len(probs) > 0 else 0.0,
            "prob_Gplus": float(probs[1]) if len(probs) > 1 else 0.0,
        }
        # Also store all class probs for reference
        for idx, p in enumerate(probs):
            label = config.INT_TO_LABEL.get(idx, f"class{idx}")
            entry[f"prob_{label}"] = float(p)

        raw[iname] = entry

    print()
    return raw