#!/usr/bin/env python
"""Inference and evaluation utilities."""

from pathlib import Path
from typing import Optional

import albumentations as A
import cv2
import numpy as np
import os
import pandas as pd
import torch
from openpyxl import Workbook
from openpyxl.drawing.image import Image as XLImage
from sklearn.metrics import classification_report, fbeta_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import glob
import config as config
from .data import PatchClassificationDataset, _build_valid_transform
from .utils import (
    _ckpt_output_tag,
    load_obj,
    load_weights_into_model,
)
from .viz import save_confusion_matrix
from src.model.model import _build_model_for_weights



# ---------------- evaluation ----------------
def evaluate_model(model, csv_path: str, cfg, stage: str) -> None:
    """
    Evaluate on a validation split, save a confusion matrix PNG, and print a report.

    PNG path: <config.BASE_SAVE_DIR>/eval/confusion_matrix_<stage or best-fold>.png
    """
    # build validation split
    df = pd.read_csv(csv_path)
    _, valid_df = train_test_split(
        df,
        test_size=cfg.data.valid_split,
        random_state=cfg.training.seed,
        stratify=df[cfg.data.label_col],
    )

    # transforms & loader
    tf = A.Compose(
        [load_obj(aug["class_name"])(**aug["params"]) for aug in cfg.augmentation.valid.augs]
    )
    ds = PatchClassificationDataset(
        valid_df,
        cfg.data.folder_path,
        transforms=tf,
        image_col=getattr(cfg.data, "image_col", "filename"),
        label_col=cfg.data.label_col,
    )
    loader = DataLoader(
        ds,
        batch_size=cfg.data.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        pin_memory=True,
        persistent_workers=cfg.data.num_workers > 0,
    )

    # forward pass
    device = next(model.parameters()).device
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, lbs in loader:
            imgs = imgs.to(device, non_blocking=True)
            lbs  = lbs.to(device,  non_blocking=True)
            logits = model(imgs)
            preds  = torch.argmax(logits, dim=1)
            all_preds.extend(preds.detach().cpu().numpy())
            all_labels.extend(lbs.detach().cpu().numpy())

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)

    num_classes = int(getattr(getattr(cfg, "model", {}), "num_classes", 2))
    avg = "macro" if num_classes > 2 else "binary"

    # save CM
    eval_dir = os.path.join(config.BASE_SAVE_DIR, "eval")
    os.makedirs(eval_dir, exist_ok=True)

    is_best_fold = bool(getattr(model, "best_ckpt_path", None))
    title_tag = "Best fold" if is_best_fold else str(stage)
    cm_title  = f"Confusion Matrix: {title_tag}"
    cm_name   = f"confusion_matrix_{'best-fold' if is_best_fold else stage}.png"
    cm_path   = os.path.join(eval_dir, cm_name)

    cls_labels = ["Gminus", "Gplus", "mixed"][:num_classes] if num_classes == 3 else ["0", "1"]
    save_confusion_matrix(
        y_true=all_labels, y_pred=all_preds,
        out_path=cm_path, title=cm_title,
        labels=cls_labels, num_classes=num_classes, cmap="Blues",
    )
    print(f"[Evaluate] Saved confusion matrix → {cm_path}")

    # console report
    print(classification_report(all_labels, all_preds,
                                 target_names=cls_labels, digits=4, zero_division=0))
    f2 = fbeta_score(all_labels, all_preds, beta=2, average=avg, zero_division=0)
    print(f"[Evaluate] {avg.capitalize()} F2: {f2:.4f}")


# ---------------- single-folder prediction (excel with thumbs) ----------------
def predict_test_folder(
    model: torch.nn.Module,
    test_folder: str,
    transform,
    output_excel: str,
    print_results: bool = True,
    model_path: Optional[str] = None,
) -> None:
    """
    Predict on all images in a test folder and save results in an Excel file.
    """
    if not test_folder or test_folder.lower() == "none":
        print("No test folder provided. Skipping test predictions.")
        return

    if model_path and model_path.lower() != "none":
        print(f"Loading model checkpoint from {model_path}")
        state = torch.load(model_path, map_location=torch.device("cpu"))
        state_dict = state.get("state_dict", state) if isinstance(state, dict) else state
        new_state_dict = {}
        for k, v in state_dict.items():
            new_key = k[len("model.") :] if k.startswith("model.") else k
            new_state_dict[new_key] = v
        model.load_state_dict(new_state_dict, strict=False)

    image_files = []
    for root, _, files in os.walk(test_folder):
        for file in files:
            if file.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif")):
                image_files.append(os.path.join(root, file))

    if not image_files:
        print("No image files found in test folder. Skipping test predictions.")
        return

    predictions = []
    device = next(model.parameters()).device
    model.eval()

    with torch.no_grad():
        for file in image_files:
            image = cv2.imread(file, cv2.IMREAD_COLOR)
            if image is None:
                print(f"Warning: Could not read {file}")
                continue
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
            augmented = transform(image=image)
            image_tensor = augmented["image"]
            if not isinstance(image_tensor, torch.Tensor):
                image_tensor = torch.tensor(image_tensor)
            if image_tensor.ndim == 3:
                image_tensor = image_tensor.unsqueeze(0)

            image_tensor = image_tensor.to(device)
            logits = model(image_tensor)
            pred = torch.argmax(logits, dim=1).item()
            prob = torch.softmax(logits, dim=1)[0, 1].item()

            predictions.append(
                {"filename": file, "predicted_label": pred, "probability": prob}
            )
            if print_results:
                print(f"File: {file} -> Predicted Label: {pred}, Prob: {prob:.8f}")

    predictions = sorted(predictions, key=lambda x: x["filename"])
    wb = Workbook()
    ws = wb.active
    ws.title = "Test Predictions"
    ws.append(["Filename", "Predicted Label", "Probability", "Image"])

    row_num = 2
    for pred in predictions:
        ws.append([pred["filename"], pred["predicted_label"], pred["probability"]])
        try:
            img = XLImage(pred["filename"])
            img.width = 100
            img.height = 100
            ws.add_image(img, f"D{row_num}")
        except Exception as e:
            print(f"Could not insert image for {pred['filename']}: {e}")
        row_num += 1

    wb.save(output_excel)
    print(f"Saved test predictions with images to {output_excel}")

    pred_df = pd.DataFrame(predictions)
    print("\nFinal Test Predictions:")
    print(pred_df.to_string(index=False))

# ---------------- copy predict folders ----------------
def copy_predicted_images_by_folder(
    results_by_folder: dict[str, pd.DataFrame],
    dst_root: str,
    *,
    label_to_name: dict[int, str] | None = None,
    overwrite: bool = False,
) -> None:
    """
    Copy source test images into:
      dst_root/<folder_name>/<class_name or pred_{k}>/

    results_by_folder: {folder_name: df with at least ['filepath','predicted_label']}
    dst_root: output directory base, e.g. <out_dir>/predicted_images
    label_to_name: {0: "neg", 1: "pos"} or None -> uses pred_0/pred_1
    overwrite: if False, skip existing files
    """
    import shutil

    if label_to_name is None:
        label_to_name = {}

    os.makedirs(dst_root, exist_ok=True)

    for folder_name, df in results_by_folder.items():
        if df is None or df.empty:
            continue
        if "filepath" not in df.columns or "predicted_label" not in df.columns:
            print(f"[Copy] Skip '{folder_name}': missing filepath/predicted_label columns.")
            continue

        for _, row in df.iterrows():
            src = str(row["filepath"])
            try:
                y = int(row["predicted_label"])
            except Exception:
                continue

            if not src or not os.path.exists(src):
                continue

            cls_name = label_to_name.get(y, f"pred_{y}")
            dst_dir = os.path.join(dst_root, folder_name, cls_name)
            os.makedirs(dst_dir, exist_ok=True)

            base = os.path.basename(src)
            dst = os.path.join(dst_dir, base)

            if (not overwrite) and os.path.exists(dst):
                continue

            try:
                shutil.copy2(src, dst)
            except Exception as e:
                print(f"[Copy] Failed: {src} -> {dst} ({e})")

# ---------------- small helpers ----------------
def apply_compose(augs, x):
    """
    Apply a list of Albumentations transforms sequentially (manual compose).

    Args:
        augs: List of Albumentations transforms.
        x: Dict with "image": <image>.
    """
    image = x["image"]
    for aug in augs:
        image = aug(image=image)["image"]
    return {"image": image}


def _short_sheet(name: str, maxlen: int = 28) -> str:
    """Excel sheet-name safe: strip illegal chars and clamp length."""
    safe = "".join(c for c in str(name) if c not in r'[]:*?/\\')
    return safe[:maxlen] if len(safe) > maxlen else safe


# ---------------- folder → dataframe ----------------
@torch.no_grad()
@torch.no_grad()
def predict_folder_to_df(
    model: torch.nn.Module,
    folder: str,
    transform: A.Compose,
    *,
    device: str | torch.device = None,
) -> pd.DataFrame:
    """
    Run model on all images in a folder and return a DataFrame with:
    ['filepath', 'filename', 'predicted_label', 'probability'].

    - filepath: absolute path to source image (used for copying)
    - filename: basename for reporting
    - predicted_label: int
    - probability: max-softmax confidence
    """
    model.eval()
    if device is None:
        device = next(model.parameters()).device

    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff")
    files = []
    for pat in exts:
        files.extend(glob.glob(os.path.join(folder, "**", pat), recursive=True))
    files = sorted(set(files))

    rows = []
    for f in files:
        img_bgr = cv2.imread(f, cv2.IMREAD_COLOR)
        if img_bgr is None:
            continue
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        aug = transform(image=img)
        x = aug["image"]

        if isinstance(x, np.ndarray):
            x = np.transpose(x, (2, 0, 1)).astype("float32")
            xt = torch.from_numpy(x)
        else:
            xt = x

        if xt.ndim == 3:
            xt = xt.unsqueeze(0)
        xt = xt.to(device)

        logits = model(xt)
        probs = torch.softmax(logits, dim=1)
        conf, pred = probs.max(dim=1)

        rows.append(
            {
                "filepath": os.path.abspath(f),
                "filename": os.path.basename(f),
                "predicted_label": int(pred.item()),
                "probability": float(conf.item()),
            }
        )

    return pd.DataFrame(
        rows,
        columns=["filepath", "filename", "predicted_label", "probability"],
    )

# ---------------- multi-sheet writers ----------------
def save_multi_sheet_workbook(
    results_by_folder: dict[str, pd.DataFrame],
    out_xlsx: str,
) -> None:
    """
    One workbook, one sheet per folder + a Summary sheet.

    results_by_folder: {folder_name: DataFrame}
    """
    os.makedirs(os.path.dirname(out_xlsx), exist_ok=True)
    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
        # detail sheets
        used_names = set()
        for folder_name, df in results_by_folder.items():
            sheet = _short_sheet(folder_name)
            base, suffix = sheet, 1
            while sheet in used_names:
                sheet = _short_sheet(f"{base}_{suffix}")
                suffix += 1
            used_names.add(sheet)
            (df if df is not None else pd.DataFrame()).to_excel(
                writer, sheet_name=sheet, index=False
            )

        # summary sheet
        summary_rows = []
        for folder_name, df in results_by_folder.items():
            if df is None or df.empty:
                c2 = c1 = c0 = n = 0
            else:
                n  = int(len(df))
                c0 = int((df["predicted_label"] == 0).sum()) if "predicted_label" in df.columns else 0
                c1 = int((df["predicted_label"] == 1).sum()) if "predicted_label" in df.columns else 0
                c2 = int((df["predicted_label"] == 2).sum()) if "predicted_label" in df.columns else 0
            summary_rows.append(
                {"folder": folder_name, "n_images": n,
                 "pred_Gminus(0)": c0, "pred_Gplus(1)": c1, "pred_mixed(2)": c2}
            )
        pd.DataFrame(summary_rows).to_excel(writer, sheet_name="Summary", index=False)


def save_multi_ckpt_comparison(
    all_results: dict[str, dict[str, pd.DataFrame]],
    out_xlsx: str,
) -> None:
    """
    all_results: {ckpt_tag: {folder_name: df(...)}}

    Creates a sheet per folder that merges predictions by filename across ckpts,
    plus a summary sheet of 0/1 counts per (ckpt, folder).
    """
    os.makedirs(os.path.dirname(out_xlsx), exist_ok=True)
    summary = []

    # union of all folders
    all_folders = set()
    for d in all_results.values():
        all_folders.update(d.keys())
    all_folders = sorted(all_folders)

    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
        used_names = set()
        for folder in all_folders:
            merged = None
            for ckpt_tag, by_folder in all_results.items():
                df = by_folder.get(
                    folder,
                    pd.DataFrame(columns=["filename", "predicted_label", "probability"]),
                )
                df = df[["filename", "predicted_label", "probability"]].copy()
                df = df.rename(
                    columns={
                        "predicted_label": f"pred_{ckpt_tag}",
                        "probability": f"prob_{ckpt_tag}",
                    }
                )
                merged = df if merged is None else merged.merge(
                    df, on="filename", how="outer"
                )

                src = by_folder.get(folder, pd.DataFrame())
                if not src.empty and "predicted_label" in src.columns:
                    c0 = int((src["predicted_label"] == 0).sum())
                    c1 = int((src["predicted_label"] == 1).sum())
                    c2 = int((src["predicted_label"] == 2).sum())
                else:
                    c0 = c1 = c2 = 0
                summary.append(
                    {"folder": folder, "ckpt": ckpt_tag,
                     "pred_Gminus(0)": c0, "pred_Gplus(1)": c1, "pred_mixed(2)": c2}
                )

            sheet = _short_sheet(folder + "_cmp")
            base, suffix = sheet, 1
            while sheet in used_names:
                sheet = _short_sheet(f"{base}_{suffix}")
                suffix += 1
            used_names.add(sheet)

            (merged if merged is not None else pd.DataFrame()).to_excel(
                writer, sheet_name=sheet, index=False
            )

        pd.DataFrame(summary).to_excel(writer, sheet_name="Summary_All", index=False)


# ---------------- test folds & multi-ckpt runner ----------------
def _parse_test_folds(cfg):
    """
    Normalize and validate a list of test folders.

    Accepts cfg.test.folder_path (str/list) or cfg.test.folder_paths;
    supports globbing; de-duplicates.

    Extra behavior:
      - If a given test folder has subfolders with images (e.g. 'pos', 'neg', 'others'),
        those subfolders are treated as separate test folders.
      - Otherwise, the folder itself is used as a single test folder.
    """
    import ast
    import glob as _glob
    from collections.abc import Sequence

    try:
        from omegaconf import ListConfig

        _seq_types = (list, tuple, set, ListConfig)
    except Exception:
        _seq_types = (list, tuple, set)

    def _norm(p: str) -> str:
        p = os.path.expandvars(os.path.expanduser(str(p).strip()))
        p = os.path.normpath(p)
        return os.path.abspath(p)

    def _folder_has_images(folder: str) -> bool:
        exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff")
        for pat in exts:
            if _glob.glob(os.path.join(folder, "**", pat), recursive=True):
                return True
        return False

    # ----- original root folder collection -----
    raw = getattr(cfg.test, "folder_paths", None) or getattr(cfg.test, "folder_path", None)

    if isinstance(raw, _seq_types) and not isinstance(raw, (str, bytes)):
        candidates = [str(p) for p in list(raw) if p is not None]
    elif isinstance(raw, (str, bytes)):
        s = raw.strip()
        if (s.startswith("[") and s.endswith("]")) or (s.startswith("(") and s.endswith(")")):
            try:
                parsed = ast.literal_eval(s)
                if isinstance(parsed, Sequence) and not isinstance(parsed, (str, bytes)):
                    candidates = [str(p) for p in parsed if p is not None]
                else:
                    candidates = [s]
            except Exception:
                candidates = [s]
        else:
            candidates = [s]
    else:
        candidates = []

    valid_roots = []
    for item in candidates:
        s = str(item).strip()
        expanded = _glob.glob(_norm(s)) or []
        for path in expanded:
            npath = _norm(path)
            if os.path.isdir(npath):
                valid_roots.append(npath)

    if not valid_roots:
        print("[Main] No valid test folders.")
        return []

    # ----- NEW: auto-expand subfolders (pos/neg/others) -----
    expanded = []
    for root in valid_roots:
        # immediate subdirectories
        try:
            entries = os.listdir(root)
        except OSError:
            entries = []

        subdirs = [
            os.path.join(root, d)
            for d in entries
            if os.path.isdir(os.path.join(root, d))
        ]

        # keep only subdirs that actually contain images
        image_subdirs = [sd for sd in subdirs if _folder_has_images(sd)]

        if image_subdirs:
            # Treat each subfolder (pos, neg, others, etc.) as its own test folder
            expanded.extend(image_subdirs)
        else:
            # No image-holding subfolders → use the root itself (old behavior)
            expanded.append(root)

    # de-duplicate while preserving order
    seen = set()
    final = []
    for f in expanded:
        if f not in seen:
            seen.add(f)
            final.append(f)

    if not final:
        print("[Main] No valid test folders after expansion.")
        return []

    print("[Main] Using test folders:")
    for f in final:
        print(f"  - {f}")
    return final


def predict_folders_to_combined_workbook(
    model: torch.nn.Module,
    folders: list[str],
    transform: A.Compose,
    combined_xlsx_path: str,
    *,
    ckpt_print_prefix: str = "",
) -> dict[str, pd.DataFrame]:
    """
    For a given model/ckpt:
      - Run predictions for each folder
      - Save one workbook with one sheet per folder (+ Summary)
      - Copy predicted images into subfolders by predicted label
      - Return dict {folder_name: DataFrame}
    """
    results_by_folder: dict[str, pd.DataFrame] = {}

    for folder in folders:
        folder_name = Path(folder).name
        print(f"{ckpt_print_prefix}-> Predicting on '{folder_name}'")

        df = predict_folder_to_df(model, folder, transform)
        results_by_folder[folder_name] = df

    save_multi_sheet_workbook(results_by_folder, combined_xlsx_path)
    print(f"[Pred] Wrote multi-sheet workbook -> {combined_xlsx_path}")

    out_dir = os.path.dirname(combined_xlsx_path)
    copy_root = os.path.join(out_dir, "predicted_images")

    # Change names if you want: {0:"neg", 1:"pos"}
    copy_predicted_images_by_folder(
        results_by_folder=results_by_folder,
        dst_root=copy_root,
        label_to_name={0: "Gminus", 1: "Gplus", 2: "mixed"},
        overwrite=False,
    )
    print(f"[Pred] Copied images -> {copy_root}")

    return results_by_folder


def _run_multi_ckpt_tests(base_model, cfg, dirs, best_ckpt_path: str):
    """Predict across multiple ckpts × folders, then run XAI and optionally compare outputs."""
    valid_tf = _build_valid_transform(cfg)
    raw_ckpts = getattr(cfg.test, "ckpt_paths", None) or best_ckpt_path
    ckpt_paths = [raw_ckpts] if isinstance(raw_ckpts, str) else list(raw_ckpts)
    ckpt_paths = list(dict.fromkeys(ckpt_paths))
    test_folds = _parse_test_folds(cfg)

    if not test_folds:
        print("[Test] No valid test folders. Skipping test predictions/XAI.")
        return

    all_results_for_compare = {}
    for ckpt in ckpt_paths:
        # load exactly the provided ckpt path (no .pt preference anymore)
        use_path = ckpt
        print(f"\n[Test] Loading weights: {use_path}")

        # Build model matching saved weights (or current cfg)
        num_classes = int(getattr(getattr(cfg, "model", {}), "num_classes", 3))
        model = _build_model_for_weights(cfg, use_path, num_classes=num_classes)
        load_weights_into_model(model, use_path, strict=False)
        model.eval()
        print(f"[Test] Classifier head: Standard CNN | num_classes={num_classes}")

        run_tag = Path(ckpt).parent.parent.name  # e.g. 204752_train_RN50_model3
        ckpt_tag = _ckpt_output_tag(ckpt)        # e.g. "best" or "fold2"
        ckpt_name = f"{run_tag}__{ckpt_tag}"     # unique folder name

        out_dir = os.path.join(dirs["multi_predictions"], ckpt_name)
        os.makedirs(out_dir, exist_ok=True)

        combined_xlsx = os.path.join(
            out_dir, f"predictions_{ckpt_name}_ALL_FOLDERS.xlsx"
        )
        results_dict = predict_folders_to_combined_workbook(
            model=model,
            folders=test_folds,
            transform=valid_tf,
            combined_xlsx_path=combined_xlsx,
            ckpt_print_prefix=f"[{ckpt_name}] ",
        )
        all_results_for_compare[ckpt_name] = results_dict

    if len(all_results_for_compare) > 1:
        compare_xlsx = os.path.join(
            dirs["multi_predictions"], "predictions_COMPARE_ALL_CKPTS.xlsx"
        )
        save_multi_ckpt_comparison(all_results_for_compare, compare_xlsx)
        print(f"[Test] Wrote cross-ckpt comparison workbook → {compare_xlsx}")
