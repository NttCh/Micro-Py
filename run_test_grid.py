"""
run_test_grid.py
================
Runs PyReason ONLY on grid-eligible patches.

Individual images (no row_col in filename) and sparse-grid slides
(fewer than MIN_PATCHES_FOR_VOTE patches) are excluded from inference,
metrics, and all output files entirely.

Usage:
    python run_test_grid.py

Results saved to: OUTPUT_DIR/grid_only/evaluation_results.xlsx

Why use this?
    run_test.py includes every patch but PyReason never fires on
    individual/sparse images. The metrics are diluted by patches the
    system cannot improve. This script gives you the true picture of
    what PyReason achieves on images it can actually affect.
"""

import os
import sys
from collections import Counter, defaultdict

import torch

sys.path.insert(0, os.path.dirname(__file__))

import config
from src.model_loader import build_and_load
from src.ml_inference import build_transform, build_image_index, run_inference
from src.pyreason_engine import compute_slide_vote, run_pyreason, apply_corrections
from src.evaluate import compute_slide_final, evaluate


def _resolve_ckpt() -> str:
    if config.CKPT_PATH:
        return config.CKPT_PATH
    # Auto-find: walk TRAIN_OUTPUT_DIR for newest best_model*.ckpt
    import glob
    pattern = os.path.join(config.TRAIN_OUTPUT_DIR, "**", "best_model", "*.ckpt")
    found = sorted(glob.glob(pattern, recursive=True), reverse=True)
    if found:
        print(f"[Auto] Using checkpoint: {found[0]}")
        return found[0]
    return ""


def _image_type(iname, image_info, slides):
    """Classify one patch as individual / sparse_grid / grid."""
    info = image_info[iname]
    if info["row"] is None:
        return "individual"
    sid = info["slide_id"]
    n_grid_in_slide = sum(
        1 for n in slides.get(sid, [])
        if image_info.get(n, {}).get("row") is not None
    )
    if n_grid_in_slide < config.MIN_PATCHES_FOR_VOTE:
        return "sparse_grid"
    return "grid"


def scan_and_filter(slides_all, image_info_all):
    """
    Print eligibility breakdown, then return filtered (slides, image_info)
    containing only grid-eligible patches.
    """
    counts = Counter(
        _image_type(n, image_info_all, slides_all) for n in image_info_all
    )
    n_total      = len(image_info_all)
    n_individual = counts["individual"]
    n_sparse     = counts["sparse_grid"]
    n_grid       = counts["grid"]

    W = 65
    print("\n" + "=" * W)
    print("  PYREASON ELIGIBILITY SCAN")
    print("=" * W)
    print(f"\n  Total patches found   : {n_total}")
    print(f"  Total slides found    : {len(slides_all)}")
    print()
    print(f"  {'Type':<16}  {'Count':>6}  {'%':>6}  {'Included?':>10}")
    print("  " + "-" * 44)

    rows_info = [
        ("individual",  n_individual, "EXCLUDED — no row_col or 1-patch slide"),
        ("sparse_grid", n_sparse,     f"EXCLUDED — <{config.MIN_PATCHES_FOR_VOTE} patches in slide"),
        ("grid",        n_grid,       "INCLUDED — PyReason eligible"),
    ]
    for tname, cnt, note in rows_info:
        pct  = f"{cnt/n_total*100:.1f}%" if n_total else "0%"
        flag = "YES ✓" if tname == "grid" else "NO ✗"
        print(f"  {tname:<16}  {cnt:>6}  {pct:>6}  {flag:>10}   ({note})")

    print()
    print(f"  Running inference on {n_grid} / {n_total} patches  ({n_grid/n_total*100:.1f}%)")
    print("=" * W)

    if n_grid == 0:
        print("\n[ERROR] No grid-eligible patches found.")
        print("  All images are individual or sparse — cannot run grid-only test.")
        sys.exit(1)

    grid_names     = {n for n in image_info_all
                      if _image_type(n, image_info_all, slides_all) == "grid"}
    new_image_info = {n: image_info_all[n] for n in grid_names}

    new_slides = defaultdict(list)
    for n in grid_names:
        new_slides[image_info_all[n]["slide_id"]].append(n)

    print(f"\n  After filtering:")
    print(f"    Patches : {len(new_image_info)}")
    print(f"    Slides  : {len(new_slides)}")
    n_gt = sum(1 for i in new_image_info.values() if i["gt_label"] is not None)
    print(f"    With GT : {n_gt}")

    return dict(new_slides), new_image_info


def main():
    print("=" * 65)
    print("PYREASON EVALUATION — GRID-ONLY")
    print("(individual + sparse patches excluded from inference)")
    print("=" * 65)

    ckpt_path = _resolve_ckpt()
    if not ckpt_path or not os.path.exists(ckpt_path):
        print("[ERROR] Checkpoint not found.")
        print("  Set CKPT_PATH in config.py or run pytrain/main.py first.")
        sys.exit(1)

    if not os.path.isdir(config.TEST_FOLDER):
        print(f"[ERROR] Test folder not found: {config.TEST_FOLDER}")
        sys.exit(1)

    out_dir = os.path.join(config.OUTPUT_DIR, "grid_only")
    os.makedirs(out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[Device] {device}")

    print(f"\n[Step 1] Loading model...")
    model, num_classes = build_and_load(
        ckpt_path=ckpt_path,
        backbone=config.BACKBONE,
        backbone_weights=config.BACKBONE_WEIGHTS,
        num_classes=None,
    )
    model = model.to(device)
    print(f"  num_classes={num_classes}")

    print(f"\n[Step 2] Indexing test folder: {config.TEST_FOLDER}")
    has_subfolders = any(
        os.path.isdir(os.path.join(config.TEST_FOLDER, sf))
        for sf in (config.LABELLED_SUBFOLDERS or [])
    )
    if has_subfolders:
        subfolder_names = config.LABELLED_SUBFOLDERS
        print("  [Mode] Labelled subfolders found → GT labels available")
    else:
        subfolder_names = None
        print("  [Mode] No subfolders found → flat folder, no GT labels")
        print("  [Mode] PyReason will run; metrics skipped (no ground truth)")

    slides_all, image_info_all = build_image_index(config.TEST_FOLDER, subfolder_names)

    if len(image_info_all) == 0:
        print("[ERROR] No images found. Check TEST_FOLDER in config.py.")
        sys.exit(1)

    slides, image_info = scan_and_filter(slides_all, image_info_all)

    print(f"\n[Step 3] Running ML inference on {len(image_info)} grid patches...")
    transform = build_transform(config.IMG_SIZE)
    raw = run_inference(model, image_info, transform, device, num_classes)

    pred_counts = Counter(p["predicted"] for p in raw.values())
    tier_counts = Counter(p["conf_tier"]  for p in raw.values())
    print(f"  ML predictions : {dict(pred_counts)}")
    print(f"  Confidence     : HIGH={tier_counts['HIGH']}  "
          f"MEDIUM={tier_counts['MEDIUM']}  LOW={tier_counts['LOW']}")

    print(f"\n[Step 4] Computing slide votes...")
    slide_vote = compute_slide_vote(slides, raw)
    n_dom = sum(1 for sv in slide_vote.values() if sv["dominant"])
    print(f"  Dominant slides : {n_dom} / {len(slides)}")

    print(f"\n[Step 5] Running PyReason...")
    corrections = run_pyreason(slides, raw, image_info, slide_vote)

    print(f"\n[Step 6] Applying corrections...")
    final = apply_corrections(raw, corrections)
    n_changed = sum(1 for f in final.values() if f["changed"])
    n_review  = sum(1 for f in final.values() if f["needs_review"])
    print(f"  Changed      : {n_changed}")
    print(f"  Review flags : {n_review}")

    slide_final = compute_slide_final(slides, final, slide_vote)

    print(f"\n[Step 7] Evaluating (grid patches only)...")
    results = evaluate(
        raw=raw, final=final, image_info=image_info,
        slides=slides, slide_vote=slide_vote,
        slide_final=slide_final, output_dir=out_dir,
        image_info_all=image_info_all,
    )

    print(f"\n[Done] Results saved to: {out_dir}")

    if results["has_gt"]:
        pm_ml = results["patch_metrics_ml"]
        pm_pr = results["patch_metrics_pr"]
        delta_f2 = pm_pr.get("f2", 0) - pm_ml.get("f2", 0)
        verdict  = ("IMPROVED" if delta_f2 > 0.001 else
                    "WORSENED" if delta_f2 < -0.001 else "NO CHANGE")
        print(f"\n[Verdict] On {results['n_patches']} grid-eligible patches:")
        print(f"  ML F2       : {pm_ml.get('f2', 0):.4f}")
        print(f"  PyReason F2 : {pm_pr.get('f2', 0):.4f}  ({delta_f2:+.4f})  → {verdict}")
    else:
        print("[Note] No GT labels — metrics not computed.")
        print("  Add labelled subfolders (pos/ neg/ mixed/) to TEST_FOLDER.")


if __name__ == "__main__":
    main()
