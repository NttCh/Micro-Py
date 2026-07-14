"""
run_val_sweep.py
================
Self-contained threshold sweep — no external sweep_thresholds.py needed.
All sweep logic is embedded directly in this file.

Usage:
    python run_val_sweep.py

Reads  : <TRAIN_OUTPUT_DIR>/**/eval/val_patch_results_fold*.csv
Writes : best_thresholds.json        (used by run_test_grid.py)
         val_threshold_sweep.csv     (full ranked results)
         val_patch_results_combined.csv

Set EVAL_DIR below to target a specific training run, or leave as ""
to auto-find the most recent one under config.TRAIN_OUTPUT_DIR.
"""

from __future__ import annotations

import itertools
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

# ── optionally hard-code a specific eval dir ─────────────────────
# Leave as "" to auto-find the most recent training run.
EVAL_DIR = ""


# ─────────────────────────────────────────────────────────────────
#  Sweep grid
# ─────────────────────────────────────────────────────────────────

SWEEP = {
    "SECTION_MODE":                 ["window"],
    "SECTION_MAJORITY_RATIO_THR":   [0.75, 0.80, 0.85],
    "SECTION_MIN_PATCHES":          [3, 4],
    "N_SECTIONS":                   [4],
    "WINDOW_ROWS":                  [3, 5],
    "WINDOW_COLS":                  [5],
    "HIGH_CONF_THR":                [0.85, 0.90],
    "ANCHOR_CONF_THR":              [0.80, 0.85, 0.90],
    "NEIGHBOR_AGREE_MIN":           [3, 4],
    "SECTION_MINORITY_MIN_PATCHES": [3, 4, 5],
    "SECTION_MINORITY_RATIO_THR":   [0.80, 0.85, 0.90],
    "SECTION_MAX_MIXED_FRACTION":   [0.25, 0.30, 0.35, 0.4],
}

BINARY       = {"G", "Gplus"}
UNCERTAIN    = {"Mixed", "Mix", "mix", "mixed"}
LABEL_TO_INT = {"G": 0, "Gplus": 1}

SAFETY_TIER1_ERRORS_RANGE = [1, 2, 3, 4, 5]
SAFETY_WORSENED_RANGE     = [0, 1]
SAFETY_UNFLAGGED_RANGE    = [5, 10, 15]
SAFETY_COMMIT_ACC_RANGE   = [0.80, 0.75]

THRESH_KEYS = [
    "SECTION_MODE", "HIGH_CONF_THR", "ANCHOR_CONF_THR",
    "SECTION_MAJORITY_RATIO_THR", "SECTION_MIN_PATCHES",
    "N_SECTIONS", "WINDOW_ROWS", "WINDOW_COLS",
    "NEIGHBOR_AGREE_MIN", "SECTION_MINORITY_MIN_PATCHES",
    "SECTION_MINORITY_RATIO_THR", "SECTION_MAX_MIXED_FRACTION",
]


# ─────────────────────────────────────────────────────────────────
#  Load / rebuild
# ─────────────────────────────────────────────────────────────────

def rebuild_raw(df: pd.DataFrame) -> Dict:
    return {
        row["image_name"]: {
            "predicted":  str(row["ml_predicted"]),
            "conf":       float(row["ml_conf"]),
            "conf_tier":  "HIGH",
            "prob_G":     float(row.get("ml_prob_G",     0.0)),
            "prob_Gplus": float(row.get("ml_prob_Gplus", 0.0)),
        }
        for _, row in df.iterrows()
    }


def rebuild_structures(df: pd.DataFrame) -> Tuple[Dict, Dict]:
    slides: Dict     = defaultdict(list)
    image_info: Dict = {}
    for _, row in df.iterrows():
        iname = row["image_name"]
        sid   = row["slide_id"]
        slides[sid].append(iname)
        image_info[iname] = {
            "slide_id":  sid,
            "row":       None if pd.isna(row.get("row")) else int(row["row"]),
            "col":       None if pd.isna(row.get("col")) else int(row["col"]),
            "gt_label":  row.get("gt_label"),
            "subfolder": str(row.get("subfolder", "")),
            "filename":  str(row.get("filename",  iname)),
        }
    return dict(slides), image_info


# ─────────────────────────────────────────────────────────────────
#  Metrics
# ─────────────────────────────────────────────────────────────────

def _metrics(y_true: List, y_pred: List) -> Dict:
    firm_pairs: List[Tuple[int, int]] = []
    n_deferred  = 0
    n_binary_gt = 0
    for gt, pred in zip(y_true, y_pred):
        if gt not in BINARY:
            continue
        n_binary_gt += 1
        if pred in BINARY:
            firm_pairs.append((LABEL_TO_INT[gt], LABEL_TO_INT[pred]))
        else:
            n_deferred += 1

    n_firm   = len(firm_pairs)
    coverage = n_firm / n_binary_gt if n_binary_gt > 0 else 0.0

    if not firm_pairs:
        return dict(acc=0, prec=0, rec=0, f1=0, f2=0, mcc=0, bal_acc=0,
                    tp=0, tn=0, fp=0, fn=0, n_firm=0, n_deferred=n_deferred,
                    n_binary_gt=n_binary_gt, coverage=0.0)

    yt = [t for t, p in firm_pairs]
    yp = [p for t, p in firm_pairs]
    tp  = sum(1 for a, b in zip(yt, yp) if a == 1 and b == 1)
    tn  = sum(1 for a, b in zip(yt, yp) if a == 0 and b == 0)
    fp  = sum(1 for a, b in zip(yt, yp) if a == 0 and b == 1)
    fn  = sum(1 for a, b in zip(yt, yp) if a == 1 and b == 0)
    acc     = (tp + tn) / n_firm
    prec    = tp / (tp + fp)            if (tp + fp) > 0    else 0.0
    rec     = tp / (tp + fn)            if (tp + fn) > 0    else 0.0
    f1      = 2*prec*rec / (prec+rec)   if (prec+rec) > 0   else 0.0
    f2      = 5*prec*rec / (4*prec+rec) if (4*prec+rec) > 0 else 0.0
    denom   = ((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)) ** 0.5
    mcc     = (tp*tn - fp*fn) / denom   if denom > 0        else 0.0
    spec    = tn / (tn + fp)            if (tn + fp) > 0    else 0.0
    bal_acc = (rec + spec) / 2
    return dict(acc=acc, prec=prec, rec=rec, f1=f1, f2=f2, mcc=mcc,
                bal_acc=bal_acc, tp=tp, tn=tn, fp=fp, fn=fn,
                n_firm=n_firm, n_deferred=n_deferred,
                n_binary_gt=n_binary_gt, coverage=coverage)


# ─────────────────────────────────────────────────────────────────
#  Core helpers
# ─────────────────────────────────────────────────────────────────

def _apply_conf_tiers(raw: Dict, high_thr: float) -> Dict:
    return {
        k: {**v, "conf_tier": "HIGH" if v["conf"] >= high_thr else "LOW"}
        for k, v in raw.items()
    }


def _build_nb_index(image_info: Dict) -> Dict:
    grid = {}
    for iname, info in image_info.items():
        if info["row"] is not None:
            grid[(info["slide_id"], info["row"], info["col"])] = iname
    nb = {}
    for iname, info in image_info.items():
        if info["row"] is None:
            nb[iname] = []
            continue
        r, c, sid = info["row"], info["col"], info["slide_id"]
        nb[iname] = [
            grid[(sid, r+dr, c+dc)]
            for dr, dc in [(0,1),(0,-1),(1,0),(-1,0)]
            if (sid, r+dr, c+dc) in grid
        ]
    return nb


def _make_vote_entry(
    names: List[str],
    raw: Dict,
    section_min_patches: int,
    section_ratio_thr: float,
    section_minority_min: int,
    section_minority_ratio: float,
    section_max_mixed_frac: float = 0.30,
    gt_labels: Optional[Dict] = None,
) -> Dict:
    def _is_firm(n: str) -> bool:
        if raw[n]["predicted"] in UNCERTAIN:
            return False
        if gt_labels is not None and gt_labels.get(n) in UNCERTAIN:
            return False
        return True

    firm   = [n for n in names if n in raw and _is_firm(n)]
    n_firm = len(firm)
    n_gp   = sum(1 for n in firm if raw[n]["predicted"] == "Gplus")
    n_g    = n_firm - n_gp
    total  = len(names)

    if n_firm == 0:   smaj, smaj_n = "TIE",   0
    elif n_gp > n_g:  smaj, smaj_n = "Gplus", n_gp
    elif n_g > n_gp:  smaj, smaj_n = "G",     n_g
    else:             smaj, smaj_n = "TIE",   n_g

    sratio = smaj_n / n_firm if n_firm > 0 else 0.0
    n_mixed_total = sum(
        1 for n in names
        if n in raw and (
            raw[n]["predicted"] in UNCERTAIN
            or (gt_labels is not None and gt_labels.get(n) in UNCERTAIN)
        )
    )
    mix_frac = n_mixed_total / total if total > 0 else 0.0

    return dict(
        majority=smaj, ratio=sratio,
        dominant=(
            n_firm >= section_min_patches
            and smaj != "TIE"
            and sratio >= section_ratio_thr
            and mix_frac <= section_max_mixed_frac
        ),
        dominant_strict=(
            n_firm >= section_minority_min
            and smaj != "TIE"
            and sratio >= section_minority_ratio
            and mix_frac <= section_max_mixed_frac
        ),
    )


# ─────────────────────────────────────────────────────────────────
#  Single threshold combination run
# ─────────────────────────────────────────────────────────────────

def run_once(
    raw_base: Dict,
    slides: Dict,
    image_info: Dict,
    high_thr: float,
    nb_min: int,
    section_mode: str             = "quadrant",
    section_ratio_thr: float      = 0.70,
    section_min_patches: int      = 2,
    n_sections: int               = 4,
    window_rows: int              = 4,
    window_cols: int              = 3,
    anchor_conf_thr: float        = 0.85,
    section_minority_min: int     = 3,
    section_minority_ratio: float = 0.85,
    section_max_mixed_frac: float = 0.30,
) -> Dict:
    raw        = _apply_conf_tiers(raw_base, high_thr)
    nb         = _build_nb_index(image_info)
    corr: Dict   = {}
    flagged: set = set()
    gt_labels    = {n: image_info[n].get("gt_label") for n in image_info}

    if section_mode == "window":
        grid: Dict = {}
        for n, info in image_info.items():
            if info.get("row") is not None:
                grid[(info["slide_id"], info["row"], info["col"])] = n
        hr = window_rows // 2
        hc = window_cols // 2
        ctx_vote: Dict = {}
        for n, info in image_info.items():
            r, c, sid = info.get("row"), info.get("col"), info.get("slide_id")
            if r is None or c is None:
                ctx_vote[n] = dict(majority="TIE", ratio=0.0,
                                   dominant=False, dominant_strict=False)
                continue
            win_names = [
                grid[(sid, r+dr, c+dc)]
                for dr in range(-hr, window_rows - hr)
                for dc in range(-hc, window_cols - hc)
                if (sid, r+dr, c+dc) in grid
            ]
            ctx_vote[n] = _make_vote_entry(
                win_names, raw, section_min_patches, section_ratio_thr,
                section_minority_min, section_minority_ratio,
                section_max_mixed_frac=section_max_mixed_frac,
                gt_labels=gt_labels,
            )
        def _get_ctx(n: str) -> Dict:
            return ctx_vote.get(n, {})

    else:
        sq = max(1, int(n_sections ** 0.5))
        sec_assignments: Dict = {}
        sec_vote_cache: Dict  = {}
        for sid, names in slides.items():
            grid_names = [
                n for n in names
                if n in raw and image_info.get(n, {}).get("row") is not None
            ]
            if not grid_names:
                continue
            rows_ = [image_info[n]["row"] for n in grid_names]
            cols_ = [image_info[n]["col"] for n in grid_names]
            rmin_, rmax_ = min(rows_), max(rows_)
            cmin_, cmax_ = min(cols_), max(cols_)
            row_span_ = max(rmax_ - rmin_, 1)
            col_span_ = max(cmax_ - cmin_, 1)
            by_sec: Dict = {}
            for n in grid_names:
                r_ = image_info[n]["row"]
                c_ = image_info[n]["col"]
                rb = min(int((r_ - rmin_) / row_span_ * sq), sq - 1)
                cb = min(int((c_ - cmin_) / col_span_ * sq), sq - 1)
                sec = rb * sq + cb
                sec_assignments[n] = sec
                by_sec.setdefault(sec, []).append(n)
            for sec_id, sec_names in by_sec.items():
                sec_vote_cache[(sid, sec_id)] = _make_vote_entry(
                    sec_names, raw, section_min_patches, section_ratio_thr,
                    section_minority_min, section_minority_ratio,
                    section_max_mixed_frac=section_max_mixed_frac,
                    gt_labels=gt_labels,
                )
        def _get_ctx(n: str) -> Dict:  # type: ignore[misc]
            sid    = image_info.get(n, {}).get("slide_id")
            sec_id = sec_assignments.get(n, -1)
            if sid is None or sec_id == -1:
                return {}
            return sec_vote_cache.get((sid, sec_id), {})

    n_r1a_fired = 0
    n_r1b_fired = 0
    for n, pred in raw.items():
        if n in corr:
            continue
        if pred["conf"] >= anchor_conf_thr and pred["predicted"] not in UNCERTAIN:
            continue
        is_unc = pred["predicted"] in UNCERTAIN or pred["conf_tier"] == "LOW"
        if not is_unc:
            continue
        ctx = _get_ctx(n)
        if not ctx or not ctx.get("dominant", False):
            continue
        ctx_maj = ctx["majority"]
        if ctx_maj == "TIE":
            continue
        if pred["predicted"] not in UNCERTAIN and pred["predicted"] == ctx_maj:
            continue
        if pred["predicted"] not in UNCERTAIN:
            if not ctx.get("dominant_strict", False):
                continue
            corr[n] = ctx_maj
            n_r1b_fired += 1
        else:
            corr[n] = ctx_maj
            n_r1a_fired += 1

    for n, pred in raw.items():
        if n in corr:
            continue
        if pred["predicted"] not in UNCERTAIN and pred["conf_tier"] != "LOW":
            continue
        for target in ("Gplus", "G"):
            agree = [
                nb2 for nb2 in nb.get(n, [])
                if nb2 in raw
                and raw[nb2]["predicted"] == target
                and raw[nb2]["conf_tier"] == "HIGH"
            ]
            if len(agree) >= nb_min:
                corr[n] = target
                break

    for n, pred in raw.items():
        if (pred["conf_tier"] == "LOW" or pred["predicted"] in UNCERTAIN) and n not in corr:
            flagged.add(n)

    final_preds = {n: corr.get(n, raw[n]["predicted"]) for n in raw}

    pairs = [(gt_labels[n], final_preds[n]) for n in raw if gt_labels[n] is not None]
    m = _metrics([p[0] for p in pairs], [p[1] for p in pairs])

    m["n_changed"] = sum(1 for n in raw if corr.get(n) and corr[n] != raw[n]["predicted"])
    m["n_flagged"] = len(flagged)
    m["n_r1a"]     = n_r1a_fired
    m["n_r1b"]     = n_r1b_fired

    n_worsened = 0
    n_worsened_near_mixed = 0
    for n in raw:
        gt = gt_labels.get(n)
        if gt is None or gt not in BINARY:
            continue
        if raw[n]["predicted"] == gt and final_preds[n] != gt:
            n_worsened += 1
            if any(gt_labels.get(nb_n) in UNCERTAIN for nb_n in nb.get(n, [])):
                n_worsened_near_mixed += 1
    m["n_worsened"]            = n_worsened
    m["n_worsened_near_mixed"] = n_worsened_near_mixed

    n_unflagged = 0
    for n in raw:
        gt = gt_labels.get(n)
        if gt is None:
            continue
        pr = final_preds[n]
        if pr != gt and pr in BINARY and n not in flagged:
            n_unflagged += 1
    m["n_unflagged"] = n_unflagged

    n_commits = 0
    n_commit_errors = 0
    for n in raw:
        gt = gt_labels.get(n)
        if gt is None or gt not in BINARY:
            continue
        if raw[n]["predicted"] in UNCERTAIN and final_preds[n] in BINARY:
            n_commits += 1
            if final_preds[n] != gt:
                n_commit_errors += 1
    m["n_commits"]       = n_commits
    m["n_commit_errors"] = n_commit_errors
    m["commit_accuracy"] = (n_commits - n_commit_errors) / n_commits if n_commits > 0 else 1.0

    n_tier1_errors = 0
    for n in raw:
        gt = gt_labels.get(n)
        if gt is None or gt not in BINARY:
            continue
        if final_preds[n] == gt or final_preds[n] not in BINARY:
            continue
        if (raw[n]["conf"] >= high_thr
                and n not in corr
                and n not in flagged
                and raw[n]["predicted"] not in UNCERTAIN):
            n_tier1_errors += 1
    m["n_tier1_errors"] = n_tier1_errors

    return m


# ─────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────

def find_latest_eval_dir(train_output_dir: str) -> str:
    candidates = sorted(
        Path(train_output_dir).rglob("val_patch_results_fold1.csv"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(
            f"No val_patch_results_fold1.csv found under {train_output_dir}\n"
            "Make sure training has completed and the val fold CSVs were written."
        )
    return str(candidates[0].parent)


def main():
    eval_dir = EVAL_DIR.strip() or find_latest_eval_dir(config.TRAIN_OUTPUT_DIR)
    print(f"[Sweep] Eval dir : {eval_dir}")

    fold_csvs = sorted(Path(eval_dir).glob("val_patch_results_fold*.csv"))
    if not fold_csvs:
        print("[Sweep] No val_patch_results_fold*.csv found. Aborting.")
        sys.exit(1)

    dfs      = [pd.read_csv(p) for p in fold_csvs]
    combined = pd.concat(dfs, ignore_index=True)
    n_grid   = int(combined["row"].notna().sum())
    print(f"[Sweep] Loaded {len(fold_csvs)} fold(s): {len(combined)} patches, {n_grid} grid-eligible.")

    if n_grid == 0:
        print(
            "[Sweep] No grid patches found.\n"
            "  Filenames need pattern <slide>_<row>_<col>.jpg (e.g. 000043_2_2.jpg).\n"
            "  Aborting."
        )
        sys.exit(1)

    combined.to_csv(os.path.join(eval_dir, "val_patch_results_combined.csv"), index=False)

    raw_base           = rebuild_raw(combined)
    slides, image_info = rebuild_structures(combined)

    pairs = [
        (image_info[n].get("gt_label"), raw_base[n]["predicted"])
        for n in raw_base if image_info[n].get("gt_label") is not None
    ]
    if not pairs:
        print("[Sweep] No GT labels in val data. Aborting.")
        sys.exit(1)

    baseline = _metrics([p[0] for p in pairs], [p[1] for p in pairs])
    print(
        f"[Sweep] Baseline  MCC={baseline['mcc']:.4f}  "
        f"Coverage={baseline['coverage']:.1%}  Deferred={baseline['n_deferred']}"
    )

    keys   = list(SWEEP.keys())
    combos = list(itertools.product(*SWEEP.values()))
    print(f"[Sweep] Testing {len(combos)} combinations ...")

    rows_out = []
    n_errors = 0
    for i, combo in enumerate(combos, 1):
        if i % 200 == 0 or i == len(combos):
            print(f"  {i}/{len(combos)} ...", end="\r")
        params = dict(zip(keys, combo))
        try:
            m = run_once(
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
        except Exception as e:
            n_errors += 1
            if n_errors == 1:
                print(f"\n[Sweep] run_once error (first shown): {e}")
            continue
        rows_out.append({**params, **m, "delta_mcc": m["mcc"] - baseline["mcc"]})

    print()
    if not rows_out:
        print(f"[Sweep] All {len(combos)} combinations failed. Aborting.")
        sys.exit(1)

    print(f"[Sweep] {len(rows_out)} succeeded, {n_errors} failed.")

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

    safe_df = pd.DataFrame()
    found   = False
    for t1e in SAFETY_TIER1_ERRORS_RANGE:
        g1 = result_df[result_df["n_tier1_errors"] <= t1e]
        if not len(g1): continue
        for wor in SAFETY_WORSENED_RANGE:
            g2 = g1[g1["n_worsened"] <= wor]
            if not len(g2): continue
            for unflg in SAFETY_UNFLAGGED_RANGE:
                g3 = g2[g2["n_unflagged"] <= unflg]
                if not len(g3): continue
                for cmt in SAFETY_COMMIT_ACC_RANGE:
                    g4 = g3[g3["commit_accuracy"] >= cmt].copy()
                    if not len(g4): continue
                    safe_df = g4; found = True; break
                if found: break
            if found: break
        if found: break

    if found:
        best_row = safe_df.iloc[0]
        print(
            f"[Sweep] Safe config found  "
            f"tier1_errors={int(best_row['n_tier1_errors'])}  "
            f"worsened={int(best_row['n_worsened'])}  "
            f"unflagged={int(best_row['n_unflagged'])}"
        )
    else:
        best_row = result_df.iloc[0]
        print("[Sweep] No safe config found — using best overall.")

    sweep_csv = os.path.join(eval_dir, "val_threshold_sweep.csv")
    result_df.to_csv(sweep_csv, index=False)
    print(f"[Sweep] Full results  → {sweep_csv}")

    best_thresholds: Dict = {}
    for k in THRESH_KEYS:
        v = best_row.get(k)
        if v is not None:
            best_thresholds[k] = v.item() if hasattr(v, "item") else v

    best_thresholds["_val_mcc"]        = float(best_row["mcc"])
    best_thresholds["_val_n_worsened"] = int(best_row["n_worsened"])
    best_thresholds["_baseline_mcc"]   = float(baseline["mcc"])
    best_thresholds["_delta_mcc"]      = float(best_row["mcc"] - baseline["mcc"])
    best_thresholds["_n_folds_used"]   = len(fold_csvs)
    best_thresholds["_n_grid_patches"] = n_grid

    json_path = os.path.join(eval_dir, "best_thresholds.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(best_thresholds, f, indent=2, ensure_ascii=False)

    print(
        f"\n[Sweep] ✓ best_thresholds.json → {json_path}\n"
        f"\n  Best config:"
        f"\n    MCC          = {best_thresholds['_val_mcc']:.4f}"
        f"  (Δ {best_thresholds['_delta_mcc']:+.4f} vs baseline {baseline['mcc']:.4f})"
        f"\n    worsened     = {best_thresholds['_val_n_worsened']}"
        f"\n    HIGH_CONF    = {best_thresholds['HIGH_CONF_THR']}"
        f"\n    ANCHOR       = {best_thresholds['ANCHOR_CONF_THR']}"
        f"\n    SECTION_MODE = {best_thresholds['SECTION_MODE']}"
        f"\n    MAJ_RATIO    = {best_thresholds['SECTION_MAJORITY_RATIO_THR']}"
        f"\n    MINOR_RATIO  = {best_thresholds['SECTION_MINORITY_RATIO_THR']}"
        f"\n    MAX_MIXED    = {best_thresholds['SECTION_MAX_MIXED_FRACTION']}"
        f"\n    WINDOW       = {best_thresholds['WINDOW_ROWS']}x{best_thresholds['WINDOW_COLS']}"
        f"\n\nNow run:  python run_test_grid.py"
    )


if __name__ == "__main__":
    main()