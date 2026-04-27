"""
src/evaluate.py
===============
Compare ML-only vs PyReason predictions against ground truth.

Saves: results/evaluation_results.xlsx

KEY DESIGN DECISIONS (medically sound evaluation):
─────────────────────────────────────────────────
  Problem: CNN is trained on 3 classes (G / Gplus / Mixed) but test set has
  no Mixed GT folder.  When the model outputs "Mixed" on a G or Gplus patch
  it is expressing *uncertainty*, NOT making a wrong binary prediction.
  Treating that as wrong (mapping to opposite class) is statistically invalid
  and penalises the system for being appropriately cautious.

  Solution — 3-zone evaluation:
    Zone 1  FIRM predictions   (ML or PR output is G or Gplus)
            → binary metrics (MCC, BalAcc, F1, F2, Recall, Precision)
            → these are the primary diagnostic performance numbers
    Zone 2  DEFERRED predictions (ML or PR output is Mixed on G/Gplus GT)
            → deferral rate, coverage %
            → a 10-15% deferral with good triage quality is clinically good
    Zone 3  Mixed GT patches   (only present in training set)
            → screening / flag-rate metrics; skipped gracefully if no Mixed GT

  Slide-level: a slide whose majority post-PyReason prediction is Mixed
  → diagnosis = "REFER" (send to microbiologist), not forced G or Gplus.

Excel sheets (in order):
  Summary          — Table A (firm metrics) + Table B (deferral/safety)
                     + confusion matrix + rule breakdown + correction quality
  Coverage         — image type breakdown, PyReason eligibility,
                     Rule 4 flag quality, unflagged silent failures
  Patch Results    — every patch with all columns
  Slide Results    — every slide diagnosis (includes REFER)
  Changed Patches  — only corrected patches
  Review Flags     — Rule 4 flagged patches
  Errors Remaining — still-wrong patches after PyReason (not flagged)
  By Class         — per GT class breakdown
  By Slide Combo   — breakdown by slide label combination
  Stratified       — paper-ready table: neg / pos / pos+neg subsets
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

import config


# ─────────────────────────────────────────────────────────────────
#  Constants
# ─────────────────────────────────────────────────────────────────

BINARY_LABELS  = {"G", "Gplus"}
UNCERTAIN_LABELS = {"Mixed", "Mix", "mix", "mixed"}


# ─────────────────────────────────────────────────────────────────
#  Slide final  (with REFER support)
# ─────────────────────────────────────────────────────────────────

def compute_slide_final(slides: Dict, final: Dict, slide_vote: Dict) -> Dict:
    """
    Compute per-slide diagnosis after PyReason.

    diagnosis values:
      "Gplus"  — majority of final predictions are Gplus
      "G"      — majority of final predictions are G
      "REFER"  — majority of final predictions are Mixed (uncertain → refer)
      "TIE"    — equal Gplus and G, no Mixed majority
    """
    slide_final: Dict = {}
    for sid, names in slides.items():
        fps = [final[n] for n in names if n in final]
        if not fps:
            continue
        n_g   = sum(1 for p in fps if p["final_predicted"] == "G")
        n_gp  = sum(1 for p in fps if p["final_predicted"] == "Gplus")
        n_mx  = sum(1 for p in fps if p["final_predicted"] in UNCERTAIN_LABELS)
        n_ch  = sum(1 for p in fps if p["changed"])
        n_rv  = sum(1 for p in fps if p["needs_review"])

        # REFER if Mixed is the plurality — clinically safe outcome
        if n_mx > n_g and n_mx > n_gp:
            diagnosis = "REFER"
        elif n_gp > n_g:
            diagnosis = "Gplus"
        elif n_g > n_gp:
            diagnosis = "G"
        else:
            diagnosis = "TIE"

        gt_vals  = [final[n]["gt_label"] for n in names
                    if n in final and final[n].get("gt_label") is not None]
        slide_gt = max(set(gt_vals), key=gt_vals.count) if gt_vals else None

        slide_final[sid] = dict(
            diagnosis=diagnosis, slide_gt=slide_gt,
            n_G=n_g, n_Gplus=n_gp, n_Mixed=n_mx, total=len(fps),
            n_changed=n_ch, n_needs_review=n_rv,
            ratio=slide_vote[sid]["ratio"],
            dominant=slide_vote[sid]["dominant"],
        )
    return slide_final


# ─────────────────────────────────────────────────────────────────
#  Metrics helpers
# ─────────────────────────────────────────────────────────────────

def _label_to_int(label: Optional[str]) -> Optional[int]:
    if label is None:
        return None
    return config.LABEL_MAP_INV.get(label)


def _cls_metrics(y_true: List[int], y_pred: List[int]) -> Dict:
    """
    Binary classification metrics (G=0 vs Gplus=1).
    Only call this on FIRM predictions — i.e. GT in {G, Gplus} AND
    predicted in {G, Gplus}.  Do NOT pass deferred (Mixed) predictions here.
    """
    if not y_true:
        return dict(acc=0, precision=0, recall=0, f1=0, f2=0,
                    mcc=0, bal_acc=0, tp=0, tn=0, fp=0, fn=0, n=0)
    yt = np.array(y_true)
    yp = np.array(y_pred)
    n   = len(yt)
    acc = float((yt == yp).sum() / n)
    tp  = int(((yp == 1) & (yt == 1)).sum())
    tn  = int(((yp == 0) & (yt == 0)).sum())
    fp  = int(((yp == 1) & (yt == 0)).sum())
    fn  = int(((yp == 0) & (yt == 1)).sum())
    prec    = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec     = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1      = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    f2      = (5 * prec * rec) / (4 * prec + rec) if (4 * prec + rec) > 0 else 0.0
    denom   = ((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)) ** 0.5
    mcc     = (tp*tn - fp*fn) / denom if denom > 0 else 0.0
    spec    = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    bal_acc = (rec + spec) / 2
    return dict(acc=acc, precision=prec, recall=rec, f1=f1, f2=f2,
                mcc=mcc, bal_acc=bal_acc,
                tp=tp, tn=tn, fp=fp, fn=fn, n=n)


def _compute_deferral_stats(
    rows_binary: List[Dict],
    pred_key_ml: str = "ml_predicted",
    pred_key_pr: str = "final_predicted",
) -> Dict:
    """
    Compute deferral (Mixed prediction) rates for patches whose GT is binary.

    Returns dict with:
      n_total           total G/Gplus GT patches
      n_ml_firm         ML made a firm (G or Gplus) prediction
      n_ml_deferred     ML predicted Mixed on G/Gplus GT patch
      ml_deferral_rate  n_ml_deferred / n_total
      ml_coverage       n_ml_firm / n_total
      n_pr_firm         PR firm predictions
      n_pr_deferred     PR deferred (Mixed) on G/Gplus GT patches
      pr_deferral_rate
      pr_coverage
    """
    n = len(rows_binary)
    if n == 0:
        return dict(n_total=0, n_ml_firm=0, n_ml_deferred=0,
                    ml_deferral_rate=0.0, ml_coverage=0.0,
                    n_pr_firm=0, n_pr_deferred=0,
                    pr_deferral_rate=0.0, pr_coverage=0.0)

    n_ml_def = sum(1 for r in rows_binary if r[pred_key_ml] in UNCERTAIN_LABELS)
    n_pr_def = sum(1 for r in rows_binary if r[pred_key_pr] in UNCERTAIN_LABELS)
    n_ml_firm = n - n_ml_def
    n_pr_firm = n - n_pr_def

    return dict(
        n_total=n,
        n_ml_firm=n_ml_firm,
        n_ml_deferred=n_ml_def,
        ml_deferral_rate=n_ml_def / n,
        ml_coverage=n_ml_firm / n,
        n_pr_firm=n_pr_firm,
        n_pr_deferred=n_pr_def,
        pr_deferral_rate=n_pr_def / n,
        pr_coverage=n_pr_firm / n,
    )


# ─────────────────────────────────────────────────────────────────
#  Fair comparison helper
# ─────────────────────────────────────────────────────────────────

def _compute_fair_comparisons(rows: List[Dict]) -> Dict:
    """
    Compute three valid comparison scenarios that address the
    denominator problem (ML N≠PR N when PR resolves deferred patches).

    The reported Table A compares ML on N=ml_firm vs PR on N=pr_firm.
    When PR resolves some Mixed patches to firm, the denominators differ —
    PR is tested on strictly harder/larger set, making naive comparison unfair.

    Scenario A — same denominator, ML-firm patches only:
        Restrict BOTH ML and PR to the exact patches ML was firm on.
        ML N = PR N = n_ml_firm. Apples-to-apples accuracy comparison.
        Answers: "Did PyReason harm or help on ML's committed patches?"

    Scenario B — full population, forced-answer (old metric):
        Treat every ML-deferred patch as wrong (Mixed → opposite class).
        All 807 patches scored for both ML and PR.
        Answers: "How much does PyReason's coverage gain help overall
                  if we penalise refusing to answer?"

    Scenario C — extra patches PR committed to (coverage quality):
        Only the patches ML deferred but PR converted to firm.
        PR accuracy on these 28 extra patches = quality of coverage gain.
        Answers: "Were the extra commits good decisions?"

    Returns dict with keys scenario_A, scenario_B, scenario_C,
    each containing ml/pr metrics plus metadata.
    """
    BINARY = BINARY_LABELS
    LABEL  = {"G": 0, "Gplus": 1}

    def _metrics_from_pairs(pairs):
        if not pairs:
            return _cls_metrics([], [])
        yt = [p[0] for p in pairs]
        yp = [p[1] for p in pairs]
        return _cls_metrics(yt, yp)

    binary_rows = [r for r in rows if r.get("gt_label") in BINARY]
    if not binary_rows:
        return {}

    # Which patches ML was firm on
    ml_firm_names = {r["image_name"] for r in binary_rows
                     if r.get("ml_predicted") in BINARY}

    # ── Scenario A: same N = ML-firm patches ──────────────────────
    ml_firm_rows = [r for r in binary_rows if r["image_name"] in ml_firm_names]
    # ML pairs: all firm (by definition)
    ml_a_pairs = [(LABEL[r["gt_label"]], LABEL[r["ml_predicted"]])
                  for r in ml_firm_rows if r["ml_predicted"] in BINARY]
    # PR pairs on same set: some may now be deferred (PR changed to Mixed — rare)
    pr_a_firm   = [r for r in ml_firm_rows if r.get("final_predicted") in BINARY]
    pr_a_pairs  = [(LABEL[r["gt_label"]], LABEL[r["final_predicted"]])
                   for r in pr_a_firm if r["final_predicted"] in BINARY]
    n_pr_a_deferred = len(ml_firm_rows) - len(pr_a_firm)

    m_ml_a = _metrics_from_pairs(ml_a_pairs)
    m_pr_a = _metrics_from_pairs(pr_a_pairs)

    # ── Scenario B: all 807, Mixed = forced wrong ──────────────────
    ml_b_pairs = []
    pr_b_pairs = []
    for r in binary_rows:
        gt = LABEL.get(r.get("gt_label"))
        if gt is None:
            continue
        ml = r.get("ml_predicted")
        pr = r.get("final_predicted")
        ml_b_pairs.append((gt, LABEL[ml] if ml in BINARY else 1 - gt))
        pr_b_pairs.append((gt, LABEL[pr] if pr in BINARY else 1 - gt))

    m_ml_b = _metrics_from_pairs(ml_b_pairs)
    m_pr_b = _metrics_from_pairs(pr_b_pairs)

    # ── Scenario C: the extra 28 patches PR committed to ──────────
    extra_rows = [r for r in binary_rows
                  if r["image_name"] not in ml_firm_names   # ML deferred
                  and r.get("final_predicted") in BINARY]   # PR committed
    extra_correct  = sum(1 for r in extra_rows
                         if r.get("pr_correct") is True)
    extra_wrong    = len(extra_rows) - extra_correct
    extra_accuracy = extra_correct / len(extra_rows) if extra_rows else 0.0
    extra_pairs    = [(LABEL[r["gt_label"]], LABEL[r["final_predicted"]])
                      for r in extra_rows if r["final_predicted"] in BINARY]
    m_extra = _metrics_from_pairs(extra_pairs)

    return dict(
        scenario_A=dict(
            description="Same N as ML — apples-to-apples on ML-firm patches",
            n=len(ml_a_pairs),
            n_pr_deferred_on_ml_set=n_pr_a_deferred,
            ml=m_ml_a, pr=m_pr_a,
            delta_mcc=m_pr_a.get("mcc", 0) - m_ml_a.get("mcc", 0),
            delta_recall=m_pr_a.get("recall", 0) - m_ml_a.get("recall", 0),
        ),
        scenario_B=dict(
            description="All patches, Mixed=wrong — full population forced answer",
            n=len(ml_b_pairs),
            ml=m_ml_b, pr=m_pr_b,
            delta_mcc=m_pr_b.get("mcc", 0) - m_ml_b.get("mcc", 0),
            delta_recall=m_pr_b.get("recall", 0) - m_ml_b.get("recall", 0),
        ),
        scenario_C=dict(
            description="Extra patches PR committed (ML deferred, PR firm)",
            n=len(extra_rows),
            n_correct=extra_correct,
            n_wrong=extra_wrong,
            accuracy=extra_accuracy,
            metrics=m_extra,
        ),
    )


# ─────────────────────────────────────────────────────────────────
#  Main evaluate
# ─────────────────────────────────────────────────────────────────

def evaluate(
    raw: Dict, final: Dict, image_info: Dict,
    slides: Dict, slide_vote: Dict, slide_final: Dict,
    output_dir: str,
    image_info_all: Dict = None,
) -> Dict:
    os.makedirs(output_dir, exist_ok=True)

    for iname, fp in final.items():
        fp["gt_label"] = image_info[iname].get("gt_label")

    has_gt = any(fp.get("gt_label") is not None for fp in final.values())

    # ── Patch DataFrame ──────────────────────────────────────────
    rows = []
    for iname, fp in final.items():
        info   = image_info[iname]
        sv     = slide_vote.get(info["slide_id"], {})
        ml_pred = raw[iname]["predicted"]
        pr_pred = fp["final_predicted"]
        gt      = fp.get("gt_label")

        # correctness: only valid when gt is known
        ml_ok = (ml_pred == gt) if gt else None
        pr_ok = (pr_pred == gt) if gt else None

        has_grid = (info["row"] is not None)
        n_patches_in_slide = len(slides.get(info["slide_id"], []))

        if not has_grid:
            img_type = "individual"
        elif n_patches_in_slide == 1:
            img_type = "individual"
        elif n_patches_in_slide < config.MIN_PATCHES_FOR_VOTE:
            img_type = "sparse_grid"
        else:
            img_type = "grid"

        # Prediction zone — used in deferral analysis
        ml_zone = "firm"     if ml_pred in BINARY_LABELS else "deferred"
        pr_zone = "firm"     if pr_pred in BINARY_LABELS else "deferred"

        rows.append({
            "image_name":      iname,
            "slide_id":        info["slide_id"],
            "subfolder":       info["subfolder"],
            "filename":        info["filename"],
            "row":             info["row"],
            "col":             info["col"],
            "image_type":      img_type,
            "gt_label":        gt,
            "ml_predicted":    ml_pred,
            "ml_zone":         ml_zone,
            "ml_prob_G":       round(raw[iname]["prob_G"],     4),
            "ml_prob_Gplus":   round(raw[iname]["prob_Gplus"], 4),
            "ml_prob_Mixed":   round(max(0.0, 1.0 - raw[iname]["prob_G"] - raw[iname]["prob_Gplus"]), 4),
            "ml_conf":         round(raw[iname]["conf"],       4),
            "ml_conf_tier":    raw[iname]["conf_tier"],
            "ml_correct":      ml_ok,
            "final_predicted": pr_pred,
            "pr_zone":         pr_zone,
            "rule_applied":    fp["rule_applied"],
            "changed":         fp["changed"],
            "needs_review":    fp["needs_review"],
            "pr_correct":      pr_ok,
            "outcome": (
                "improved"     if fp["changed"] and not ml_ok and pr_ok else
                "worsened"     if fp["changed"] and ml_ok and not pr_ok else
                "no_change"    if not fp["changed"] else
                "changed_same"
            ) if gt else "unknown",
            "slide_majority":  sv.get("majority"),
            "slide_ratio":     round(sv.get("ratio", 0), 4),
            "slide_dominant":  sv.get("dominant"),
            "slide_standalone":sv.get("is_standalone", False),
            "slide_sparse":    sv.get("is_sparse", False),
        })

    patch_df = pd.DataFrame(rows)

    # ── Triage tier assignment ────────────────────────────────────
    # Tier 1 Confirmed : HIGH-conf, not changed, not needs_review
    # Tier 2 Suggested : changed=True OR (conf >= MEDIUM_THR AND needs_review)
    # Tier 3 Refer     : needs_review AND not changed AND (conf < MEDIUM_THR OR Mixed)
    _MEDIUM_THR = float(getattr(config, "MEDIUM_CONF_THR", 0.75))
    _HIGH_THR   = float(getattr(config, "HIGH_CONF_THR",   0.85))

    def _assign_tier(row):
        _changed  = bool(row["changed"])
        _rev      = bool(row["needs_review"])
        _conf     = float(row["ml_conf"])
        _is_mixed = str(row["ml_predicted"]) in UNCERTAIN_LABELS
        if _changed:
            return 2
        if _rev and (_conf < _MEDIUM_THR or _is_mixed):
            return 3
        if _rev and _conf >= _MEDIUM_THR:
            return 2
        if _conf >= _HIGH_THR and not _is_mixed:
            return 1
        return 2

    patch_df["triage_tier"]  = patch_df.apply(_assign_tier, axis=1)
    patch_df["triage_label"] = patch_df["triage_tier"].map(
        {1: "Confirmed", 2: "Suggested", 3: "Refer"}
    )
    _tc     = patch_df["triage_tier"].value_counts().sort_index()
    n_tier1 = int(_tc.get(1, 0))
    n_tier2 = int(_tc.get(2, 0))
    n_tier3 = int(_tc.get(3, 0))


    # ── Patch metrics — 3-zone approach ──────────────────────────
    # pm_ml / pm_pr   : FIRM predictions only (Zone 1)
    # deferral_stats  : deferral rates (Zone 2)
    # mixed_stats     : Mixed GT screening quality (Zone 3, train-only usually)
    pm_ml, pm_pr       = {}, {}
    deferral_stats_ml  = {}
    deferral_stats_pr  = {}
    mixed_stats        = {}

    if has_gt:
        # Patches with binary GT (G or Gplus) — the diagnostic population
        binary_rows = [r for r in rows if r["gt_label"] in BINARY_LABELS]

        # ── Zone 2: deferral stats (over ALL binary GT patches) ───
        deferral_stats = _compute_deferral_stats(binary_rows)

        # ── Zone 1: firm metrics (GT binary AND prediction binary) ─
        # ML firm: GT ∈ {G,Gplus} AND ml_predicted ∈ {G,Gplus}
        ml_firm_rows = [r for r in binary_rows if r["ml_predicted"] in BINARY_LABELS]
        if ml_firm_rows:
            gt_i  = [_label_to_int(r["gt_label"])    for r in ml_firm_rows]
            ml_i  = [_label_to_int(r["ml_predicted"]) for r in ml_firm_rows]
            pm_ml = _cls_metrics(gt_i, ml_i)

        # PR firm: GT ∈ {G,Gplus} AND final_predicted ∈ {G,Gplus}
        pr_firm_rows = [r for r in binary_rows if r["final_predicted"] in BINARY_LABELS]
        if pr_firm_rows:
            gt_i  = [_label_to_int(r["gt_label"])     for r in pr_firm_rows]
            pr_i  = [_label_to_int(r["final_predicted"]) for r in pr_firm_rows]
            pm_pr = _cls_metrics(gt_i, pr_i)

        # ── Fair comparison scenarios (denominator analysis) ─────────
        fair_comparisons = _compute_fair_comparisons(rows)

        # ── Zone 3: Mixed GT patches (training eval; often empty at test) ─
        UNCERTAIN_GT  = UNCERTAIN_LABELS
        info_source   = image_info_all if image_info_all else image_info
        all_mixed_gt  = [n for n, info in info_source.items()
                         if info.get("gt_label") in UNCERTAIN_GT]
        if all_mixed_gt:
            n_mx = len(all_mixed_gt)
            n_ml_flag       = sum(1 for n in all_mixed_gt if n in raw
                                  and raw[n]["conf_tier"] == "LOW")
            n_pr_flag       = sum(1 for n in all_mixed_gt if n in final
                                  and final[n].get("needs_review", False))
            n_ml_mixed_pred = sum(1 for n in all_mixed_gt if n in raw
                                  and raw[n]["predicted"] in UNCERTAIN_GT)
            n_changed_mx    = sum(1 for n in all_mixed_gt if n in final
                                  and final[n].get("changed", False))
            mixed_stats = dict(
                n_mixed=n_mx,
                n_ml_predicted_mixed=n_ml_mixed_pred,
                n_flagged_ml=n_ml_flag,
                n_flagged_pr=n_pr_flag,
                n_changed=n_changed_mx,
                flag_rate_ml=n_ml_flag / n_mx,
                flag_rate_pr=n_pr_flag / n_mx,
                ml_mixed_pred_rate=n_ml_mixed_pred / n_mx,
            )
    else:
        deferral_stats    = {}
        fair_comparisons  = {}

    # ── Slide DataFrame ──────────────────────────────────────────
    slide_rows = []
    for sid, sf in slide_final.items():
        sv    = slide_vote[sid]
        ml_dg = sv["majority"] if sv["dominant"] else "UNCERTAIN"
        pr_dg = sf["diagnosis"]   # may now be "REFER"
        gt_dg = sf["slide_gt"]

        # For correctness: REFER counts as wrong (patient needs definitive answer)
        # but is flagged separately so clinicians know the reason
        ml_ok = (ml_dg == gt_dg) if gt_dg and ml_dg not in ("UNCERTAIN",) else None
        pr_ok = (pr_dg == gt_dg) if gt_dg and pr_dg not in ("REFER", "TIE") else None
        pr_referred = (pr_dg == "REFER")

        slide_rows.append({
            "slide_id":       sid,
            "gt_label":       gt_dg,
            "ml_diagnosis":   ml_dg,
            "pr_diagnosis":   pr_dg,
            "pr_referred":    pr_referred,
            "ml_correct":     ml_ok,
            "pr_correct":     pr_ok,
            "outcome": (
                "referred"  if pr_referred else
                "improved"  if ml_ok is not None and not ml_ok and pr_ok else
                "worsened"  if ml_ok and pr_ok is not None and not pr_ok else
                "correct"   if ml_ok and pr_ok else
                "wrong"     if ml_ok is not None and not ml_ok and
                               pr_ok is not None and not pr_ok else "no_gt"
            ),
            "n_patches":      sf["total"],
            "n_G_final":      sf["n_G"],
            "n_Gplus_final":  sf["n_Gplus"],
            "n_Mixed_final":  sf.get("n_Mixed", 0),
            "n_changed":      sf["n_changed"],
            "n_needs_review": sf["n_needs_review"],
            "majority_ratio": round(sf["ratio"], 4),
            "dominant":       sf["dominant"],
        })
    slide_df = pd.DataFrame(slide_rows)

    # ── Slide metrics (firm slides only — exclude REFER / UNCERTAIN / TIE) ──
    sm_ml, sm_pr = {}, {}
    if has_gt and len(slide_df) > 0:
        s_gt = slide_df["gt_label"].notna()
        # Only include slides where both ML and PR gave firm binary diagnoses
        valid_s = [
            (g, m, p)
            for g, m, p in zip(
                slide_df.loc[s_gt, "gt_label"],
                slide_df.loc[s_gt, "ml_diagnosis"],
                slide_df.loc[s_gt, "pr_diagnosis"],
            )
            if _label_to_int(g) is not None
            and _label_to_int(m) is not None
            and _label_to_int(p) is not None
            and p not in ("REFER", "TIE", "UNCERTAIN")
            and m not in ("UNCERTAIN",)
        ]
        if valid_s:
            g_s, m_s, p_s = zip(*valid_s)
            sm_ml = _cls_metrics([_label_to_int(v) for v in g_s],
                                  [_label_to_int(v) for v in m_s])
            sm_pr = _cls_metrics([_label_to_int(v) for v in g_s],
                                  [_label_to_int(v) for v in p_s])

    # ── Sub-DataFrames ───────────────────────────────────────────
    changed_df = patch_df[patch_df["changed"]].copy()
    review_df  = patch_df[patch_df["needs_review"]].copy()
    # Errors remaining = firm wrong prediction AND not flagged
    errors_df  = patch_df[
        (patch_df["pr_correct"] == False) &
        (patch_df["pr_zone"] == "firm") &       # only firm wrong, not deferred
        patch_df["gt_label"].notna()
    ].copy() if has_gt else pd.DataFrame()

    # ── Coverage stats ───────────────────────────────────────────
    n_total      = len(patch_df)
    n_individual = int((patch_df["image_type"] == "individual").sum())
    n_sparse     = int((patch_df["image_type"] == "sparse_grid").sum())
    n_grid       = int((patch_df["image_type"] == "grid").sum())

    eligible_df  = patch_df[patch_df["image_type"] == "grid"]
    n_eligible   = len(eligible_df)
    n_pr_changed = int(eligible_df["changed"].sum())
    n_pr_flagged = int(eligible_df["needs_review"].sum())

    # Rule 4 flag analysis
    r4_df        = patch_df[
        patch_df["rule_applied"].str.contains("rule4", na=False) & patch_df["needs_review"]
    ].copy()
    n_r4_total      = len(r4_df)
    n_r4_errors     = int((r4_df["ml_correct"] == False).sum()) if has_gt else None
    n_r4_ok         = int((r4_df["ml_correct"] == True).sum())  if has_gt else None
    n_r4_false_alarm = n_r4_ok if has_gt else None

    # Silent failures: firm wrong prediction AND not flagged
    if has_gt:
        unflagged_errors_df = patch_df[
            (patch_df["pr_correct"] == False) &
            (patch_df["pr_zone"] == "firm") &
            (patch_df["needs_review"] == False) &
            patch_df["gt_label"].notna()
        ].copy()
        n_unflagged = len(unflagged_errors_df)
    else:
        unflagged_errors_df = pd.DataFrame()
        n_unflagged = None

    # ── Improvement breakdown by image type ──────────────────────
    def _type_metrics(df_sub):
        if not has_gt or len(df_sub) == 0:
            return {}
        # Accuracy only on firm predictions
        firm_sub = df_sub[df_sub["pr_zone"] == "firm"]
        ml_ok_n  = int((firm_sub["ml_correct"] == True).sum())
        pr_ok_n  = int((firm_sub["pr_correct"] == True).sum())
        n_firm   = len(firm_sub)
        n_def    = int((df_sub["pr_zone"] == "deferred").sum())
        return dict(
            n=len(df_sub),
            n_firm=n_firm,
            n_deferred=n_def,
            ml_acc  = f"{ml_ok_n/n_firm*100:.1f}%" if n_firm else "N/A",
            pr_acc  = f"{pr_ok_n/n_firm*100:.1f}%" if n_firm else "N/A",
            delta   = f"{(pr_ok_n - ml_ok_n)/n_firm*100:+.1f}%" if n_firm else "N/A",
            changed = int(df_sub["changed"].sum()),
            improved= int((df_sub["outcome"]=="improved").sum()),
            worsened= int((df_sub["outcome"]=="worsened").sum()),
        )

    type_stats = {
        t: _type_metrics(patch_df[patch_df["image_type"] == t])
        for t in ["individual", "sparse_grid", "grid"]
    }

    # ── Slide combo analysis ─────────────────────────────────────
    slide_combo_rows = []
    if has_gt:
        BINARY_GT      = BINARY_LABELS
        UNCERTAIN_GT_S = UNCERTAIN_LABELS

        for sid, names in slides.items():
            gt_classes = set()
            for n in names:
                gt = image_info.get(n, {}).get("gt_label")
                if gt:
                    gt_classes.add(gt)

            has_neg   = "G"     in gt_classes
            has_pos   = "Gplus" in gt_classes
            has_mixed = bool(gt_classes & UNCERTAIN_GT_S)
            if has_neg and has_pos:
                combo = "pos+neg"
            elif has_neg:
                combo = "neg"
            elif has_pos:
                combo = "pos"
            elif has_mixed:
                combo = "mixed_only"
            else:
                combo = "unknown"

            for n in names:
                info  = image_info.get(n, {})
                gt    = info.get("gt_label")
                if gt is None:
                    continue
                ml_p  = raw.get(n, {}).get("predicted", "")
                pr_p  = final.get(n, {}).get("final_predicted", ml_p)
                ml_ok = (ml_p == gt) if ml_p in BINARY_LABELS else None
                pr_ok = (pr_p == gt) if pr_p in BINARY_LABELS else None
                slide_combo_rows.append({
                    "slide_id":     sid,
                    "label_combo":  combo,
                    "gt_label":     gt,
                    "ml_predicted": ml_p,
                    "ml_zone":      "firm" if ml_p in BINARY_LABELS else "deferred",
                    "pr_predicted": pr_p,
                    "pr_zone":      "firm" if pr_p in BINARY_LABELS else "deferred",
                    "ml_correct":   ml_ok,
                    "pr_correct":   pr_ok,
                    "changed":      final.get(n, {}).get("changed", False),
                    "needs_review": final.get(n, {}).get("needs_review", False),
                })

    # No-GT path: classify slides by subfolder name so combo sheet shows something useful
    if not slide_combo_rows:
        for sid, names in slides.items():
            subfolders = set()
            for n in names:
                sf_val = image_info.get(n, {}).get("subfolder", "")
                if sf_val:
                    subfolders.add(str(sf_val).strip().lower())
            has_neg = any("neg" in s or "gminus" in s or "_g_" in s for s in subfolders)
            has_pos = any("pos" in s or "gplus" in s or "_gp_" in s for s in subfolders)
            combo = ("pos+neg" if (has_neg and has_pos)
                     else "neg" if has_neg
                     else "pos" if has_pos
                     else "unknown")
            for n in names:
                info_n = image_info.get(n, {})
                slide_combo_rows.append({
                    "slide_id":    sid,
                    "label_combo": combo,
                    "gt_label":    None,
                    "ml_predicted": raw.get(n, {}).get("predicted", ""),
                    "ml_zone":     "firm" if raw.get(n, {}).get("predicted", "") in BINARY_LABELS else "deferred",
                    "pr_predicted": final.get(n, {}).get("final_predicted", ""),
                    "pr_zone":     "firm" if final.get(n, {}).get("final_predicted", "") in BINARY_LABELS else "deferred",
                    "ml_correct":  None,
                    "pr_correct":  None,
                    "changed":     final.get(n, {}).get("changed", False),
                    "needs_review": final.get(n, {}).get("needs_review", False),
                })
    combo_df = pd.DataFrame(slide_combo_rows) if slide_combo_rows else pd.DataFrame()
    if len(combo_df) > 0:
        combo_df.to_csv(f"{output_dir}/slide_combo_patch_results.csv", index=False)

    # ── Save CSVs ────────────────────────────────────────────────
    patch_df.to_csv(f"{output_dir}/patch_results.csv",   index=False)
    slide_df.to_csv(f"{output_dir}/slide_results.csv",   index=False)
    if len(changed_df) > 0:
        changed_df.to_csv(f"{output_dir}/changed_patches.csv", index=False)

    # ── Console report ───────────────────────────────────────────
    _print_report(
        pm_ml, pm_pr, sm_ml, sm_pr,
        patch_df, changed_df, slide_df,
        n_individual, n_sparse, n_grid, n_eligible,
        n_r4_total, n_r4_errors, n_r4_false_alarm, n_unflagged,
        has_gt, output_dir,
        mixed_stats=mixed_stats,
        combo_df=combo_df,
        deferral_stats=deferral_stats if has_gt else {},
        fair_comparisons=fair_comparisons,
    )

    # ── Excel workbook ───────────────────────────────────────────
    try:
        _save_excel(
            output_dir=output_dir,
            patch_df=patch_df,
            slide_df=slide_df,
            changed_df=changed_df,
            review_df=review_df,
            errors_df=errors_df,
            unflagged_errors_df=unflagged_errors_df,
            pm_ml=pm_ml, pm_pr=pm_pr,
            sm_ml=sm_ml, sm_pr=sm_pr,
            has_gt=has_gt,
            n_total=n_total, n_individual=n_individual,
            n_sparse=n_sparse, n_grid=n_grid, n_eligible=n_eligible,
            n_pr_changed=n_pr_changed, n_pr_flagged=n_pr_flagged,
            n_r4_total=n_r4_total, n_r4_errors=n_r4_errors,
            n_r4_false_alarm=n_r4_false_alarm, n_unflagged=n_unflagged,
            type_stats=type_stats,
            mixed_stats=mixed_stats,
            combo_df=combo_df,
            deferral_stats=deferral_stats if has_gt else {},
            fair_comparisons=fair_comparisons,
            n_tier1=n_tier1, n_tier2=n_tier2, n_tier3=n_tier3,
        )
    except Exception as e:
        import traceback
        print(f"  [Excel] Could not save workbook: {e}")
        traceback.print_exc()

    return dict(
        patch_metrics_ml=pm_ml, patch_metrics_pr=pm_pr,
        slide_metrics_ml=sm_ml, slide_metrics_pr=sm_pr,
        deferral_stats=deferral_stats if has_gt else {},
        fair_comparisons=fair_comparisons,
        mixed_stats=mixed_stats,
        has_gt=has_gt,
        n_patches=n_total, n_changed=int(patch_df["changed"].sum()),
        n_slides=len(slide_df),
    )


# ─────────────────────────────────────────────────────────────────
#  Excel
# ─────────────────────────────────────────────────────────────────

def _save_excel(
    output_dir, patch_df, slide_df, changed_df, review_df,
    errors_df, unflagged_errors_df,
    pm_ml, pm_pr, sm_ml, sm_pr, has_gt,
    n_total, n_individual, n_sparse, n_grid, n_eligible,
    n_pr_changed, n_pr_flagged,
    n_r4_total, n_r4_errors, n_r4_false_alarm, n_unflagged,
    type_stats, mixed_stats=None, combo_df=None, deferral_stats=None,
    fair_comparisons=None, n_tier1=0, n_tier2=0, n_tier3=0,
):
    import openpyxl
    from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
    from openpyxl.utils import get_column_letter

    deferral_stats   = deferral_stats   or {}
    mixed_stats      = mixed_stats      or {}
    fair_comparisons = fair_comparisons or {}

    wb = openpyxl.Workbook()

    # ── Style constants ───────────────────────────────────────────
    C_NAVY  = "1E3A5F"; C_BLUE  = "2E6DA4"
    C_GB    = "C6EFCE"; C_GT    = "276221"
    C_RB    = "FFCCCC"; C_RT    = "9C0006"
    C_AB    = "FFF2CC"; C_AT    = "7D6608"
    C_GREY  = "F2F2F2"; C_WHITE = "FFFFFF"
    C_PURP  = "E8E0F4"; C_PURPT = "4B0082"
    C_TEAL  = "D0F0F0"; C_TEALT = "005555"
    C_BLACK = "000000"

    thin   = Side(style="thin", color="CCCCCC")
    border = Border(left=thin, right=thin, top=thin, bottom=thin)

    def _fill(h):
        return PatternFill("solid", fgColor=str(h).replace("#", ""))

    def _hdr(ws, r, c, v, bg=C_NAVY, ft=C_WHITE, sz=10, bold=True):
        cell = ws.cell(row=r, column=c, value=v)
        cell.font      = Font(bold=bold, color=str(ft).replace("#", ""), size=sz)
        cell.fill      = _fill(bg)
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        cell.border    = border
        return cell

    def _cell(ws, r, c, v, bg=None, ft=C_BLACK, bold=False, fmt=None, align="center"):
        cell = ws.cell(row=r, column=c, value=v)
        cell.font      = Font(bold=bold, color=str(ft).replace("#", ""), size=9)
        cell.alignment = Alignment(horizontal=align, vertical="center")
        cell.border    = border
        if bg:  cell.fill          = _fill(bg)
        if fmt: cell.number_format = fmt
        return cell

    def _auto_w(ws, mn=8, mx=32):
        for col in ws.columns:
            w = max((len(str(c.value or "")) for c in col), default=8)
            ws.column_dimensions[get_column_letter(col[0].column)].width = min(max(w+2, mn), mx)

    def _write_df(ws, df, sr=1, outcome_col=None, changed_col=None, correct_col=None):
        if df is None or len(df) == 0:
            ws.cell(row=sr, column=1, value="No data.")
            return
        for ci, col in enumerate(df.columns, 1):
            _hdr(ws, sr, ci, col)
        for ri, (_, row) in enumerate(df.iterrows(), sr+1):
            bg = C_GREY if ri % 2 == 0 else None
            oc = str(row.get(outcome_col, "")) if outcome_col else ""
            if   oc == "improved":  bg = C_GB
            elif oc == "worsened":  bg = C_RB
            elif oc == "correct":   bg = C_GB
            elif oc == "wrong":     bg = C_RB
            elif oc == "referred":  bg = C_TEAL

            for ci, col in enumerate(df.columns, 1):
                val = row[col]
                cbg = bg; cft = C_BLACK; cbl = False

                if changed_col and col == changed_col and val is True:
                    cbg, cft, cbl = C_AB, C_AT, True
                if correct_col and col == correct_col:
                    if val is True:   cbg, cft = C_GB, C_GT
                    elif val is False: cbg, cft = C_RB, C_RT
                if col == "outcome":
                    if val == "improved":  cbg, cft, cbl = C_GB, C_GT, True
                    elif val == "worsened": cbg, cft, cbl = C_RB, C_RT, True
                    elif val == "referred": cbg, cft, cbl = C_TEAL, C_TEALT, True
                if col in ("ml_zone", "pr_zone"):
                    if val == "deferred": cbg, cft = C_TEAL, C_TEALT
                if col == "image_type":
                    if val == "individual":  cbg = C_AB
                    elif val == "sparse_grid": cbg = C_PURP

                fmt = "0.0000" if col in ("ml_conf", "ml_prob_G", "ml_prob_Gplus",
                                           "majority_ratio", "slide_ratio") else None
                _cell(ws, ri, ci, val, bg=cbg, ft=cft, bold=cbl, fmt=fmt)

    # ─────────────────────────────────────────────────────────────
    #  Sheet 1: Summary
    # ─────────────────────────────────────────────────────────────
    ws1 = wb.active
    ws1.title = "Summary"
    ws1.sheet_view.showGridLines = False
    t = ws1.cell(row=1, column=1, value="PyReason Evaluation — Results Summary")
    t.font = Font(bold=True, size=14, color=C_WHITE)
    t.fill = _fill(C_NAVY)
    t.alignment = Alignment(horizontal="center", vertical="center")
    ws1.merge_cells("A1:H1")
    ws1.row_dimensions[1].height = 28

    r = 3

    # ── Explanation banner ────────────────────────────────────────
    note_text = (
        "EVALUATION DESIGN: CNN trained on 3 classes (G / Gplus / Mixed). "
        "Test set has no Mixed GT folder. "
        "When the model predicts Mixed on a G/Gplus patch it is expressing uncertainty — "
        "this is reported as DEFERRED (safe referral), NOT as a wrong prediction. "
        "Table A reports metrics on FIRM predictions only. "
        "Table B reports deferral/coverage (the safety metric)."
    )
    note_cell = ws1.cell(row=r, column=1, value=note_text)
    note_cell.font      = Font(italic=True, size=9, color="595959")
    note_cell.alignment = Alignment(wrap_text=True, horizontal="left")
    ws1.merge_cells(f"A{r}:H{r}")
    ws1.row_dimensions[r].height = 42
    r += 2

    # ── TABLE A: removed — not apple-to-apple (ML N != PR N) ────────
    n_ml_firm = deferral_stats.get("n_ml_firm", pm_ml.get("n", 0))
    n_pr_firm = deferral_stats.get("n_pr_firm", pm_pr.get("n", 0))

    ws1.cell(row=r, column=1,
             value="TABLE A — DIAGNOSTIC PERFORMANCE  (see note)").font = Font(
        bold=True, size=11, color=C_BLUE)
    r += 1
    note2 = ws1.cell(row=r, column=1,
        value=(
            f"NOT SHOWN: ML N={n_ml_firm}, PR N={n_pr_firm} — different denominators. "
            "PR resolves some deferred patches so it is tested on a larger, harder set. "
            "Comparing metrics on unequal sets is misleading. "
            "See TABLE C below: Scenario A (same N, apples-to-apples), "
            "Scenario B (full population), Scenario C (coverage quality)."
        ))
    note2.font      = Font(italic=True, size=9, color="9C0006")
    note2.alignment = Alignment(wrap_text=True, horizontal="left")
    ws1.merge_cells(f"A{r}:H{r}")
    ws1.row_dimensions[r].height = 36
    r += 2

    # ── TABLE B: Deferral / Safety ────────────────────────────────
    ws1.cell(row=r, column=1,
             value="TABLE B — DEFERRAL & COVERAGE  (safety metric)").font = Font(
        bold=True, size=11, color=C_BLUE)
    r += 1
    note3 = ws1.cell(row=r, column=1,
        value="A Mixed prediction on a G/Gplus GT patch = system defers to human review. "
              "Higher deferral with good accuracy on firm calls = clinically desirable. "
              "'Coverage' = fraction of patches where a firm call was made.")
    note3.font      = Font(italic=True, size=9, color="595959")
    note3.alignment = Alignment(wrap_text=True, horizontal="left")
    ws1.merge_cells(f"A{r}:H{r}")
    r += 1
    for ci, h in enumerate(["Metric", "ML only", "PyReason", "Change", "Interpretation"], 1):
        _hdr(ws1, r, ci, h, bg=C_BLUE)
    r += 1

    ds = deferral_stats
    n_tot_b = ds.get("n_total", 0)

    def _pct(num, denom):
        return f"{num/denom*100:.1f}%" if denom else "N/A"

    deferral_rows_b = [
        ("Total binary GT patches",
         n_tot_b, n_tot_b, "—",
         "All G/Gplus GT patches evaluated"),
        ("Firm predictions (N)",
         ds.get("n_ml_firm", 0), ds.get("n_pr_firm", 0),
         f"{ds.get('n_pr_firm',0) - ds.get('n_ml_firm',0):+d}",
         "Patches with a definitive G or Gplus output"),
        ("Coverage %",
         _pct(ds.get("n_ml_firm",0), n_tot_b),
         _pct(ds.get("n_pr_firm",0), n_tot_b),
         "—",
         "Higher = fewer patches deferred"),
        ("Deferred (Mixed prediction)",
         ds.get("n_ml_deferred", 0), ds.get("n_pr_deferred", 0),
         f"{ds.get('n_pr_deferred',0) - ds.get('n_ml_deferred',0):+d}",
         "Patches routed to human review (safe outcome)"),
        ("Deferral rate %",
         _pct(ds.get("n_ml_deferred",0), n_tot_b),
         _pct(ds.get("n_pr_deferred",0), n_tot_b),
         "—",
         "10-15% deferral is clinically acceptable if coverage accuracy is high"),
    ]
    for lbl, ml_v, pr_v, chg, interp in deferral_rows_b:
        _cell(ws1, r, 1, lbl, bold=True, align="left")
        _cell(ws1, r, 2, ml_v, bg=C_AB)
        _cell(ws1, r, 3, pr_v, bg=C_TEAL)
        _cell(ws1, r, 4, chg)
        _cell(ws1, r, 5, interp, align="left")
        r += 1
    r += 1

    # ── TABLE C: Fair comparison (denominator analysis) ───────────
    if fair_comparisons:
        ws1.cell(row=r, column=1,
                 value="TABLE C — FAIR COMPARISON  (denominator analysis)").font = Font(
            bold=True, size=11, color=C_BLUE)
        r += 1
        note_fc = ws1.cell(row=r, column=1,
            value=(
                "Table A compares ML (N=ML-firm) vs PR (N=PR-firm). "
                "When PR resolves deferred patches, denominators differ — PR is tested on a larger, harder set. "
                "This table provides three unambiguous comparisons. "
                "Scenario A is the fairest accuracy comparison. "
                "Scenario B shows the value of coverage. "
                "Scenario C shows quality of extra commits."
            ))
        note_fc.font      = Font(italic=True, size=9, color="595959")
        note_fc.alignment = Alignment(wrap_text=True, horizontal="left")
        ws1.merge_cells(f"A{r}:H{r}")
        ws1.row_dimensions[r].height = 36
        r += 1

        # Scenario A
        sa = fair_comparisons.get("scenario_A", {})
        if sa:
            ws1.cell(row=r, column=1,
                     value=f"Scenario A — Same N={sa.get('n',0)} (ML-firm patches only, apples-to-apples)").font = Font(
                bold=True, size=10, color=C_BLUE)
            r += 1
            _cell(ws1, r, 1, "Note: 'ML correct on same set' should equal or exceed 'ML only (Table A)'.",
                  align="left", bg=C_AB)
            r += 1
            for ci, h in enumerate(["Metric", "ML only (N=ML-firm)", "PR (same N)", "Delta", "Interpretation"], 1):
                _hdr(ws1, r, ci, h, bg=C_BLUE)
            r += 1
            ml_a = sa.get("ml", {}); pr_a = sa.get("pr", {})
            for key, label in [("mcc","MCC"), ("bal_acc","Balanced Accuracy"),
                                ("recall","Recall"), ("precision","Precision"), ("f1","F1")]:
                ml_v  = float(ml_a.get(key, 0))
                pr_v  = float(pr_a.get(key, 0))
                delta = pr_v - ml_v
                bg    = C_GB if delta > 0.001 else (C_RB if delta < -0.001 else None)
                ft    = C_GT if delta > 0.001 else (C_RT if delta < -0.001 else "000000")
                interp = (
                    "PR improved on ML's committed patches" if delta > 0.001 else
                    "PR slightly harmed ML's committed patches" if delta < -0.001 else
                    "No change on shared patches"
                )
                _cell(ws1, r, 1, label, bold=True, align="left")
                _cell(ws1, r, 2, round(ml_v, 4), fmt="0.0000", bg=C_AB)
                _cell(ws1, r, 3, round(pr_v, 4), fmt="0.0000", bg=bg, ft=ft, bold=(delta > 0.001))
                _cell(ws1, r, 4, round(delta, 4), fmt="+0.0000;-0.0000", bg=bg, ft=ft)
                _cell(ws1, r, 5, interp, align="left")
                r += 1
            _cell(ws1, r, 1,
                  f"N(PR deferred on ML-firm set) = {sa.get('n_pr_deferred_on_ml_set',0)}  "
                  f"(patches ML committed to but PR later deferred — very rare)",
                  align="left", bg=C_TEAL, ft=C_TEALT)
            r += 2

        # Scenario B
        sb = fair_comparisons.get("scenario_B", {})
        if sb:
            ws1.cell(row=r, column=1,
                     value=f"Scenario B — All {sb.get('n',0)} patches, Mixed = forced wrong (full-population view)").font = Font(
                bold=True, size=10, color=C_BLUE)
            r += 1
            _cell(ws1, r, 1,
                  "Treats ML-deferred patches as wrong answers (penalises refusing to answer). "
                  "Shows the full benefit of PyReason's coverage gain.",
                  align="left", bg=C_AB)
            r += 1
            for ci, h in enumerate(["Metric", "ML (forced, N=all)", "PR (forced, N=all)", "Delta", "Interpretation"], 1):
                _hdr(ws1, r, ci, h, bg=C_BLUE)
            r += 1
            ml_b = sb.get("ml", {}); pr_b = sb.get("pr", {})
            for key, label in [("mcc","MCC"), ("bal_acc","Balanced Accuracy"),
                                ("recall","Recall"), ("f1","F1")]:
                ml_v  = float(ml_b.get(key, 0))
                pr_v  = float(pr_b.get(key, 0))
                delta = pr_v - ml_v
                bg    = C_GB if delta > 0.001 else (C_RB if delta < -0.001 else None)
                ft    = C_GT if delta > 0.001 else (C_RT if delta < -0.001 else "000000")
                interp = "Significant improvement from coverage gain" if delta > 0.02 else \
                         "Modest improvement" if delta > 0.001 else \
                         "No change / slight decline"
                _cell(ws1, r, 1, label, bold=True, align="left")
                _cell(ws1, r, 2, round(ml_v, 4), fmt="0.0000", bg=C_AB)
                _cell(ws1, r, 3, round(pr_v, 4), fmt="0.0000", bg=bg, ft=ft, bold=(delta > 0.001))
                _cell(ws1, r, 4, round(delta, 4), fmt="+0.0000;-0.0000", bg=bg, ft=ft)
                _cell(ws1, r, 5, interp, align="left")
                r += 1
            r += 1

        # Scenario C
        sc = fair_comparisons.get("scenario_C", {})
        if sc and sc.get("n", 0) > 0:
            ws1.cell(row=r, column=1,
                     value=f"Scenario C — Extra {sc.get('n',0)} patches PR committed (ML deferred, PR firm)").font = Font(
                bold=True, size=10, color=C_BLUE)
            r += 1
            _cell(ws1, r, 1,
                  "Quality of PR's coverage gain: accuracy on the patches ML refused to answer. "
                  "These are inherently harder (CNN was uncertain). 75%+ accuracy is a good result.",
                  align="left", bg=C_AB)
            r += 1
            for ci, h in enumerate(["Metric", "Value", "Interpretation"], 1):
                _hdr(ws1, r, ci, h, bg=C_BLUE)
            r += 1
            n_sc = sc.get("n", 0)
            acc_sc = sc.get("accuracy", 0)
            bg_acc = C_GB if acc_sc >= 0.75 else (C_AB if acc_sc >= 0.60 else C_RB)
            sc_rows = [
                ("Extra patches committed",       n_sc,                    "ML deferred, PR gave firm answer"),
                ("Correct (PR right)",            sc.get("n_correct", 0),  f"{sc.get('n_correct',0)/n_sc*100:.1f}% of extra" if n_sc else "—"),
                ("Wrong (PR introduced error)",   sc.get("n_wrong", 0),    f"{sc.get('n_wrong',0)/n_sc*100:.1f}% of extra" if n_sc else "—"),
                ("Accuracy on extra patches",     f"{acc_sc*100:.1f}%",    "≥75% = good; <60% = net harmful coverage"),
                ("MCC on extra patches",
                 round(sc.get("metrics",{}).get("mcc",0), 4),
                 "Binary metric on the extra-commit set"),
            ]
            for lbl, val, interp in sc_rows:
                _cell(ws1, r, 1, lbl, bold=True, align="left")
                _cell(ws1, r, 2, val, bg=bg_acc if "Accuracy" in lbl else C_AB)
                _cell(ws1, r, 3, interp, align="left")
                r += 1
            r += 1

    # ── Confusion matrix (firm predictions only) ──────────────────
    ws1.cell(row=r, column=1,
             value="CONFUSION MATRIX  (firm predictions only — Mixed-predicted patches excluded)").font = Font(
        bold=True, size=11, color=C_BLUE)
    r += 1
    note4 = ws1.cell(row=r, column=1,
        value="Positive class = Gplus. "
              "Only patches where the system gave a firm G or Gplus answer are counted here. "
              "Deferred patches are reported separately in Table B and the Coverage sheet.")
    note4.font = Font(italic=True, size=9, color="595959")
    note4.alignment = Alignment(wrap_text=True, horizontal="left")
    ws1.merge_cells(f"A{r}:H{r}")
    r += 1
    for ci, h in enumerate(["", "TP", "TN", "FP", "FN", "N (firm)"], 1):
        _hdr(ws1, r, ci, h, bg=C_BLUE)
    r += 1
    for label, pm, is_pr in [("ML only", pm_ml, False), ("PyReason", pm_pr, True)]:
        tp_v = int(pm.get("tp", 0))
        tn_v = int(pm.get("tn", 0))
        fp_v = int(pm.get("fp", 0))
        fn_v = int(pm.get("fn", 0))
        n_firm_v = tp_v + tn_v + fp_v + fn_v
        _cell(ws1, r, 1, label, bold=is_pr, align="left")
        _cell(ws1, r, 2, tp_v, bg=C_GB, bold=is_pr)
        _cell(ws1, r, 3, tn_v, bg=C_GB, bold=is_pr)
        _cell(ws1, r, 4, fp_v, bg=C_RB)
        _cell(ws1, r, 5, fn_v, bg=C_RB)
        _cell(ws1, r, 6, n_firm_v, bold=is_pr)
        r += 1
    r += 1

    # ── Mixed GT patches (Zone 3 — training eval, often empty at test) ──
    if mixed_stats and mixed_stats.get("n_mixed", 0) > 0:
        ws1.cell(row=r, column=1,
                 value="MIXED GT PATCHES — screening quality  "
                       "(training set eval; test set typically has no Mixed GT)").font = Font(
            bold=True, size=11, color=C_BLUE)
        r += 1
        note5 = ws1.cell(row=r, column=1,
            value="Goal for Mixed GT patches: route them to human review (flag), "
                  "not force them into G/Gplus. Higher flag rate = better.")
        note5.font      = Font(italic=True, size=9, color="595959")
        note5.alignment = Alignment(wrap_text=True, horizontal="left")
        ws1.merge_cells(f"A{r}:H{r}")
        r += 1
        for ci, h in enumerate(["Metric", "ML only", "PyReason", "Rate ML", "Rate PR"], 1):
            _hdr(ws1, r, ci, h, bg=C_BLUE)
        r += 1
        mx = mixed_stats
        flag_better = mx.get("flag_rate_pr", 0) >= mx.get("flag_rate_ml", 0)
        mx_rows = [
            ("Total Mixed GT patches",        mx["n_mixed"],                   mx["n_mixed"],              "—",   "—"),
            ("CNN predicted Mixed (correct)", mx.get("n_ml_predicted_mixed",0), "—",
             f"{mx.get('ml_mixed_pred_rate',0)*100:.1f}%", "—"),
            ("Flagged for review (Rule 4)",   mx["n_flagged_ml"],               mx["n_flagged_pr"],
             f"{mx['flag_rate_ml']*100:.1f}%", f"{mx['flag_rate_pr']*100:.1f}%"),
            ("Changed by PyReason",           "—",                              mx["n_changed"],             "—", "—"),
        ]
        for lbl, ml_v, pr_v, ml_r, pr_r in mx_rows:
            is_flag = "Flagged" in lbl
            bg_pr = (C_GB if flag_better else C_RB) if is_flag else C_AB
            _cell(ws1, r, 1, lbl, bold=True, align="left")
            _cell(ws1, r, 2, ml_v, bg=C_AB if ml_v != "—" else None)
            _cell(ws1, r, 3, pr_v, bg=bg_pr if pr_v != "—" else None)
            _cell(ws1, r, 4, ml_r)
            _cell(ws1, r, 5, pr_r)
            r += 1
        r += 1

    # ── Slide-level metrics ───────────────────────────────────────
    if sm_ml:
        ws1.cell(row=r, column=1,
                 value="SLIDE-LEVEL METRICS  (firm diagnoses only — REFER slides excluded)").font = Font(
            bold=True, size=11, color=C_BLUE)
        r += 1
        for ci, h in enumerate(["Metric", "ML only", "PyReason", "Delta", "% Change"], 1):
            _hdr(ws1, r, ci, h, bg=C_BLUE)
        r += 1
        for key, label in [("mcc","MCC"), ("bal_acc","Balanced Accuracy"),
                            ("f1","F1"), ("f2","F2"),
                            ("acc","Accuracy"), ("precision","Precision"), ("recall","Recall")]:
            ml_v  = float(sm_ml.get(key, 0)); pr_v = float(sm_pr.get(key, 0))
            delta = pr_v - ml_v; pct = (delta / ml_v * 100) if ml_v else 0.0
            bg = C_GB if delta > 0.001 else (C_RB if delta < -0.001 else None)
            ft = C_GT if delta > 0.001 else (C_RT if delta < -0.001 else "000000")
            _cell(ws1, r, 1, label, bold=True, align="left")
            _cell(ws1, r, 2, round(ml_v, 4), fmt="0.0000", bg=C_AB)
            _cell(ws1, r, 3, round(pr_v, 4), fmt="0.0000", bg=bg, ft=ft)
            _cell(ws1, r, 4, round(delta, 4), fmt="+0.0000;-0.0000", bg=bg, ft=ft)
            _cell(ws1, r, 5, f"{pct:+.2f}%", bg=bg, ft=ft)
            r += 1
        # Referred slides count
        n_referred = int((slide_df["pr_diagnosis"] == "REFER").sum()) if len(slide_df) else 0
        _cell(ws1, r, 1,
              f"Slides referred (REFER diagnosis): {n_referred}  "
              f"({n_referred/len(slide_df)*100:.1f}% of all slides)" if len(slide_df) else "No slides",
              align="left", bg=C_TEAL, ft=C_TEALT)
        r += 2

    # ── Rule breakdown ────────────────────────────────────────────
    ws1.cell(row=r, column=1, value="RULE BREAKDOWN").font = Font(bold=True, size=11, color=C_BLUE)
    r += 1
    n_ch = int(patch_df["changed"].sum())
    n_rv = int(patch_df["needs_review"].sum())

    # Rule 3 and Rule 4 flagged counts (no prediction change)
    n_r3_flagged = int(patch_df[
        patch_df["rule_applied"].str.contains("rule3", na=False) & patch_df["needs_review"]
    ].shape[0])
    n_r4_flagged = int(patch_df[
        patch_df["rule_applied"].str.contains("rule4", na=False) & patch_df["needs_review"]
    ].shape[0])

    for ci, h in enumerate(["Rule", "Total", "Improved", "Worsened", "No GT change", "Type"], 1):
        _hdr(ws1, r, ci, h, bg=C_BLUE)
    r += 1

    # Summary row
    n_imp_all = int((patch_df[patch_df["changed"]]["outcome"] == "improved").sum()) if n_ch else 0
    n_wor_all = int((patch_df[patch_df["changed"]]["outcome"] == "worsened").sum()) if n_ch else 0
    n_sam_all = n_ch - n_imp_all - n_wor_all
    _cell(ws1, r, 1, "All changes (total)", bold=True, align="left")
    _cell(ws1, r, 2, n_ch, bg=C_AB, bold=True)
    _cell(ws1, r, 3, n_imp_all, bg=C_GB, ft=C_GT, bold=True)
    _cell(ws1, r, 4, n_wor_all, bg=C_RB if n_wor_all else None, ft=C_RT if n_wor_all else C_BLACK, bold=bool(n_wor_all))
    _cell(ws1, r, 5, n_sam_all, bg=C_GREY if has_gt else C_AB)
    _cell(ws1, r, 6, "prediction changed (improved/worsened require GT)", align="left")
    r += 1

    # Per-rule rows (prediction-changing rules)
    if n_ch > 0 and has_gt:
        changed_df_s = patch_df[patch_df["changed"]].copy()
        for rule in changed_df_s["rule_applied"].dropna().unique():
            sub_r = changed_df_s[changed_df_s["rule_applied"] == rule]
            cnt   = len(sub_r)
            imp   = int((sub_r["outcome"] == "improved").sum())
            wor   = int((sub_r["outcome"] == "worsened").sum())
            sam   = cnt - imp - wor
            bg_w  = C_RB if wor else None
            ft_w  = C_RT if wor else C_BLACK
            _cell(ws1, r, 1, f"  {rule}", align="left")
            _cell(ws1, r, 2, cnt, bg=C_AB)
            _cell(ws1, r, 3, imp, bg=C_GB if imp else None, ft=C_GT if imp else C_BLACK)
            _cell(ws1, r, 4, wor, bg=bg_w, ft=ft_w, bold=bool(wor))
            _cell(ws1, r, 5, sam, bg=C_GREY if sam else None)
            _cell(ws1, r, 6, "changed", align="left")
            r += 1

    # Rule 3: cluster flag (no prediction change)
    _cell(ws1, r, 1, "rule3_cluster_flag (flag only)", align="left")
    _cell(ws1, r, 2, n_r3_flagged, bg=C_AB)
    _cell(ws1, r, 3, "—"); _cell(ws1, r, 4, "—"); _cell(ws1, r, 5, "—")
    _cell(ws1, r, 6, "flag for review, no prediction change", align="left")
    r += 1

    # Rule 4: low-conf flag (no prediction change)
    _cell(ws1, r, 1, "rule4_review (flag only)", align="left")
    _cell(ws1, r, 2, n_r4_flagged, bg=C_AB)
    _cell(ws1, r, 3, "—"); _cell(ws1, r, 4, "—"); _cell(ws1, r, 5, "—")
    _cell(ws1, r, 6, "LOW-conf or Mixed flagged for human review", align="left")
    r += 1

    _auto_w(ws1)

    # ─────────────────────────────────────────────────────────────
    #  Sheet 2: Coverage
    # ─────────────────────────────────────────────────────────────
    ws2 = wb.create_sheet("Coverage")
    ws2.sheet_view.showGridLines = False
    t2 = ws2.cell(row=1, column=1, value="Image Coverage & PyReason Eligibility")
    t2.font      = Font(bold=True, size=13, color=C_WHITE)
    t2.fill      = _fill(C_NAVY)
    t2.alignment = Alignment(horizontal="center", vertical="center")
    ws2.merge_cells("A1:F1")
    ws2.row_dimensions[1].height = 24

    r = 3
    ws2.cell(row=r, column=1,
             value="IMAGE TYPE BREAKDOWN").font = Font(bold=True, size=11, color=C_BLUE)
    r += 1
    for ci, h in enumerate(["Type", "Count", "% of Total", "Can PyReason Fire?", "Explanation"], 1):
        _hdr(ws2, r, ci, h, bg=C_BLUE)
    r += 1
    type_info = [
        ("individual",  n_individual, "NO",
         "No row/col in filename OR only 1 patch in slide. Cannot do majority vote."),
        ("sparse_grid", n_sparse, "NO",
         f"Has row/col but fewer than {config.MIN_PATCHES_FOR_VOTE} firm patches. Vote unreliable."),
        ("grid",        n_grid,   "YES",
         "Has row/col AND enough patches. Full Rule 1+2+3 eligible."),
    ]
    type_bg = {"individual": C_AB, "sparse_grid": C_PURP, "grid": C_GB}
    type_ft = {"individual": C_AT, "sparse_grid": C_PURPT, "grid": C_GT}
    for tname, cnt, can_fire, expl in type_info:
        pct = f"{cnt/n_total*100:.1f}%" if n_total else "0%"
        bg  = type_bg[tname]; ft = type_ft[tname]
        _cell(ws2, r, 1, tname, bg=bg, ft=ft, bold=True, align="left")
        _cell(ws2, r, 2, cnt,  bg=bg, ft=ft)
        _cell(ws2, r, 3, pct,  bg=bg, ft=ft)
        _cell(ws2, r, 4, can_fire,
              bg=(C_GB if can_fire == "YES" else C_RB),
              ft=(C_GT if can_fire == "YES" else C_RT), bold=True)
        _cell(ws2, r, 5, expl, align="left")
        r += 1
    _cell(ws2, r, 1, "TOTAL", bold=True, align="left")
    _cell(ws2, r, 2, n_total, bold=True)
    _cell(ws2, r, 3, "100%", bold=True)
    r += 2

    # Deferral breakdown on Coverage sheet
    if deferral_stats and deferral_stats.get("n_total", 0) > 0:
        ws2.cell(row=r, column=1,
                 value="DEFERRAL BREAKDOWN  (G/Gplus GT patches — prediction zone analysis)").font = Font(
            bold=True, size=11, color=C_BLUE)
        r += 1
        note_def = ws2.cell(row=r, column=1,
            value="'Firm' = model predicted G or Gplus. "
                  "'Deferred' = model predicted Mixed (appropriate uncertainty). "
                  "Deferred patches are NOT included in Table A metrics.")
        note_def.font      = Font(italic=True, size=9, color="595959")
        note_def.alignment = Alignment(wrap_text=True, horizontal="left")
        ws2.merge_cells(f"A{r}:F{r}")
        r += 1
        for ci, h in enumerate(["Metric", "ML only", "PyReason", "Explanation"], 1):
            _hdr(ws2, r, ci, h, bg=C_BLUE)
        r += 1
        ds = deferral_stats
        n_b = ds.get("n_total", 1)
        def_rows_cov = [
            ("Total G/Gplus GT patches",   ds.get("n_total",0),       ds.get("n_total",0),
             "Denominator for all rates"),
            ("Firm predictions",           ds.get("n_ml_firm",0),      ds.get("n_pr_firm",0),
             "G or Gplus output — used in Table A metrics"),
            ("Coverage %",                 f"{ds.get('ml_coverage',0)*100:.1f}%",
             f"{ds.get('pr_coverage',0)*100:.1f}%",
             "Firm / Total — % of patches with a definitive answer"),
            ("Deferred (Mixed output)",    ds.get("n_ml_deferred",0),  ds.get("n_pr_deferred",0),
             "Routed to human review — safe outcome, not a wrong answer"),
            ("Deferral rate %",            f"{ds.get('ml_deferral_rate',0)*100:.1f}%",
             f"{ds.get('pr_deferral_rate',0)*100:.1f}%",
             "Deferred / Total"),
        ]
        for lbl, ml_v, pr_v, expl in def_rows_cov:
            _cell(ws2, r, 1, lbl, bold=True, align="left")
            _cell(ws2, r, 2, ml_v, bg=C_AB)
            _cell(ws2, r, 3, pr_v, bg=C_TEAL, ft=C_TEALT)
            _cell(ws2, r, 4, expl, align="left")
            r += 1
        r += 1

    ws2.cell(row=r, column=1,
             value="PYREASON FIRING SUMMARY (grid patches only)").font = Font(bold=True, size=11, color=C_BLUE)
    r += 1
    for ci, h in enumerate(["Metric", "Count", "% of Grid"], 1):
        _hdr(ws2, r, ci, h, bg=C_BLUE)
    r += 1
    for label, cnt in [
        ("Grid patches (eligible)", n_grid),
        ("Changed by Rule 1 or 2",  n_pr_changed),
        ("Flagged by Rule 4",       n_pr_flagged),
    ]:
        pct = f"{cnt/n_grid*100:.1f}%" if n_grid else "0%"
        bg  = C_GB if "Changed" in label else (C_AB if "Flag" in label else None)
        _cell(ws2, r, 1, label, bold=True, align="left")
        _cell(ws2, r, 2, cnt, bg=bg)
        _cell(ws2, r, 3, pct, bg=bg)
        r += 1
    r += 1

    ws2.cell(row=r, column=1,
             value="RULE 4 — FLAG REVIEW QUALITY").font = Font(bold=True, size=11, color=C_BLUE)
    r += 1
    _cell(ws2, r, 1, "Rule 4 flags LOW confidence patches for human review.", align="left")
    r += 1
    for ci, h in enumerate(["Metric", "Count", "% of Flagged", "Explanation"], 1):
        _hdr(ws2, r, ci, h, bg=C_BLUE)
    r += 1
    pct_e = f"{n_r4_errors/n_r4_total*100:.1f}%"   if (n_r4_total and n_r4_errors is not None)     else "N/A"
    pct_f = f"{n_r4_false_alarm/n_r4_total*100:.1f}%" if (n_r4_total and n_r4_false_alarm is not None) else "N/A"
    r4_rows_cov = [
        ("Total flagged (Rule 4)",  n_r4_total,       "100%",  "All LOW-confidence patches"),
        ("True errors in flags",    n_r4_errors,       pct_e,
         "Flagged AND actually wrong — correct to send for review" if n_r4_errors is not None else "Need GT"),
        ("False alarms in flags",   n_r4_false_alarm,  pct_f,
         "Flagged BUT actually correct — unnecessary review burden" if n_r4_false_alarm is not None else "Need GT"),
    ]
    for label, cnt, pct, expl in r4_rows_cov:
        bg = C_GB if "True errors" in label else (C_RB if "False alarms" in label else C_AB)
        _cell(ws2, r, 1, label, bold=True, align="left")
        _cell(ws2, r, 2, cnt if cnt is not None else "N/A", bg=bg)
        _cell(ws2, r, 3, pct,  bg=bg)
        _cell(ws2, r, 4, expl, align="left")
        r += 1
    r += 1

    ws2.cell(row=r, column=1,
             value="SILENT FAILURES (firm wrong AND not flagged)").font = Font(bold=True, size=11, color=C_BLUE)
    r += 1
    _cell(ws2, r, 1,
          "Patches where the system made a FIRM wrong prediction AND did not flag for review. "
          "Deferred wrong predictions are NOT counted here — they were already routed to a human.",
          align="left")
    r += 1
    for ci, h in enumerate(["Metric", "Count", "% of Firm Total", "Explanation"], 1):
        _hdr(ws2, r, ci, h, bg=C_BLUE)
    r += 1
    n_pr_firm_total = deferral_stats.get("n_pr_firm", n_total)
    pct_uf = f"{n_unflagged/n_pr_firm_total*100:.1f}%" if (n_unflagged is not None and n_pr_firm_total) else "N/A"
    _cell(ws2, r, 1, "Silent failures (unflagged firm errors)", bold=True, align="left")
    _cell(ws2, r, 2, n_unflagged if n_unflagged is not None else "N/A", bg=C_RB, ft=C_RT, bold=True)
    _cell(ws2, r, 3, pct_uf, bg=C_RB, ft=C_RT)
    _cell(ws2, r, 4, "Firm wrong AND not flagged. These are the true misses.", align="left")
    r += 2

    if has_gt:
        ws2.cell(row=r, column=1,
                 value="IMPROVEMENT BY IMAGE TYPE  (firm predictions only)").font = Font(
            bold=True, size=11, color=C_BLUE)
        r += 1
        for ci, h in enumerate(["Type", "N total", "N firm", "N deferred",
                                 "ML Acc (firm)", "PR Acc (firm)", "Delta",
                                 "Changed", "Improved", "Worsened"], 1):
            _hdr(ws2, r, ci, h, bg=C_BLUE)
        r += 1
        for tname in ["individual", "sparse_grid", "grid"]:
            s  = type_stats.get(tname, {})
            bg = type_bg[tname]; ft = type_ft[tname]
            _cell(ws2, r, 1, tname,                    bg=bg, ft=ft, bold=True, align="left")
            _cell(ws2, r, 2, s.get("n", ""),           bg=bg, ft=ft)
            _cell(ws2, r, 3, s.get("n_firm", ""),      bg=bg, ft=ft)
            _cell(ws2, r, 4, s.get("n_deferred", ""),  bg=C_TEAL, ft=C_TEALT)
            _cell(ws2, r, 5, s.get("ml_acc", "N/A"),   bg=bg, ft=ft)
            _cell(ws2, r, 6, s.get("pr_acc", "N/A"),   bg=bg, ft=ft)
            delta_v   = s.get("delta", "N/A")
            delta_bg  = (C_GB if isinstance(delta_v, str) and delta_v.startswith("+") else
                         C_RB if isinstance(delta_v, str) and delta_v.startswith("-") else bg)
            _cell(ws2, r, 7,  delta_v,              bg=delta_bg, bold=True)
            _cell(ws2, r, 8,  s.get("changed", ""))
            _cell(ws2, r, 9,  s.get("improved", ""),  bg=C_GB, ft=C_GT)
            _cell(ws2, r, 10, s.get("worsened", ""),  bg=C_RB, ft=C_RT)
            r += 1

    _auto_w(ws2)
    ws2.column_dimensions["E"].width = 55

    # ─────────────────────────────────────────────────────────────
    #  Sheet: By Slide Combo
    # ─────────────────────────────────────────────────────────────
    if combo_df is not None and len(combo_df) > 0:
        ws_combo = wb.create_sheet("By Slide Combo")
        ws_combo.sheet_view.showGridLines = False
        t_combo = ws_combo.cell(row=1, column=1,
            value="Slide-level outcome breakdown by label combination")
        t_combo.font      = Font(bold=True, size=13, color=C_WHITE)
        t_combo.fill      = _fill(C_NAVY)
        t_combo.alignment = Alignment(horizontal="center", vertical="center")
        ws_combo.merge_cells("A1:J1")
        ws_combo.row_dimensions[1].height = 24

        note_combo = ws_combo.cell(row=2, column=1,
            value=(
                "Slide-level outcomes only. ML acc vs PR acc is NOT shown here because "
                "ML and PR evaluate on different patch counts (unequal denominators). "
                "Outcomes (improved / worsened / same) are fair: they measure "
                "whether PyReason changed a slide diagnosis for better or worse, "
                "regardless of how many patches each system answered."
            ))
        note_combo.font      = Font(italic=True, size=9, color="595959")
        note_combo.alignment = Alignment(wrap_text=True, horizontal="left")
        ws_combo.merge_cells("A2:J2")
        ws_combo.row_dimensions[2].height = 32

        COMBO_COLORS = {
            "neg":        ("E6F1FB", "185FA5"),
            "pos":        ("FBEAF0", "993556"),
            "pos+neg":    ("EEEDFE", "534AB7"),
            "mixed_only": ("FAEEDA", "854F0B"),
        }
        combos_ordered = ["neg", "pos", "pos+neg", "mixed_only"]

        # Build slide-level outcome table from slide_df + combo_df
        # slide_df has pr_diagnosis; combo_df has per-patch slide_id and label_combo
        slide_combo_map = {}
        if "label_combo" in combo_df.columns and "slide_id" in combo_df.columns:
            slide_combo_map = (
                combo_df[["slide_id","label_combo"]]
                .drop_duplicates()
                .set_index("slide_id")["label_combo"]
                .to_dict()
            )

        rr = 4
        for combo in combos_ordered:
            combo_slides = [sid for sid, lc in slide_combo_map.items() if lc == combo]
            if not combo_slides:
                continue

            sub_slides = slide_df[slide_df["slide_id"].isin(combo_slides)] if len(slide_df) else pd.DataFrame()
            bg_h, ft_h = COMBO_COLORS.get(combo, ("F2F2F2", "000000"))

            ws_combo.cell(row=rr, column=1,
                value=f"Combo: {combo}  ({len(combo_slides)} slides)").font = Font(
                bold=True, size=11, color=ft_h)
            rr += 1

            hdrs = ["Slide outcome", "N slides", "% of combo",
                    "Description",
                    "N patches changed", "N patches flagged"]
            for ci, h in enumerate(hdrs, 1):
                _hdr(ws_combo, rr, ci, h, bg=C_BLUE)
            rr += 1

            if len(sub_slides) == 0:
                ws_combo.cell(row=rr, column=1, value="No slide data for this combo.")
                rr += 2
                continue

            n_slides_combo = len(sub_slides)
            outcome_defs = [
                ("improved",  C_GB,   C_GT,   "PyReason changed slide diagnosis: was wrong, now correct"),
                ("worsened",  C_RB,   C_RT,   "PyReason changed slide diagnosis: was correct, now wrong"),
                ("correct",   C_GB,   C_GT,   "Both ML and PR correct (no change needed)"),
                ("wrong",     C_RB,   C_RT,   "Both ML and PR wrong (PyReason could not help)"),
                ("referred",  C_TEAL, C_TEALT,"PR referred to human (Mixed majority — safe outcome)"),
                ("no_gt",     C_GREY, "000000","No GT label available for this slide"),
            ]
            for oc_name, bg_oc, ft_oc, desc in outcome_defs:
                cnt = int((sub_slides["outcome"] == oc_name).sum()) if "outcome" in sub_slides.columns else 0
                if cnt == 0:
                    continue
                pct = f"{cnt/n_slides_combo*100:.1f}%"
                # patches changed and flagged for slides with this outcome
                oc_slide_ids = sub_slides[sub_slides["outcome"] == oc_name]["slide_id"].tolist()
                n_patches_chg = int(patch_df[patch_df["slide_id"].isin(oc_slide_ids)]["changed"].sum()) if len(patch_df) else 0
                n_patches_flg = int(patch_df[patch_df["slide_id"].isin(oc_slide_ids)]["needs_review"].sum()) if len(patch_df) else 0
                _cell(ws_combo, rr, 1, oc_name,     bold=True, align="left", bg=bg_oc, ft=ft_oc)
                _cell(ws_combo, rr, 2, cnt,          bg=bg_oc, ft=ft_oc, bold=True)
                _cell(ws_combo, rr, 3, pct,          bg=bg_oc, ft=ft_oc)
                _cell(ws_combo, rr, 4, desc,         align="left")
                _cell(ws_combo, rr, 5, n_patches_chg, bg=C_AB if n_patches_chg else None)
                _cell(ws_combo, rr, 6, n_patches_flg)
                rr += 1
            rr += 1

        _auto_w(ws_combo)
        ws_combo.column_dimensions["D"].width = 52
        ws_combo.freeze_panes = "B4"

    # ─────────────────────────────────────────────────────────────
    #  Data sheets
    # ─────────────────────────────────────────────────────────────
    # Drop ml_correct and pr_correct from slide_df display — no information value
    # (slide outcome column already encodes this)
    slide_df_display = slide_df.drop(
        columns=[c for c in ("ml_correct", "pr_correct") if c in slide_df.columns],
        errors="ignore"
    )
    # ─────────────────────────────────────────────────────────────
    #  Sheet: Triage Tier Summary
    # ─────────────────────────────────────────────────────────────
    ws_triage = wb.create_sheet("Triage Tiers")
    ws_triage.sheet_view.showGridLines = False
    t_tr = ws_triage.cell(row=1, column=1, value="PyReason Triage Tier Summary")
    t_tr.font = Font(bold=True, size=13, color=C_WHITE)
    t_tr.fill = _fill(C_NAVY)
    t_tr.alignment = Alignment(horizontal="center", vertical="center")
    ws_triage.merge_cells("A1:F1")
    ws_triage.row_dimensions[1].height = 24

    note_tr = ws_triage.cell(row=2, column=1,
        value=(
            "Tier 1 Confirmed: HIGH-conf, not changed, not flagged. "
            "Tier 2 Suggested: rule corrected (changed=True) or medium-conf flagged. "
            "Tier 3 Refer: flagged with no correction, low-conf or Mixed. "
            "All grid-eligible patches are covered. "
            "Improved/Worsened require GT labels."
        ))
    note_tr.font = Font(italic=True, size=9, color="595959")
    note_tr.alignment = Alignment(wrap_text=True, horizontal="left")
    ws_triage.merge_cells("A2:F2")
    ws_triage.row_dimensions[2].height = 36

    rtr = 4
    # Summary table
    _hdr(ws_triage, rtr, 1, "Tier",    bg=C_NAVY, sz=10)
    _hdr(ws_triage, rtr, 2, "Label",   bg=C_NAVY, sz=10)
    _hdr(ws_triage, rtr, 3, "N patches", bg=C_NAVY, sz=10)
    _hdr(ws_triage, rtr, 4, "% of eligible", bg=C_NAVY, sz=10)
    _hdr(ws_triage, rtr, 5, "N changed", bg=C_NAVY, sz=10)
    _hdr(ws_triage, rtr, 6, "N flagged", bg=C_NAVY, sz=10)
    rtr += 1

    n_elig_triage = max(n_tier1 + n_tier2 + n_tier3, 1)
    tier_defs = [
        (1, "Confirmed", n_tier1, C_GB, C_GT),
        (2, "Suggested", n_tier2, C_AB, C_AT),
        (3, "Refer",     n_tier3, C_RB, C_RT),
    ]
    for tier_num, tier_lbl, tier_n, bg_t, ft_t in tier_defs:
        pct = f"{tier_n / n_elig_triage * 100:.1f}%" if n_elig_triage else "0%"
        tier_patch = patch_df[patch_df["triage_tier"] == tier_num] if "triage_tier" in patch_df.columns else pd.DataFrame()
        n_chg_t = int(tier_patch["changed"].sum()) if len(tier_patch) else 0
        n_flg_t = int(tier_patch["needs_review"].sum()) if len(tier_patch) else 0
        _cell(ws_triage, rtr, 1, tier_num,  bold=True, bg=bg_t, ft=ft_t)
        _cell(ws_triage, rtr, 2, tier_lbl,  bold=True, bg=bg_t, ft=ft_t, align="left")
        _cell(ws_triage, rtr, 3, tier_n,    bold=True, bg=bg_t, ft=ft_t)
        _cell(ws_triage, rtr, 4, pct,       bg=bg_t, ft=ft_t)
        _cell(ws_triage, rtr, 5, n_chg_t,   bg=C_AB if n_chg_t else None)
        _cell(ws_triage, rtr, 6, n_flg_t,   bg=C_AB if n_flg_t else None)
        rtr += 1

    # Total row
    n_tot_triage = n_tier1 + n_tier2 + n_tier3
    _cell(ws_triage, rtr, 1, "TOTAL", bold=True, align="left")
    _cell(ws_triage, rtr, 2, "",     bold=True)
    _cell(ws_triage, rtr, 3, n_tot_triage, bold=True)
    _cell(ws_triage, rtr, 4, "100%", bold=True)
    rtr += 2

    # Rule firing per tier
    ws_triage.cell(row=rtr, column=1, value="RULE BREAKDOWN BY TIER").font = Font(bold=True, size=10, color=C_BLUE)
    rtr += 1
    _hdr(ws_triage, rtr, 1, "Rule",         bg=C_BLUE)
    _hdr(ws_triage, rtr, 2, "Total fired",  bg=C_BLUE)
    _hdr(ws_triage, rtr, 3, "In Tier 1",    bg=C_BLUE)
    _hdr(ws_triage, rtr, 4, "In Tier 2",    bg=C_BLUE)
    _hdr(ws_triage, rtr, 5, "In Tier 3",    bg=C_BLUE)
    _hdr(ws_triage, rtr, 6, "Action type",  bg=C_BLUE)
    rtr += 1

    rule_action = {
        "rule1a":             "Prediction changed",
        "rule1b":             "Prediction changed",
        "rule2_neighbor":     "Prediction changed",
        "rule3_cluster_flag": "Flag only",
        "rule4_review":       "Flag only",
    }
    if "rule_applied" in patch_df.columns and "triage_tier" in patch_df.columns:
        for rule_name, action in rule_action.items():
            sub_rule = patch_df[patch_df["rule_applied"].str.contains(
                rule_name.split("_")[0], na=False)]
            cnt_total = len(sub_rule)
            if cnt_total == 0:
                continue
            cnt_t1 = int((sub_rule["triage_tier"] == 1).sum())
            cnt_t2 = int((sub_rule["triage_tier"] == 2).sum())
            cnt_t3 = int((sub_rule["triage_tier"] == 3).sum())
            bg_act = C_AB if action == "Prediction changed" else C_GREY
            _cell(ws_triage, rtr, 1, rule_name, bold=True, align="left")
            _cell(ws_triage, rtr, 2, cnt_total, bg=bg_act)
            _cell(ws_triage, rtr, 3, cnt_t1,    bg=C_GB if cnt_t1 else None)
            _cell(ws_triage, rtr, 4, cnt_t2,    bg=C_AB if cnt_t2 else None)
            _cell(ws_triage, rtr, 5, cnt_t3,    bg=C_RB if cnt_t3 else None)
            _cell(ws_triage, rtr, 6, action,    align="left")
            rtr += 1

    # If has_gt: show improved/worsened per tier
    if has_gt and "outcome" in patch_df.columns and "triage_tier" in patch_df.columns:
        rtr += 1
        ws_triage.cell(row=rtr, column=1,
            value="OUTCOME BY TIER (requires GT)").font = Font(bold=True, size=10, color=C_BLUE)
        rtr += 1
        _hdr(ws_triage, rtr, 1, "Tier", bg=C_BLUE)
        _hdr(ws_triage, rtr, 2, "Improved", bg=C_BLUE)
        _hdr(ws_triage, rtr, 3, "Worsened", bg=C_BLUE)
        _hdr(ws_triage, rtr, 4, "No change", bg=C_BLUE)
        _hdr(ws_triage, rtr, 5, "Unknown (no GT)", bg=C_BLUE)
        rtr += 1
        for tier_num, tier_lbl, _, bg_t, ft_t in tier_defs:
            sub_t = patch_df[patch_df["triage_tier"] == tier_num]
            n_imp = int((sub_t["outcome"] == "improved").sum())
            n_wor = int((sub_t["outcome"] == "worsened").sum())
            n_noc = int((sub_t["outcome"] == "no_change").sum())
            n_unk = int((sub_t["outcome"] == "unknown").sum())
            _cell(ws_triage, rtr, 1, f"Tier {tier_num} {tier_lbl}", bold=True, align="left", bg=bg_t, ft=ft_t)
            _cell(ws_triage, rtr, 2, n_imp, bg=C_GB if n_imp else None, ft=C_GT if n_imp else C_BLACK)
            _cell(ws_triage, rtr, 3, n_wor, bg=C_RB if n_wor else None, ft=C_RT if n_wor else C_BLACK)
            _cell(ws_triage, rtr, 4, n_noc)
            _cell(ws_triage, rtr, 5, n_unk, bg=C_GREY)
            rtr += 1

    # Combo breakdown per tier (no GT path — uses subfolder-derived combo)
    if "triage_tier" in patch_df.columns and combo_df is not None and len(combo_df) > 0:
        _sc_combo = {}
        if "label_combo" in combo_df.columns and "slide_id" in combo_df.columns:
            _sc_combo = (
                combo_df[["slide_id", "label_combo"]]
                .drop_duplicates()
                .set_index("slide_id")["label_combo"]
                .to_dict()
            )
        if _sc_combo:
            rtr += 1
            ws_triage.cell(row=rtr, column=1,
                value="TRIAGE TIER BY SLIDE COMBO").font = Font(bold=True, size=10, color=C_BLUE)
            rtr += 1
            _hdr(ws_triage, rtr, 1, "Combo",       bg=C_BLUE)
            _hdr(ws_triage, rtr, 2, "N slides",    bg=C_BLUE)
            _hdr(ws_triage, rtr, 3, "Tier 1 N",    bg=C_BLUE)
            _hdr(ws_triage, rtr, 4, "Tier 2 N",    bg=C_BLUE)
            _hdr(ws_triage, rtr, 5, "Tier 3 N",    bg=C_BLUE)
            _hdr(ws_triage, rtr, 6, "N changed",   bg=C_BLUE)
            rtr += 1
            patch_df_c = patch_df.copy()
            patch_df_c["label_combo"] = patch_df_c["slide_id"].map(_sc_combo)
            COMBO_COLORS_TR = {
                "neg":     ("E6F1FB", "185FA5"),
                "pos":     ("FBEAF0", "993556"),
                "pos+neg": ("EEEDFE", "534AB7"),
                "unknown": ("F2F2F2", "444444"),
            }
            for combo_n in ["neg", "pos", "pos+neg", "unknown"]:
                sub_c = patch_df_c[patch_df_c["label_combo"] == combo_n]
                if len(sub_c) == 0:
                    continue
                n_sl_c = sub_c["slide_id"].nunique()
                t1c = int((sub_c["triage_tier"] == 1).sum())
                t2c = int((sub_c["triage_tier"] == 2).sum())
                t3c = int((sub_c["triage_tier"] == 3).sum())
                chgc = int(sub_c["changed"].sum())
                bg_c, ft_c = COMBO_COLORS_TR.get(combo_n, ("F2F2F2", "444444"))
                _cell(ws_triage, rtr, 1, combo_n, bold=True, align="left", bg=bg_c, ft=ft_c)
                _cell(ws_triage, rtr, 2, n_sl_c)
                _cell(ws_triage, rtr, 3, t1c, bg=C_GB if t1c else None, ft=C_GT if t1c else C_BLACK)
                _cell(ws_triage, rtr, 4, t2c, bg=C_AB if t2c else None)
                _cell(ws_triage, rtr, 5, t3c, bg=C_RB if t3c else None, ft=C_RT if t3c else C_BLACK)
                _cell(ws_triage, rtr, 6, chgc, bg=C_AB if chgc else None)
                rtr += 1

    _auto_w(ws_triage)

    for title, df, kwargs in [
        ("Patch Results",   patch_df,        dict(outcome_col="outcome", changed_col="changed", correct_col="pr_correct")),
        ("Slide Results",   slide_df_display, dict(outcome_col="outcome")),
        ("Changed Patches", changed_df,       dict(outcome_col="outcome", changed_col="changed",  correct_col="pr_correct")),
        ("Review Flags",    review_df,        dict(correct_col="ml_correct")),
        ("Errors Remaining",
         errors_df if has_gt else pd.DataFrame(),
         dict(outcome_col="outcome", changed_col="changed")),
    ]:
        ws = wb.create_sheet(title)
        ws.sheet_view.showGridLines = False
        if len(df) == 0:
            ws.cell(row=1, column=1, value=f"No data for: {title}")
        else:
            _write_df(ws, df, **kwargs)
        _auto_w(ws)
        if len(df) > 0:
            ws.freeze_panes = "C2"

    # ── By Class sheet ────────────────────────────────────────────
    wsc = wb.create_sheet("By Class")
    wsc.sheet_view.showGridLines = False
    if has_gt:
        rc = 1
        note_cls = wsc.cell(row=rc, column=1,
            value=(
                "Fair stats only. ML acc vs PR acc is NOT shown (different denominators). "
                "Outcome counts (improved/worsened/same) are fair: they measure what "
                "PyReason changed, not overall accuracy on unequal sets. "
                "Scenario A MCC = PR on same N as ML (apples-to-apples)."
            ))
        note_cls.font      = Font(italic=True, size=9, color="595959")
        note_cls.alignment = Alignment(wrap_text=True, horizontal="left")
        wsc.merge_cells(f"A{rc}:D{rc}")
        wsc.row_dimensions[rc].height = 36
        rc += 2

        for cls_name in sorted(patch_df["gt_label"].dropna().unique()):
            sub = patch_df[patch_df["gt_label"] == cls_name]
            n   = len(sub)
            sub_pr_firm = sub[sub["pr_zone"] == "firm"]
            n_pr_firm_c = len(sub_pr_firm)
            n_deferred  = int((sub["pr_zone"] == "deferred").sum())
            n_changed   = int(sub["changed"].sum())
            n_improved  = int((sub["outcome"] == "improved").sum())
            n_worsened  = int((sub["outcome"] == "worsened").sum())
            n_same      = n_changed - n_improved - n_worsened   # changed_same (no GT)
            n_flagged   = int(sub["needs_review"].sum())

            # Scenario A: PR correctness on patches that were ML-firm (same denominator)
            sub_ml_firm = sub[sub["ml_zone"] == "firm"]
            sub_pr_on_ml = sub[sub["image_name"].isin(sub_ml_firm["image_name"]) & (sub["pr_zone"] == "firm")]
            n_scen_a = len(sub_ml_firm)
            pr_scen_a_ok = int((sub_pr_on_ml["pr_correct"] == True).sum()) if len(sub_pr_on_ml) else 0
            ml_scen_a_ok = int((sub_ml_firm["ml_correct"] == True).sum()) if len(sub_ml_firm) else 0

            wsc.cell(row=rc, column=1,
                     value=f"GT Class: {cls_name}").font = Font(bold=True, size=11)
            rc += 1
            for ci, h in enumerate(["Metric", "Count", "Rate / note"], 1):
                _hdr(wsc, rc, ci, h, bg=C_BLUE)
            rc += 1
            items = [
                ("Total patches",                    n,            None,                                                      None),
                ("PR firm predictions",              n_pr_firm_c,  f"{n_pr_firm_c/n*100:.1f}% of class",                     C_AB),
                ("PR deferred (Mixed)",              n_deferred,   f"{n_deferred/n*100:.1f}% of class — routed to human",    C_TEAL),
                ("Changed by PR",                    n_changed,    f"{n_changed/n*100:.1f}% of class",                       C_AB),
                ("  Improved (PR fixed ML error)",   n_improved,   f"{n_improved/n_changed*100:.1f}% of changed" if n_changed else "0%", C_GB),
                ("  Worsened (PR broke ML correct)", n_worsened,   f"{n_worsened/n_changed*100:.1f}% of changed" if n_changed else "0%", C_RB),
                ("  Same (changed, no GT impact)",   n_same,       f"{n_same/n_changed*100:.1f}% of changed" if n_changed else "0%", C_GREY),
                ("Flagged for review",               n_flagged,    f"{n_flagged/n*100:.1f}% of class",                       C_AB),
                (f"Scen A: ML correct (N={n_scen_a})",   ml_scen_a_ok, f"{ml_scen_a_ok/n_scen_a*100:.1f}% of ML-firm" if n_scen_a else "—", C_AB),
                (f"Scen A: PR correct (same N={n_scen_a})", pr_scen_a_ok, f"{pr_scen_a_ok/n_scen_a*100:.1f}% of ML-firm" if n_scen_a else "—",
                 C_GB if pr_scen_a_ok >= ml_scen_a_ok else C_RB),
            ]
            for lbl, cnt, pct, bg in items:
                bold_row = "Improved" in lbl or "Worsened" in lbl
                ft_row   = C_GT if "Improved" in lbl else (C_RT if "Worsened" in lbl else C_BLACK)
                _cell(wsc, rc, 1, lbl, bold=bold_row, align="left")
                _cell(wsc, rc, 2, cnt, bg=bg, ft=ft_row if bold_row else C_BLACK, bold=bold_row)
                _cell(wsc, rc, 3, pct if pct else "", bg=bg)
                rc += 1
            rc += 1
    else:
        wsc.cell(row=1, column=1, value="No GT labels available.")
    _auto_w(wsc)
    wsc.column_dimensions["C"].width = 38

    # ── Stratified Breakdown sheet ────────────────────────────────
    ws_strat = wb.create_sheet("Stratified Breakdown")
    ws_strat.sheet_view.showGridLines = False

    if has_gt and combo_df is not None and len(combo_df) > 0:
        t = ws_strat.cell(row=1, column=1,
            value="Stratified Breakdown by Slide Label Combo  "
                  "(patch-level, binary G vs Gplus, FIRM predictions only)")
        t.font      = Font(bold=True, size=12, color="FFFFFF")
        t.fill      = _fill(C_NAVY)
        t.alignment = Alignment(horizontal="center", vertical="center")
        ws_strat.merge_cells("A1:S1")
        ws_strat.row_dimensions[1].height = 22

        hdrs = ["Slide subset", "N slides", "N GT patches",
                "N firm PR", "N defer PR", "Coverage PR",
                "Improved", "Worsened", "Same (no GT)",
                "PR acc (ScenA)", "ML acc (ScenA)", "Delta (ScenA)",
                "TP", "TN", "FP", "FN",
                "MCC (PR firm)", "Bal.Acc (PR firm)", "F2 (PR firm)"]
        for ci, h in enumerate(hdrs, 1):
            _hdr(ws_strat, 3, ci, h, bg=C_NAVY)

        note_strat = ws_strat.cell(row=2, column=1,
            value=(
                "PR metrics (MCC/BalAcc/F2) on PR-firm predictions only. "
                "'Scen A' columns = same N as ML (ML-firm patches) — apples-to-apples accuracy. "
                "ML acc (firm) vs PR acc (PR-firm) is NOT shown: different denominators. "
            ))
        note_strat.font = Font(italic=True, size=9, color="595959")
        ws_strat.merge_cells("A2:S2")

        BINARY_GT_S  = BINARY_LABELS
        COMBOS_ORD   = ["neg", "pos", "pos+neg"]
        COMBO_LABELS = {"neg": "neg-only slides",
                        "pos": "pos-only slides",
                        "pos+neg": "pos+neg slides"}
        COMBO_BG     = {"neg": "DDEEFF", "pos": "FFE8E8", "pos+neg": "E8E8FF"}

        # patch_df is the authoritative source for outcome, zones, correctness.
        # combo_df only provides label_combo + slide_id for filtering.
        # Build a slide_id -> label_combo map, then filter patch_df directly.
        _slide_combo_map = {}
        if "label_combo" in combo_df.columns and "slide_id" in combo_df.columns:
            _slide_combo_map = (
                combo_df[["slide_id", "label_combo"]]
                .drop_duplicates()
                .set_index("slide_id")["label_combo"]
                .to_dict()
            )

        # patch_df enriched with label_combo column for stratified use
        patch_df_strat = patch_df.copy()
        patch_df_strat["label_combo"] = patch_df_strat["slide_id"].map(_slide_combo_map)

        def _strat_row(slide_ids, label):
            # Use patch_df as source — has outcome, image_name, all zones
            binary = patch_df_strat[
                patch_df_strat["slide_id"].isin(slide_ids) &
                patch_df_strat["gt_label"].isin(BINARY_GT_S)
            ].copy()
            if len(binary) == 0:
                return None
            n_slides  = binary["slide_id"].nunique()
            n_patches = len(binary)

            # PR firm / deferred
            pr_firm   = binary[binary["pr_zone"] == "firm"]
            n_pr_f    = len(pr_firm)
            n_pr_d    = n_patches - n_pr_f
            cov_pr    = f"{n_pr_f/n_patches*100:.1f}%" if n_patches else "—"

            # Outcome counts (fair — patch-level improve/worsen/same)
            n_imp = int((binary["outcome"] == "improved").sum())
            n_wor = int((binary["outcome"] == "worsened").sum())
            n_sam = int(binary["changed"].sum()) - n_imp - n_wor

            # Scenario A: restrict BOTH to ML-firm patches (same denominator)
            ml_firm_b  = binary[binary["ml_zone"] == "firm"]
            pr_on_ml_b = binary[
                binary["image_name"].isin(ml_firm_b["image_name"]) &
                (binary["pr_zone"] == "firm")
            ]
            n_scen_a  = len(ml_firm_b)
            ml_ok_a   = int((ml_firm_b["ml_correct"] == True).sum()) if n_scen_a else 0
            pr_ok_a   = int((pr_on_ml_b["pr_correct"] == True).sum()) if len(pr_on_ml_b) else 0
            ml_acc_a  = f"{ml_ok_a/n_scen_a*100:.1f}%" if n_scen_a else "—"
            pr_acc_a  = f"{pr_ok_a/n_scen_a*100:.1f}%" if n_scen_a else "—"
            delta_a   = f"{(pr_ok_a - ml_ok_a)/n_scen_a*100:+.1f}%" if n_scen_a else "—"

            # PR-firm metrics (MCC/BalAcc/F2)
            gt_i   = [_label_to_int(v) for v in pr_firm["gt_label"]]
            pr_i   = [_label_to_int(v) for v in pr_firm["final_predicted"]]
            gt_c, pr_c = [], []
            for g, p in zip(gt_i, pr_i):
                if g is None or p is None:
                    continue
                gt_c.append(g); pr_c.append(p)
            m = _cls_metrics(gt_c, pr_c)
            return dict(
                label=label, n_slides=n_slides, n_patches=n_patches,
                n_pr_firm=n_pr_f, n_pr_deferred=n_pr_d, cov_pr=cov_pr,
                n_improved=n_imp, n_worsened=n_wor, n_same=n_sam,
                pr_acc_a=pr_acc_a, ml_acc_a=ml_acc_a, delta_a=delta_a,
                tp=m["tp"], tn=m["tn"], fp=m["fp"], fn=m["fn"],
                mcc=m["mcc"], bal_acc=m["bal_acc"], f2=m["f2"],
            )

        rr = 4
        strat_rows = []
        for combo in COMBOS_ORD:
            combo_slide_ids = [sid for sid, lc in _slide_combo_map.items() if lc == combo]
            if not combo_slide_ids:
                continue
            row = _strat_row(combo_slide_ids, COMBO_LABELS.get(combo, combo))
            if row is None:
                continue
            strat_rows.append(row)
            bg = COMBO_BG.get(combo, "F2F2F2")
            vals = [
                row["label"], row["n_slides"], row["n_patches"],
                row["n_pr_firm"], row["n_pr_deferred"], row["cov_pr"],
                row["n_improved"], row["n_worsened"], row["n_same"],
                row["pr_acc_a"], row["ml_acc_a"], row["delta_a"],
                row["tp"], row["tn"], row["fp"], row["fn"],
                round(row["mcc"], 4), round(row["bal_acc"], 4), round(row["f2"], 4),
            ]
            for ci, v in enumerate(vals, 1):
                _cell(ws_strat, rr, ci, v, bg=bg,
                      fmt="0.0000" if ci >= 17 else None)
            rr += 1

        # ALL row
        all_slide_ids = list(patch_df_strat[
            patch_df_strat["gt_label"].isin(BINARY_GT_S)
        ]["slide_id"].unique())
        r_all = _strat_row(all_slide_ids, "ALL slides")
        if r_all:
            strat_rows.append(r_all)
            ws_strat.cell(row=rr, column=1, value=""); rr += 1
            vals = [
                r_all["label"], r_all["n_slides"], r_all["n_patches"],
                r_all["n_pr_firm"], r_all["n_pr_deferred"], r_all["cov_pr"],
                r_all["n_improved"], r_all["n_worsened"], r_all["n_same"],
                r_all["pr_acc_a"], r_all["ml_acc_a"], r_all["delta_a"],
                r_all["tp"], r_all["tn"], r_all["fp"], r_all["fn"],
                round(r_all["mcc"], 4), round(r_all["bal_acc"], 4), round(r_all["f2"], 4),
            ]
            for ci, v in enumerate(vals, 1):
                _cell(ws_strat, rr, ci, v, bold=True, bg=C_GB,
                      fmt="0.0000" if ci >= 17 else None)

        if strat_rows:
            pd.DataFrame(strat_rows).to_csv(
                f"{output_dir}/stratified_breakdown.csv", index=False)
            print(f"  [Stratified] CSV: {output_dir}/stratified_breakdown.csv")

        _auto_w(ws_strat)
        ws_strat.freeze_panes = "B4"
    else:
        ws_strat.cell(row=1, column=1,
            value="No GT / combo data — stratified breakdown requires GT labels.")

    out = os.path.join(output_dir, "evaluation_results.xlsx")
    wb.save(out)
    print(f"  [Excel] Saved: {out}")


# ─────────────────────────────────────────────────────────────────
#  Console report
# ─────────────────────────────────────────────────────────────────

def _print_report(
    pm_ml, pm_pr, sm_ml, sm_pr,
    patch_df, changed_df, slide_df,
    n_individual, n_sparse, n_grid, n_eligible,
    n_r4_total, n_r4_errors, n_r4_false_alarm, n_unflagged,
    has_gt, output_dir,
    mixed_stats=None, combo_df=None, deferral_stats=None,
    fair_comparisons=None,
):
    W = 70
    L = []
    def h(c="="): L.append(c * W)
    def ln(t=""):  L.append(t)

    h(); ln("PYREASON EVALUATION REPORT")
    ln(f"  Total patches : {len(patch_df)}")
    ln(f"  Total slides  : {len(slide_df)}")
    ln(f"  Has GT labels : {has_gt}")
    h()

    ln(f"\nIMAGE COVERAGE")
    ln(f"  Individual (PyReason cannot fire) : {n_individual}")
    ln(f"  Sparse grid (too few patches)     : {n_sparse}")
    ln(f"  Grid patches (PyReason eligible)  : {n_grid}")

    n_ch = int(patch_df["changed"].sum())
    n_rv = int(patch_df["needs_review"].sum())
    ln(f"\nPYREASON ACTIONS (on {n_grid} eligible grid patches)")
    ln(f"  Changed (Rule 1/2) : {n_ch}")
    ln(f"  Flagged (Rule 4)   : {n_rv}")
    if n_ch > 0:
        for rule, cnt in patch_df[patch_df["changed"]]["rule_applied"].value_counts().items():
            ln(f"    {rule}: {cnt}")

    if has_gt and n_ch > 0:
        n_imp = int((patch_df[patch_df["changed"]]["outcome"] == "improved").sum())
        n_wor = int((patch_df[patch_df["changed"]]["outcome"] == "worsened").sum())
        ln(f"\n  Of {n_ch} changed: {n_imp} improved, {n_wor} worsened")

    # ── TABLE B: Deferral ─────────────────────────────────────────
    if has_gt and deferral_stats:
        ds = deferral_stats
        ln(f"\nTABLE B — DEFERRAL & COVERAGE  (safety metric)")
        ln(f"  {'Metric':<30} {'ML only':>12} {'PyReason':>12}")
        ln("  " + "-" * 56)
        n_b = ds.get("n_total", 1)
        ln(f"  {'Total G/Gplus GT patches':<30} {n_b:>12}")
        ln(f"  {'Firm predictions':<30} {ds.get('n_ml_firm',0):>12} {ds.get('n_pr_firm',0):>12}")
        ln(f"  {'Coverage %':<30} "
           f"{ds.get('ml_coverage',0)*100:>11.1f}% "
           f"{ds.get('pr_coverage',0)*100:>11.1f}%")
        ln(f"  {'Deferred (Mixed output)':<30} "
           f"{ds.get('n_ml_deferred',0):>12} "
           f"{ds.get('n_pr_deferred',0):>12}")
        ln(f"  {'Deferral rate %':<30} "
           f"{ds.get('ml_deferral_rate',0)*100:>11.1f}% "
           f"{ds.get('pr_deferral_rate',0)*100:>11.1f}%")
        ln(f"  Note: Deferred = Mixed prediction on G/Gplus GT patch.")
        ln(f"        This is a SAFE outcome (routed to human), NOT a wrong answer.")

    # ── TABLE C: Fair comparisons ─────────────────────────────────
    fc = fair_comparisons or {}
    if has_gt and fc:
        ln(f"\nTABLE C — FAIR COMPARISON  (denominator analysis)")
        ln(f"  The reported Table A uses ML N≠PR N (PR tests on more/harder patches).")
        ln(f"  Below are three valid comparisons that address this.")

        sa = fc.get("scenario_A", {})
        if sa:
            ln(f"\n  Scenario A — Same N={sa.get('n',0)} (ML-firm patches only):")
            ln(f"  {'Metric':<14} {'ML (firm only)':>16} {'PR (same N)':>14} {'Delta':>10}")
            ln("  " + "-" * 56)
            for key, label in [("mcc","MCC"), ("bal_acc","Bal.Acc"),
                                ("recall","Recall"), ("f1","F1")]:
                ml_v = sa.get("ml",{}).get(key, 0)
                pr_v = sa.get("pr",{}).get(key, 0)
                ln(f"  {label:<14} {ml_v:>16.4f} {pr_v:>14.4f} {pr_v-ml_v:>+10.4f}")
            ln(f"  → delta MCC={sa.get('delta_mcc',0):+.4f}  "
               f"(positive = PR helped on ML's own committed set)")

        sb = fc.get("scenario_B", {})
        if sb:
            ln(f"\n  Scenario B — All {sb.get('n',0)} patches, Mixed=wrong (coverage penalised):")
            ln(f"  {'Metric':<14} {'ML (forced)':>16} {'PR (forced)':>14} {'Delta':>10}")
            ln("  " + "-" * 56)
            for key, label in [("mcc","MCC"), ("bal_acc","Bal.Acc"),
                                ("recall","Recall"), ("f1","F1")]:
                ml_v = sb.get("ml",{}).get(key, 0)
                pr_v = sb.get("pr",{}).get(key, 0)
                ln(f"  {label:<14} {ml_v:>16.4f} {pr_v:>14.4f} {pr_v-ml_v:>+10.4f}")
            ln(f"  → delta MCC={sb.get('delta_mcc',0):+.4f}  "
               f"(penalises ML's deferred patches; shows full coverage gain)")

        sc = fc.get("scenario_C", {})
        if sc and sc.get("n", 0) > 0:
            acc = sc.get("accuracy", 0)
            ln(f"\n  Scenario C — Extra {sc.get('n',0)} patches PR committed (ML deferred):")
            ln(f"    Correct: {sc.get('n_correct',0)}  Wrong: {sc.get('n_wrong',0)}  "
               f"Accuracy: {acc*100:.1f}%")
            ln(f"    {'≥75% = good coverage quality' if acc>=0.75 else '<60% = harmful commits'}")

    # ── TABLE A: Firm metrics ─────────────────────────────────────
    if has_gt and pm_ml:
        ds = deferral_stats or {}
        n_ml_f = ds.get("n_ml_firm", pm_ml.get("n", 0))
        n_pr_f = ds.get("n_pr_firm", pm_pr.get("n", 0))
        ln(f"\nTABLE A — DIAGNOSTIC PERFORMANCE  "
           f"(firm only: ML N={n_ml_f}, PR N={n_pr_f})")
        ln(f"  {'Metric':<16} {'ML-only':>10} {'PyReason':>10} {'Delta':>10} {'% Chg':>8}")
        ln("  " + "-" * 58)
        for key, label in [
            ("mcc",       "MCC"),
            ("bal_acc",   "Bal.Acc"),
            ("recall",    "Recall"),
            ("precision", "Precision"),
            ("f1",        "F1"),
            ("f2",        "F2"),
            ("acc",       "Accuracy"),
        ]:
            ml_v = pm_ml.get(key, 0); pr_v = pm_pr.get(key, 0)
            delta = pr_v - ml_v
            pct = (delta / ml_v * 100) if ml_v else 0
            ln(f"  {label:<16} {ml_v:>10.4f} {pr_v:>10.4f} {delta:>+10.4f} {pct:>+7.2f}%")

    # ── Confusion matrix ──────────────────────────────────────────
    if has_gt and pm_ml:
        ln(f"\nCONFUSION MATRIX  (firm predictions only)")
        ln(f"  {'':16} {'TP':>5} {'TN':>5} {'FP':>5} {'FN':>5}  {'N firm':>8}")
        ln("  " + "-" * 54)
        for lbl, pm in [("ML only", pm_ml), ("PyReason", pm_pr)]:
            n_f = int(pm.get("tp",0)+pm.get("tn",0)+pm.get("fp",0)+pm.get("fn",0))
            ln(f"  {lbl:<16} "
               f"{pm.get('tp',0):>5} {pm.get('tn',0):>5} "
               f"{pm.get('fp',0):>5} {pm.get('fn',0):>5}  {n_f:>8}")
        ln(f"  Positive class = Gplus. Deferred patches excluded from this table.")

    # ── Rule 4 ────────────────────────────────────────────────────
    if has_gt:
        ln(f"\nRULE 4 FLAG QUALITY")
        ln(f"  Total flagged               : {n_r4_total}")
        ln(f"  True errors (right to flag) : {n_r4_errors}")
        ln(f"  False alarms (unnecessary)  : {n_r4_false_alarm}")
        ln(f"\nSILENT FAILURES (firm wrong AND not flagged)")
        ln(f"  Note: Deferred-wrong patches are NOT counted here — already routed to human.")
        ln(f"  Silent failures             : {n_unflagged}")

    # ── Mixed GT (Zone 3) ─────────────────────────────────────────
    if mixed_stats and mixed_stats.get("n_mixed", 0) > 0:
        mx = mixed_stats
        ln(f"\nMIXED GT PATCHES (screening quality)")
        ln(f"  Total Mixed GT patches      : {mx['n_mixed']}")
        ln(f"  CNN predicted Mixed         : {mx.get('n_ml_predicted_mixed',0)} "
           f"({mx.get('ml_mixed_pred_rate',0)*100:.1f}%)")
        ln(f"  ML flagged (Rule 4)         : {mx['n_flagged_ml']} "
           f"({mx['flag_rate_ml']*100:.1f}%)")
        ln(f"  PR flagged (Rule 4)         : {mx['n_flagged_pr']} "
           f"({mx['flag_rate_pr']*100:.1f}%)")
        better = mx["flag_rate_pr"] >= mx["flag_rate_ml"]
        ln(f"  Screening improvement       : {'YES ✓' if better else 'NO ✗'}")

    # ── Slide-level ───────────────────────────────────────────────
    if sm_ml:
        n_referred = int((slide_df["pr_diagnosis"] == "REFER").sum()) if len(slide_df) else 0
        ln(f"\nSLIDE-LEVEL METRICS  (firm diagnoses only — REFER excluded)")
        ln(f"  Slides referred (REFER): {n_referred} "
           f"({n_referred/len(slide_df)*100:.1f}%)" if len(slide_df) else "")
        ln(f"  {'Metric':<12} {'ML-only':>10} {'PyReason':>10} {'Delta':>10}")
        ln("  " + "-" * 46)
        for key, label in [("mcc","MCC"), ("bal_acc","Bal.Acc"),
                            ("f1","F1"), ("f2","F2"),
                            ("acc","Accuracy"), ("precision","Precision"), ("recall","Recall")]:
            ml_v = sm_ml.get(key, 0); pr_v = sm_pr.get(key, 0)
            ln(f"  {label:<12} {ml_v:>10.4f} {pr_v:>10.4f} {pr_v-ml_v:>+10.4f}")

    # ── Stratified ────────────────────────────────────────────────
    if has_gt and combo_df is not None and len(combo_df) > 0:
        BG = BINARY_LABELS
        COMBOS_C = [("neg", "neg-only  "), ("pos", "pos-only  "), ("pos+neg", "pos+neg   ")]

        # Build slide_id -> label_combo from combo_df, then filter patch_df
        _sc_map = {}
        if "label_combo" in combo_df.columns and "slide_id" in combo_df.columns:
            _sc_map = (
                combo_df[["slide_id","label_combo"]]
                .drop_duplicates()
                .set_index("slide_id")["label_combo"]
                .to_dict()
            )

        def _strat_c(slide_ids, label):
            binary  = patch_df[
                patch_df["slide_id"].isin(slide_ids) &
                patch_df["gt_label"].isin(BG)
            ]
            if len(binary) == 0: return None
            pr_firm = binary[binary["pr_zone"] == "firm"]
            n_sl    = binary["slide_id"].nunique()
            n_p     = len(binary)
            n_f     = len(pr_firm)
            pr_ok   = int((pr_firm["pr_correct"] == True).sum())
            pr_acc  = f"{pr_ok/n_f*100:.1f}%" if n_f else "—"
            cov     = f"{n_f/n_p*100:.1f}%"   if n_p else "—"
            gt_i    = [_label_to_int(v) for v in pr_firm["gt_label"]]
            pr_i    = [_label_to_int(v) for v in pr_firm["final_predicted"]]
            gt_c, pr_c_l = [], []
            for g, p in zip(gt_i, pr_i):
                if g is None or p is None: continue
                gt_c.append(g); pr_c_l.append(p)
            m = _cls_metrics(gt_c, pr_c_l)
            return (label, n_sl, n_p, n_f, cov, pr_acc,
                    f"{m['mcc']:.4f}", f"{m['bal_acc']:.4f}", f"{m['f2']:.4f}")

        ln(f"\nSTRATIFIED BREAKDOWN  (firm PR predictions, binary G vs Gplus)")
        ln(f"  {'Subset':<14} {'Slides':>6} {'GT patches':>10} "
           f"{'N firm':>7} {'Cov%':>7} {'PR acc':>8} "
           f"{'MCC':>7} {'BalAcc':>8} {'F2':>7}")
        ln("  " + "-" * 78)
        for combo, label in COMBOS_C:
            ids = [sid for sid, lc in _sc_map.items() if lc == combo]
            if not ids: continue
            row = _strat_c(ids, label)
            if row:
                ln(f"  {row[0]:<14} {row[1]:>6} {row[2]:>10} "
                   f"{row[3]:>7} {row[4]:>7} {row[5]:>8} "
                   f"{row[6]:>7} {row[7]:>8} {row[8]:>7}")
        all_ids = list(patch_df[patch_df["gt_label"].isin(BG)]["slide_id"].unique())
        r_all = _strat_c(all_ids, "ALL       ")
        if r_all:
            ln("  " + "-" * 78)
            ln(f"  {r_all[0]:<14} {r_all[1]:>6} {r_all[2]:>10} "
               f"{r_all[3]:>7} {r_all[4]:>7} {r_all[5]:>8} "
               f"{r_all[6]:>7} {r_all[7]:>8} {r_all[8]:>7}")

    h()
    ln("  evaluation_results.xlsx — full report")
    ln("  TABLE A = diagnostic metrics on FIRM predictions only")
    ln("  TABLE B = deferral/coverage (safety metric)")
    h()
    text = "\n".join(L)
    print("\n" + text)
    with open(f"{output_dir}/report.txt", "w", encoding="utf-8") as f:
        f.write(text)