"""
src/pyreason_engine.py
======================
Pure-Python implementation of the PyReason slide+section rules.

Rule naming (canonical):
  Rule 0   — HIGH-conf firm patch → immutable anchor for correction rules.
             It is not changed by Rule 1/2. However, if it is spatially
             inconsistent with the slide majority, it may still be included
             in Rule 3 as a minority-firm cluster candidate.
  Rule 1a  — Mixed patch in dominant SECTION/WINDOW → section/window majority.
  Rule 1b  — Minority firm patch in dominant SECTION/WINDOW → section/window majority
             (requires dominant_strict — higher evidence bar than Rule 1a).
  Rule 2   — Uncertain/LOW patch + >=N HIGH neighbours agree → neighbour class.
             Fires only when Rule 1a/1b did NOT already correct the patch.
  Rule 3   — Cluster of 3+ adjacent uncertainty/minority patches → flag as
             spatially coherent disagreement/uncertainty zone.
             Candidates include:
               (a) Mixed/deferred predictions
               (b) LOW-confidence predictions
               (c) firm minority predictions of any confidence tier, including HIGH
             Rule 3 does not change the predicted class.
  Rule 4   — Isolated LOW-conf or Mixed patch with no prior Rule 1/2 correction
             and no Rule 3 cluster flag → flag for human review.

Priority chain (strictly enforced; no stacking):
  Rule 1/2 correction  >  Rule 3 cluster flag  >  Rule 4 isolated flag

No tier assignment is performed in this file. The engine only returns:
  final_predicted, rule_applied, changed, needs_review, correction_confidence.

Context mode (config.SECTION_MODE):
  "quadrant"  (default) — slide split into N_SECTIONS fixed quadrants (2×2 or 3×3).
  "window"    — per-patch sliding W_ROWS × W_COLS window.

Rule execution order per patch:
  Rule 0 correction guard → Rule 1a/1b → Rule 2 → Rule 3 → Rule 4
"""

from __future__ import annotations

from typing import Dict, List, Set, Tuple

import config

UNCERTAIN_CLASSES = {"Mixed", "Mix", "mix", "mixed"}


# ─────────────────────────────────────────────────────────────────
#  Grid helpers
# ─────────────────────────────────────────────────────────────────

def _build_neighbor_index(image_info: Dict) -> Dict[str, List[str]]:
    """4-connected grid neighbour index (up/down/left/right only)."""
    grid: Dict[Tuple, str] = {}
    for iname, info in image_info.items():
        if info["row"] is not None and info["col"] is not None:
            grid[(info["slide_id"], info["row"], info["col"])] = iname

    neighbors: Dict[str, List[str]] = {}
    for iname, info in image_info.items():
        if info["row"] is None or info["col"] is None:
            neighbors[iname] = []
            continue
        r, c, sid = info["row"], info["col"], info["slide_id"]
        neighbors[iname] = [
            grid[(sid, r + dr, c + dc)]
            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]
            if (sid, r + dr, c + dc) in grid
        ]
    return neighbors


def _find_clusters(patches: List[str], neighbors: Dict[str, List[str]],
                   min_size: int) -> List[List[str]]:
    """BFS connected-component search; returns components >= min_size."""
    patch_set = set(patches)
    visited: Set[str] = set()
    clusters: List[List[str]] = []
    for start in patches:
        if start in visited:
            continue
        component: List[str] = []
        queue = [start]
        while queue:
            node = queue.pop()
            if node in visited:
                continue
            visited.add(node)
            component.append(node)
            for nb in neighbors.get(node, []):
                if nb in patch_set and nb not in visited:
                    queue.append(nb)
        if len(component) >= min_size:
            clusters.append(component)
    return clusters


# ─────────────────────────────────────────────────────────────────
#  Slide vote  (diagnostics + Rule 3 reference)
# ─────────────────────────────────────────────────────────────────

def compute_slide_vote(slides: Dict, raw: Dict) -> Dict:
    """
    Per-slide majority vote. Mixed patches abstain.
    Result used for:
      - diagnostics / logging
      - Rule 3 cluster detection (reference majority for minority labelling)
    NOT used to correct individual patch predictions.
    """
    MIN_PATCHES = config.MIN_PATCHES_FOR_VOTE
    vote = {}

    for sid, names in slides.items():
        preds  = [raw[n] for n in names if n in raw]
        firm   = [p for p in preds if p["predicted"] not in UNCERTAIN_CLASSES]
        total  = len(preds)
        n_firm = len(firm)

        n_gp = sum(1 for p in firm if p["predicted"] == "Gplus")
        n_g  = n_firm - n_gp

        if n_firm == 0:      maj, maj_n = "TIE", 0
        elif n_gp > n_g:     maj, maj_n = "Gplus", n_gp
        elif n_g > n_gp:     maj, maj_n = "G", n_g
        else:                maj, maj_n = "TIE", n_g

        ratio          = maj_n / n_firm if n_firm > 0 else 0.0
        n_mixed        = total - n_firm
        mixed_fraction = n_mixed / total if total > 0 else 0.0

        vote[sid] = dict(
            n_G=n_g, n_Gplus=n_gp, n_mixed=n_mixed,
            total=total, n_firm=n_firm,
            majority=maj, ratio=ratio,
            mixed_fraction=mixed_fraction,
            is_standalone=(total == 1),
            is_sparse=(n_firm < MIN_PATCHES),
            dominant=(
                total > 1
                and n_firm >= MIN_PATCHES
                and maj != "TIE"
                and ratio >= config.MAJORITY_RATIO_THR
            ),
        )
    return vote


# ─────────────────────────────────────────────────────────────────
#  Context vote helpers
# ─────────────────────────────────────────────────────────────────

def _make_section_vote_entry(preds: List[Dict],
                              MIN_PATCHES: int,
                              MIN_PATCHES_MINOR: int,
                              SECTION_RATIO: float,
                              RATIO_MINOR: float,
                              SECTION_MAX_MX: float) -> Dict:
    """
    Shared vote aggregation used by both quadrant and window modes.

    dominant        — threshold for Rule 1a (Mixed resolution, looser bar)
    dominant_strict — threshold for Rule 1b (minority firm flip, stricter bar)
    """
    firm   = [p for p in preds if p["predicted"] not in UNCERTAIN_CLASSES]
    total  = len(preds)
    n_firm = len(firm)
    n_gp   = sum(1 for p in firm if p["predicted"] == "Gplus")
    n_g    = n_firm - n_gp

    if n_firm == 0:      maj, maj_n = "TIE", 0
    elif n_gp > n_g:     maj, maj_n = "Gplus", n_gp
    elif n_g > n_gp:     maj, maj_n = "G", n_g
    else:                maj, maj_n = "TIE", n_g

    ratio    = maj_n / n_firm if n_firm > 0 else 0.0
    mix_frac = (total - n_firm) / total if total > 0 else 0.0

    return dict(
        majority=maj, ratio=ratio,
        n_firm=n_firm, mixed_fraction=mix_frac,
        dominant=(
            n_firm >= MIN_PATCHES
            and maj != "TIE"
            and ratio >= SECTION_RATIO
            and mix_frac <= SECTION_MAX_MX
        ),
        dominant_strict=(
            n_firm >= MIN_PATCHES_MINOR
            and maj != "TIE"
            and ratio >= RATIO_MINOR
            and mix_frac <= SECTION_MAX_MX
        ),
    )


def compute_section_vote(slides: Dict, raw: Dict, image_info: Dict) -> Dict:
    """
    QUADRANT MODE — divide each slide into N_SECTIONS quadrants (default 4 = 2×2).
    Each section is voted independently. Stamps image_info[patch]["section_id"].

    Returns:
        {slide_id: {section_id: {majority, ratio, n_firm, mixed_fraction,
                                  dominant, dominant_strict}}}
    """
    n_sections     = getattr(config, "N_SECTIONS", 4)
    sq             = max(1, int(n_sections ** 0.5))
    MIN_PATCHES    = getattr(config, "SECTION_MIN_PATCHES", 2)
    SECTION_RATIO  = getattr(config, "SECTION_MAJORITY_RATIO_THR",
                             config.MAJORITY_RATIO_THR)
    SECTION_MAX_MX = getattr(config, "SECTION_MAX_MIXED_FRACTION",
                             getattr(config, "MAX_MIXED_FRACTION_FOR_RESOLVE", 0.20))
    MIN_PATCHES_MINOR = getattr(config, "SECTION_MINORITY_MIN_PATCHES",
                                max(MIN_PATCHES, 3))
    RATIO_MINOR       = getattr(config, "SECTION_MINORITY_RATIO_THR",
                                min(config.MAJORITY_RATIO_THR + 0.15, 0.90))

    sec_votes: Dict = {}

    for sid, names in slides.items():
        grid_patches = [
            (n, image_info[n]["row"], image_info[n]["col"])
            for n in names
            if n in raw and n in image_info
            and image_info[n].get("row") is not None
        ]
        if not grid_patches:
            sec_votes[sid] = {}
            continue

        rows     = sorted(r for _, r, _ in grid_patches)
        cols     = sorted(c for _, _, c in grid_patches)
        rmin, rmax = rows[0], rows[-1]
        cmin, cmax = cols[0], cols[-1]
        row_span = max(rmax - rmin, 1)
        col_span = max(cmax - cmin, 1)

        by_sec: Dict[int, list] = {}
        for n, r, c in grid_patches:
            row_bin = min(int((r - rmin) / row_span * sq), sq - 1)
            col_bin = min(int((c - cmin) / col_span * sq), sq - 1)
            sec     = row_bin * sq + col_bin
            image_info[n]["section_id"] = sec
            by_sec.setdefault(sec, []).append(n)

        sec_votes[sid] = {}
        for sec_id, sec_names in by_sec.items():
            preds = [raw[n] for n in sec_names if n in raw]
            sec_votes[sid][sec_id] = _make_section_vote_entry(
                preds, MIN_PATCHES, MIN_PATCHES_MINOR,
                SECTION_RATIO, RATIO_MINOR, SECTION_MAX_MX,
            )

    return sec_votes


def compute_window_vote(slides: Dict, raw: Dict, image_info: Dict) -> Dict:
    """
    WINDOW MODE — each patch gets its own vote from a W_ROWS × W_COLS rectangle
    of grid positions centred on it.

    Returns:
        {patch_name: {majority, ratio, n_firm, mixed_fraction,
                       dominant, dominant_strict}}
    """
    W_ROWS = getattr(config, "WINDOW_ROWS", 4)
    W_COLS = getattr(config, "WINDOW_COLS", 3)

    MIN_PATCHES    = getattr(config, "SECTION_MIN_PATCHES", 2)
    SECTION_RATIO  = getattr(config, "SECTION_MAJORITY_RATIO_THR",
                             config.MAJORITY_RATIO_THR)
    SECTION_MAX_MX = getattr(config, "SECTION_MAX_MIXED_FRACTION",
                             getattr(config, "MAX_MIXED_FRACTION_FOR_RESOLVE", 0.20))
    MIN_PATCHES_MINOR = getattr(config, "SECTION_MINORITY_MIN_PATCHES",
                                max(MIN_PATCHES, 3))
    RATIO_MINOR       = getattr(config, "SECTION_MINORITY_RATIO_THR",
                                min(config.MAJORITY_RATIO_THR + 0.15, 0.90))

    grid: Dict[Tuple, str] = {}
    for n, info in image_info.items():
        if info.get("row") is not None and info.get("col") is not None:
            grid[(info["slide_id"], info["row"], info["col"])] = n

    hr = W_ROWS // 2
    hc = W_COLS // 2

    win_votes: Dict[str, Dict] = {}

    for n, info in image_info.items():
        r, c, sid = info.get("row"), info.get("col"), info.get("slide_id")
        if r is None or c is None or sid is None:
            win_votes[n] = dict(
                majority="TIE", ratio=0.0, n_firm=0, mixed_fraction=0.0,
                dominant=False, dominant_strict=False,
            )
            continue

        window_names = [
            grid[(sid, r + dr, c + dc)]
            for dr in range(-hr, W_ROWS - hr)
            for dc in range(-hc, W_COLS - hc)
            if (sid, r + dr, c + dc) in grid
        ]
        preds = [raw[wn] for wn in window_names if wn in raw]
        win_votes[n] = _make_section_vote_entry(
            preds, MIN_PATCHES, MIN_PATCHES_MINOR,
            SECTION_RATIO, RATIO_MINOR, SECTION_MAX_MX,
        )

    return win_votes


# ─────────────────────────────────────────────────────────────────
#  Rule engine
# ─────────────────────────────────────────────────────────────────

def run_pyreason(
    slides: Dict,
    raw: Dict,
    image_info: Dict,
    slide_vote: Dict,
) -> Dict:
    """
    Apply all rules and return a corrections dict:
        {image_name: {"rule": str, "new_class": str|None, "needs_review": bool}}

    Rule priority chain (strictly enforced — no stacking):
        Rule 1/2 correction  >  Rule 3 cluster flag  >  Rule 4 isolated flag

    Rule 3 cluster candidates:
        (a) Mixed/deferred predictions
        (b) LOW-confidence predictions
        (c) firm minority predictions of any confidence tier, including HIGH

    Rule 0 is a correction guard only. HIGH-confidence firm patches are not
    changed by Rule 1/2, but HIGH-confidence firm minority patches can still be
    flagged by Rule 3 if they belong to a spatial cluster.

    Rule 4 fires ONLY when no prior rule entry exists for the patch.
    ANY prior rule (correction OR cluster flag) suppresses Rule 4 entirely.

    No tier assignment is performed here; downstream evaluation code may map
    rule_applied/changed/needs_review to tiers if needed.

    Context mode (config.SECTION_MODE):
        "quadrant" (default) — fixed N_SECTIONS quadrant per slide
        "window"             — per-patch sliding W_ROWS × W_COLS window
    """
    corrections: Dict = {}

    nb_index = _build_neighbor_index(image_info)

    # ── Choose context mode ───────────────────────────────────────
    section_mode = str(getattr(config, "SECTION_MODE", "quadrant")).lower()

    if section_mode == "window":
        win_vote  = compute_window_vote(slides, raw, image_info)
        for n, info in image_info.items():
            info.setdefault("section_id", 0)
        print(f"  [Context] Window mode  ({getattr(config, 'WINDOW_ROWS', 4)}"
              f"×{getattr(config, 'WINDOW_COLS', 3)} patches per patch)")
    else:
        sec_vote  = compute_section_vote(slides, raw, image_info)
        win_vote  = None
        print(f"  [Context] Quadrant mode  (N_SECTIONS={getattr(config, 'N_SECTIONS', 4)})")

    # Mark non-grid slides
    for sid, sv in slide_vote.items():
        names    = [n for n in slides.get(sid, []) if n in raw]
        has_grid = any(image_info[n]["row"] is not None
                       for n in names if n in image_info)
        if not has_grid:
            sv["is_standalone"] = True
            sv["dominant"]      = False

    # ── Diagnostics ──────────────────────────────────────────────
    n_dom    = sum(1 for sv in slide_vote.values() if sv["dominant"])
    n_sparse = sum(1 for sv in slide_vote.values()
                   if sv.get("is_sparse") and not sv.get("is_standalone"))
    n_alone  = sum(1 for sv in slide_vote.values() if sv.get("is_standalone"))
    n_mixed  = sum(1 for p in raw.values() if p["predicted"] in UNCERTAIN_CLASSES)
    n_min    = sum(
        1 for sid, sv in slide_vote.items() if sv["dominant"]
        for n in slides.get(sid, [])
        if n in raw
        and raw[n]["predicted"] not in UNCERTAIN_CLASSES
        and raw[n]["predicted"] != sv["majority"]
    )
    print(f"  [Rules] Slides: {len(slide_vote)} total | {n_dom} dominant | "
          f"{n_sparse} sparse | {n_alone} standalone (skipped)")
    print(f"  [Rules] Mixed patches: {n_mixed}   Minority firm: {n_min}")

    # ── Counters ─────────────────────────────────────────────────
    r1a_count = {"G": 0, "Gplus": 0}
    r1b_count = {"G": 0, "Gplus": 0}
    r2_count  = {"G": 0, "Gplus": 0}
    r3_count  = 0
    r4_count = 0   # flag-only: isolated LOW/Mixed not corrected or cluster-flagged

    anchor_conf = getattr(config, "ANCHOR_CONF_THR", config.HIGH_CONF_THR)

    # ── Helper: look up context vote for a patch ──────────────────
    def _ctx_vote(n: str) -> Dict:
        if section_mode == "window":
            return win_vote.get(n, {})
        sid    = image_info.get(n, {}).get("slide_id")
        sec_id = image_info.get(n, {}).get("section_id", -1)
        if sid is None or sec_id == -1:
            return {}
        return sec_vote.get(sid, {}).get(sec_id, {})

    # ── Rule 1a / Rule 1b ─────────────────────────────────────────
    #
    # Rule 1a — Mixed patch in a dominant context → resolve to context majority
    # Rule 1b — minority firm patch in dominant context (strict bar) → flip
    #
    # Rule 0 guard: conf >= anchor_conf AND firm → immutable, skip ALL rules.

    for n, pred in raw.items():
        # Rule 0: fully immutable anchor
        if pred["conf"] >= anchor_conf and pred["predicted"] not in UNCERTAIN_CLASSES:
            continue

        cls = pred["predicted"]
        ctx = _ctx_vote(n)

        if not ctx:
            continue
        if not ctx.get("dominant", False):
            continue

        ctx_maj = ctx["majority"]
        if ctx_maj == "TIE":
            continue

        # ── Rule 1a: Mixed patch → resolve ───────────────────────
        if cls in UNCERTAIN_CLASSES:
            corrections[n] = {
                "rule": "rule1a",
                "new_class": ctx_maj,
                "needs_review": False,
            }
            r1a_count["G" if ctx_maj == "G" else "Gplus"] += 1

        # ── Rule 1b: minority firm patch → flip ──────────────────
        elif cls != ctx_maj:
            if not ctx.get("dominant_strict", False):
                continue
            corrections[n] = {
                "rule": "rule1b",
                "new_class": ctx_maj,
                "needs_review": False,
            }
            r1b_count["G" if ctx_maj == "G" else "Gplus"] += 1

    print(f"  [Rule 1a] Mixed  resolved : {r1a_count['G']} -> G, "
          f"{r1a_count['Gplus']} -> Gplus")
    print(f"  [Rule 1b] minority flipped: {r1b_count['G']} -> G, "
          f"{r1b_count['Gplus']} -> Gplus")

    # ── Rule 2: neighbour agreement ───────────────────────────────
    #
    # Eligible: predicted Mixed OR conf_tier == LOW, AND not corrected by Rule 1.
    # Condition: >= NEIGHBOR_AGREE_MIN immediate 4-neighbours are all HIGH-confidence
    #            and all agree on the same class.
    # Action: correct to that agreed class (Tier 2).

    min_nb = config.NEIGHBOR_AGREE_MIN

    for n, pred in raw.items():
        if n in corrections:
            continue

        cls    = pred["predicted"]
        is_unc = cls in UNCERTAIN_CLASSES or pred["conf_tier"] == "LOW"
        if not is_unc:
            continue

        nbs = nb_index.get(n, [])
        if not nbs:
            continue

        for target in ("Gplus", "G"):
            agree = [
                nb for nb in nbs
                if nb in raw
                and raw[nb]["predicted"] == target
                and raw[nb]["conf_tier"] == "HIGH"
            ]
            if len(agree) >= min_nb:
                corrections[n] = {
                    "rule": "rule2_neighbor",
                    "new_class": target,
                    "needs_review": False,
                }
                r2_count[target] += 1
                break

    print(f"  [Rule 2] neighbour : {r2_count['G']} -> G, "
          f"{r2_count['Gplus']} -> Gplus")

    # ── Rule 3: spatial cluster flag ─────────────────────────────
    #
    # Cluster candidates:
    #   (a) predicted Mixed / deferred       — uncertain class
    #   (b) firm minority of ANY confidence  — spatial disagreement
    #
    # Rule 3 is a flagging rule, not a correction rule.
    # Rule 0 only protects patches from correction by Rule 1/2.
    # Rule 3 can still flag a high-confidence firm minority patch if it is part
    # of a qualifying spatial disagreement cluster.
    #
    # Invariants:
    #   - Patches already handled by Rule 1/2 are excluded.
    #   - No prediction change.
    #   - Rule 3 does NOT overwrite a Rule 1/2 suggested label.
    #
    # Isolated LOW/Mixed patches not handled by Rule 1/2 or Rule 3 → Rule 4.

    for sid in slides:
        names = [n for n in slides[sid] if n in raw]
        maj = slide_vote.get(sid, {}).get("majority", "TIE")

        targets = []
        for n in names:
            pred_n = raw[n]

            # Rule 1/2 output takes precedence — do not reassign to Rule 3.
            if n in corrections and corrections[n]["new_class"] is not None:
                continue

            is_mixed = pred_n["predicted"] in UNCERTAIN_CLASSES
            is_minority = (
                not is_mixed
                and maj != "TIE"
                and pred_n["predicted"] != maj
            )

            if is_mixed or is_minority:
                targets.append(n)

        for cluster in _find_clusters(targets, nb_index, config.CLUSTER_MIN):
            for p in cluster:
                if p in corrections and corrections[p]["new_class"] is not None:
                    continue

                corrections[p] = {
                    "rule": "rule3_cluster_flag",
                    "new_class": None,
                    "needs_review": True,
                }
                r3_count += 1

    print(
        f"  [Rule 3] clusters  : {r3_count} "
        f"(Mixed + firm minority of any confidence)"
    )

    # ── Rule 4: isolated uncertain patch ─────────────────────────
    #
    # Fires ONLY on LOW-conf or Mixed patches that have NO prior rule entry.
    # Priority chain: Rule 1/2 correction > Rule 3 cluster flag > Rule 4.
    #
    # If any prior rule has already acted on a patch (changed it OR flagged it),
    # Rule 4 is suppressed entirely.

    for n, pred in raw.items():
        is_low_conf = pred["conf_tier"] == "LOW"
        is_mixed = pred["predicted"] in UNCERTAIN_CLASSES

        if not (is_low_conf or is_mixed):
            continue  # not uncertain — Rule 4 does not apply

        # ANY prior rule entry takes precedence — correction OR cluster flag.
        if n in corrections:
            continue

        # Isolated uncertain patch — no spatial rule acted on it.
        r4_count += 1
        corrections[n] = {
            "rule": "rule4_refer",
            "new_class": None,
            "needs_review": True,
        }

    print(f"  [Rule 4] refer     : {r4_count} (isolated LOW/Mixed, no cluster)")
    print(
        f"  [Rules] Changed: "
        f"{sum(1 for c in corrections.values() if c['new_class'] is not None)}  "
        f"Cluster flagged: {r3_count}  "
        f"Isolated refer: {r4_count}"
    )

    return corrections


# ─────────────────────────────────────────────────────────────────
#  Apply corrections
# ─────────────────────────────────────────────────────────────────

def apply_corrections(raw: Dict, corrections: Dict) -> Dict:
    """
    Merge raw CNN predictions with corrections to produce final predictions.

    Output per patch:
        final_predicted      : str   — corrected class (or original if no change)
        rule_applied         : str|None — which rule fired
        changed              : bool  — True if final_predicted != original predicted
        needs_review         : bool  — True if Rule 3 or Rule 4 flagged for review
        correction_confidence: str|None — "strong"/"moderate"/None
                               Set when Rule 1/2 corrected a patch that was LOW/Mixed.
                               Indicates the spatial rule had enough evidence to handle
                               what would otherwise have been an unresolved uncertain patch.
    """
    final: Dict = {}
    for iname, pred in raw.items():
        fp   = dict(pred)
        corr = corrections.get(iname)

        if corr and corr["new_class"] is not None:
            fp["final_predicted"] = corr["new_class"]
            fp["rule_applied"]    = corr["rule"]
            fp["changed"]         = fp["final_predicted"] != pred["predicted"]
        else:
            fp["final_predicted"] = pred["predicted"]
            fp["rule_applied"]    = corr["rule"] if corr else None
            fp["changed"]         = False

        fp["needs_review"] = bool(
            corr is not None and corr.get("needs_review", False)
        )

        # ── Correction confidence tag ─────────────────────────────
        # If a spatial rule (1a/1b/2) corrected a patch that was LOW or Mixed,
        # tag the correction so eval can separately assess "how often does the
        # spatial rule save a would-be Tier 3 patch and get it right?"
        was_uncertain = (
            pred["predicted"] in UNCERTAIN_CLASSES or
            pred["conf_tier"] == "LOW"
        )
        was_corrected = fp["changed"]
        if was_corrected and was_uncertain:
            rule = corr["rule"] if corr else ""
            if rule in ("rule1a", "rule1b"):
                fp["correction_confidence"] = "strong"   # dominant context (high bar)
            elif rule == "rule2_neighbor":
                fp["correction_confidence"] = "moderate"  # neighbour agreement
            else:
                fp["correction_confidence"] = None
        else:
            fp["correction_confidence"] = None

        final[iname] = fp

    return final
