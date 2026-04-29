"""
src/pyreason_engine.py
======================
Pure-Python implementation of the PyReason slide+section rules.

Rule naming (canonical):
  Rule 1a  — Mixed patch in dominant SECTION/WINDOW    -> section/window majority
  Rule 1b  — minority firm patch in dominant SECTION/WINDOW -> section/window majority
             (requires dominant_strict — higher evidence bar than Rule 1a)
  Rule 2   — uncertain/LOW patch + >=N HIGH neighbours agree -> neighbour class
             Fires only when Rule 1a/1b did NOT already correct the patch.
  Rule 3   — cluster of 3+ adjacent uncertain/minority patches -> flag mixed zone
  Rule 4   — conf < MEDIUM_CONF_THR or predicted Mixed  -> flag for human review
             Stacks on top of any Rule 1/2 correction (flag + change are independent).

Context mode (config.SECTION_MODE):
  "quadrant"  (default) — slide split into N_SECTIONS fixed quadrants (2×2 or 3×3).
              Each patch's context = every other patch in the same quadrant.
  "window"    — each patch's context = patches in a sliding W_ROWS × W_COLS window
              centred on that patch. No hard region boundaries.

Rule execution order per patch:
  Rule 0 (anchor guard) → Rule 1a/1b → Rule 2 → Rule 3 → Rule 4

Anchor guard (applied before all rules):
  Rule 0: conf >= ANCHOR_CONF_THR (= HIGH_CONF_THR) and not Mixed → immutable, skip all rules.
"""

from __future__ import annotations

from typing import Dict, List, Set, Tuple

import config

# With a 2-class CNN (G / Gplus only), the model never predicts "Mixed".
# UNCERTAIN_CLASSES is kept for backward compat with 3-class checkpoints
# and for Rule 2/3/4 which treat Mixed as uncertain.
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
#  Slide vote  (used only for diagnostics + Rule 3 reference)
# ─────────────────────────────────────────────────────────────────

def compute_slide_vote(slides: Dict, raw: Dict) -> Dict:
    """
    Per-slide majority vote. Mixed patches abstain.
    Result is used for:
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
#  Two modes controlled by config.SECTION_MODE:
#    "quadrant" → fixed N_SECTIONS quadrants per slide
#    "window"   → sliding W_ROWS × W_COLS window per patch
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

    Section layout for N_SECTIONS=4 (sq=2):
        | 0 | 1 |
        | 2 | 3 |
    Split point determined by the span of actual patch coordinates in that slide.

    Why sections over whole-slide vote:
      - Whole-slide vote is diluted when a slide has distinct spatial regions.
      - Neighbourhoods of 2-4 patches fire too easily on sparse grids.
      - A quadrant (typically 3-12 patches) gives a reliable local majority
        that aligns with the spatial biology without being too narrow.
    """
    n_sections     = getattr(config, "N_SECTIONS", 4)
    sq             = max(1, int(n_sections ** 0.5))   # 4→2, 9→3
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

    Advantages over fixed quadrants:
      - No hard boundary artefact: two adjacent patches always share most of their
        context, so the vote changes smoothly across the slide.
      - Patches near region boundaries get context that reflects their immediate
        neighbourhood rather than an arbitrary quadrant assignment.

    Window dimensions from config (defaults to 4 rows × 3 cols = 12-patch window):
        WINDOW_ROWS = getattr(config, "WINDOW_ROWS", 4)
        WINDOW_COLS = getattr(config, "WINDOW_COLS", 3)

    For boundary patches (edge/corner of slide), the window simply clips to
    available patches — no padding or mirror fill.
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

    # Build grid lookup: (slide_id, row, col) -> patch_name
    grid: Dict[Tuple, str] = {}
    for n, info in image_info.items():
        if info.get("row") is not None and info.get("col") is not None:
            grid[(info["slide_id"], info["row"], info["col"])] = n

    hr = W_ROWS // 2   # half-extent rows  (e.g. 4→2)
    hc = W_COLS // 2   # half-extent cols  (e.g. 3→1)

    win_votes: Dict[str, Dict] = {}

    for n, info in image_info.items():
        r, c, sid = info.get("row"), info.get("col"), info.get("slide_id")
        if r is None or c is None or sid is None:
            # Non-grid patch: no context available
            win_votes[n] = dict(
                majority="TIE", ratio=0.0, n_firm=0, mixed_fraction=0.0,
                dominant=False, dominant_strict=False,
            )
            continue

        # Collect all patches in the W_ROWS × W_COLS window
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

    new_class=None  → flag only (review flag set, prediction unchanged).
    new_class=str   → prediction changed; needs_review may also be True (Rule 4 stack).

    Rule execution order per patch:
        Rule 0 (anchor guard)
        Rule 1a (Mixed in dominant context)    → correct to context majority
        Rule 1b (minority firm in dominant context, strict bar) → flip to context majority
        — if Rule 1a or 1b fired, patch is already in corrections dict;
          Rule 2 skips it via `if n in corrections: continue`
        Rule 2  (uncertain/LOW + HIGH neighbour agreement) → correct to agreed class
        Rule 3  (cluster flag)         → needs_review only
        Rule 4  (low-conf / Mixed flag) → needs_review only, stacks on any prior rule

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
        # Stamp a dummy section_id so downstream code stays consistent
        for n, info in image_info.items():
            info.setdefault("section_id", 0)
        print(f"  [Context] Window mode  ({getattr(config, 'WINDOW_ROWS', 4)}"
              f"×{getattr(config, 'WINDOW_COLS', 3)} patches per patch)")
    else:
        sec_vote  = compute_section_vote(slides, raw, image_info)
        win_vote  = None   # not used in quadrant mode
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
    r1a_count = {"G": 0, "Gplus": 0}   # Rule 1a: Mixed → context majority
    r1b_count = {"G": 0, "Gplus": 0}   # Rule 1b: minority firm → context majority
    r2_count  = {"G": 0, "Gplus": 0}
    r3_count  = 0
    r4_count  = 0

    anchor_conf = getattr(config, "ANCHOR_CONF_THR", config.HIGH_CONF_THR)

    # ── Helper: look up context vote for a patch ──────────────────
    def _ctx_vote(n: str) -> Dict:
        """Return the context-vote dict for patch n (window or quadrant mode)."""
        if section_mode == "window":
            return win_vote.get(n, {})
        # quadrant mode
        sid    = image_info.get(n, {}).get("slide_id")
        sec_id = image_info.get(n, {}).get("section_id", -1)
        if sid is None or sec_id == -1:
            return {}
        return sec_vote.get(sid, {}).get(sec_id, {})

    # ── Rule 1a / Rule 1b  ────────────────────────────────────────
    #
    # Rule 1a — Mixed patch in a dominant context window/quadrant
    #   → resolve to context majority  (needs dominant, looser bar)
    #
    # Rule 1b — minority firm patch in a dominant context window/quadrant
    #   → flip to context majority     (needs dominant_strict, stricter bar)
    #
    # Guard applied BEFORE checking context:
    #   Rule 0: conf >= anchor_conf and not Mixed → immutable, skip ALL rules.
    #
    # If Rule 1a or 1b fires, the patch is added to corrections; Rule 2
    # will skip it via `if n in corrections: continue`.
    # If neither fires (section not dominant, patch is majority class, etc.),
    # the patch falls through to Rule 2 naturally.

    for n, pred in raw.items():
        # Rule 0: fully immutable anchor
        if pred["conf"] >= anchor_conf and pred["predicted"] not in UNCERTAIN_CLASSES:
            continue

        cls = pred["predicted"]
        ctx = _ctx_vote(n)

        if not ctx:
            continue   # no grid position or context unavailable

        if not ctx.get("dominant", False):
            continue   # context not reliable enough for Rule 1a/1b

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
                continue   # context evidence not strong enough to flip a firm pred
            corrections[n] = {
                "rule": "rule1b",
                "new_class": ctx_maj,
                "needs_review": False,
            }
            r1b_count["G" if ctx_maj == "G" else "Gplus"] += 1

        # cls == ctx_maj: patch already agrees with context → no action

    print(f"  [Rule 1a] Mixed  resolved : {r1a_count['G']} -> G, "
          f"{r1a_count['Gplus']} -> Gplus")
    print(f"  [Rule 1b] minority flipped: {r1b_count['G']} -> G, "
          f"{r1b_count['Gplus']} -> Gplus")

    # ── Rule 2: neighbour agreement ───────────────────────────────
    #
    # Eligible patches: predicted Mixed OR conf_tier == LOW,
    #                   AND not already corrected by Rule 1a/1b.
    #
    # Condition: >= NEIGHBOR_AGREE_MIN immediate 4-neighbours are all
    #            HIGH-confidence and all agree on the same class.
    # Action: correct patch to that agreed class.
    #
    # Rule 2 fires on patches that Rule 1 could not help:
    #   - patch in a non-dominant section/window (section too sparse)
    #   - patch is the only patch in a slide (standalone)
    # In all these cases, if strong local neighbour consensus exists,
    # Rule 2 can still resolve the patch.

    min_nb = config.NEIGHBOR_AGREE_MIN

    for n, pred in raw.items():
        if n in corrections:
            continue   # Rule 1a or 1b already handled this patch

        cls    = pred["predicted"]
        is_unc = cls in UNCERTAIN_CLASSES or pred["conf_tier"] == "LOW"
        if not is_unc:
            continue   # only uncertain/LOW patches are eligible for Rule 2

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

    # ── Rule 3: cluster flag ──────────────────────────────────────
    #
    # For each slide, find patches that are either:
    #   - predicted Mixed, or
    #   - firm but predicting the minority class (slide majority used as reference)
    # If 3+ such patches form a connected component → flag all of them for review.
    # No prediction change; needs_review is set to True.
    # Rule 3 adds to corrections but does not overwrite a Rule 1/2 new_class.

    for sid in slides:
        names = [n for n in slides[sid] if n in raw]
        maj   = slide_vote.get(sid, {}).get("majority", "TIE")
        targets = [
            n for n in names
            if raw[n]["predicted"] in UNCERTAIN_CLASSES
            or (maj != "TIE" and raw[n]["predicted"] != maj)
        ]
        for cluster in _find_clusters(targets, nb_index, config.CLUSTER_MIN):
            for p in cluster:
                if p not in corrections:
                    corrections[p] = {
                        "rule": "rule3_cluster_flag",
                        "new_class": None,
                        "needs_review": True,
                    }
                else:
                    corrections[p]["needs_review"] = True
                r3_count += 1

    print(f"  [Rule 3] clusters  : {r3_count}")

    # ── Rule 4: low-confidence / Mixed flag ───────────────────────
    #
    # Every patch with conf < MEDIUM_CONF_THR (= REVIEW_CONF_THR) OR
    # predicted Mixed gets flagged for human review.
    # This is purely additive:
    #   - If the patch already has a Rule 1/2 correction → flag stacks on top
    #     (changed=True AND needs_review=True in apply_corrections output).
    #   - If the patch has no prior correction → entry is created with
    #     new_class=None (flag only, no prediction change).

    review_thr = config.MEDIUM_CONF_THR

    for n, pred in raw.items():
        is_low_conf = pred["conf"] < review_thr
        is_mixed    = pred["predicted"] in UNCERTAIN_CLASSES
        if is_low_conf or is_mixed:
            r4_count += 1
            if n in corrections:
                corrections[n]["needs_review"] = True
            else:
                corrections[n] = {
                    "rule": "rule4_review",
                    "new_class": None,
                    "needs_review": True,
                }

    print(f"  [Rule 4] flagged   : {r4_count}")
    print(
        f"  [Rules] Changed: "
        f"{sum(1 for c in corrections.values() if c['new_class'] is not None)}  "
        f"Flagged: "
        f"{sum(1 for c in corrections.values() if c['new_class'] is None and c.get('needs_review'))}"
    )

    return corrections


# ─────────────────────────────────────────────────────────────────
#  Apply corrections
# ─────────────────────────────────────────────────────────────────

def apply_corrections(raw: Dict, corrections: Dict) -> Dict:
    """
    Merge raw CNN predictions with corrections to produce final predictions.

    Output per patch:
        final_predicted : str   — corrected class (or original if no change)
        rule_applied    : str|None — which rule fired last for prediction change
        changed         : bool  — True if final_predicted != original predicted
        needs_review    : bool  — True if any rule flagged this patch for review
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

        fp["needs_review"] = (
            corr is not None
            and (corr["new_class"] is None or corr.get("needs_review", False))
        )
        final[iname] = fp

    return final
