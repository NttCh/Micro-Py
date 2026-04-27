"""
visualize_slides.py
===================
Light-theme patch grid visualiser — consistent with pyreason_engine.py v2.

Context parameters (SECTION_MODE, WINDOW_ROWS, WINDOW_COLS, N_SECTIONS, etc.)
are loaded from best_threshold_evaluation/config_used.txt when available,
so the window borders always match the evaluation that produced the CSV —
NOT whatever is currently in config.py.

Fallback priority:
  1. config_used.txt next to patch_results.csv  (most correct)
  2. config.py values                            (manual config)

Output folder structure
───────────────────────
  slide_grids/
    all/          one dual-panel PNG per slide (ML | After)
    changed/      dual-panel PNGs for slides with any change
    by_rule/
      rule1/      slides changed by rule1a or rule1b — THREE panels saved separately:
                    <sid>_1_ml_only.png        before (no window overlay)
                    <sid>_2_ml_window.png      before + window border on changed patch
                    <sid>_3_after.png          after PyReason
      rule2/      slides changed by rule2 only — two panels (ML | After)
      rule3/      slides with cluster flags    — two panels
      rule4/      slides with low-conf flags   — two panels
    legend.png    shared legend, large font, standalone
    summary_5_samples.png
"""
from __future__ import annotations
import os, sys
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import config
import matplotlib.pyplot as plt

UNCERTAIN = {"Mixed", "Mix", "mix", "mixed"}
# ── Define search priority once here — edit this list to change order ──
EVAL_SUBDIR_PRIORITY = [
    "best_threshold_evaluation_full_try",
    "best_threshold_evaluation",
    "grid_only",
]

BG_COLOR = "#FFFFFF"
AX_COLOR = "#F5F5F5"

PRED_COLOR = {
    "G":     "#3E6B89",
    "Gplus": "#5C8B7E",
    "Mixed": "#efce95",
}
WRONG_COLOR = {
    "G":     "#8EA6CA",
    "Gplus": "#71936F",
    "Mixed": "#efce95",
}

CELL_TEXT_COLOR   = "#F0F0F0"
CELL_TEXT_CHANGED = "#EBEBEB"
WRONG_TEXT_COLOR  = "#EBEBEB"

RULE_COLOR = {
    "rule1a":              "#FFD269",
    "rule1b":              "#B03030",
    "rule2_neighbor":      "#1A7A8A",
    "rule3_cluster_flag":  "#666666",
    "rule4_review":        "#444444",
    "rule1f_mixed":        "#C48A00",
    "rule1a_mixed":        "#C48A00",
    "rule1f_minority":     "#B03030",
    "rule1b_minority":     "#B03030",
    "rule2_cluster_flag":  "#666666",
    "rule3_review":        "#444444",
}

SECTION_PALETTE  = ["#F4A0A0", "#70C8C0", "#70AED4", "#A0CEB0"]
ANCHOR_HATCH     = "////"
STRONGHOLD_HATCH = "xxxx"
REVIEW_DOT_COLOR = "#EDEDED"
WRONG_MARK_COLOR = "#C0392B"

GRID_ROWS = 7
GRID_COLS = 9

_RULE_ALIAS = {
    "rule1f_mixed":    "rule1a",
    "rule1a_mixed":    "rule1a",
    "rule1f_minority": "rule1b",
    "rule1b_minority": "rule1b",
}

def _norm_rule(rule: str) -> str:
    return _RULE_ALIAS.get(str(rule), str(rule))

RULE1_NAMES = {"rule1a", "rule1b",
               "rule1f_mixed", "rule1a_mixed",
               "rule1f_minority", "rule1b_minority"}

def _conf_tier_label(conf: float, high_thr: float, low_thr: float) -> str:
    if conf >= high_thr: return "H"
    elif conf < low_thr: return "L"
    return "M"

TITLE_FONT  = {"fontfamily": "Times New Roman", "fontweight": "normal"}
DETAIL_FONT = {"fontfamily": "Arial",           "fontweight": "normal"}


# ─────────────────────────────────────────────────────────────────
#  DrawConfig — carries ALL context parameters explicitly.
#  Populated once from config_used.txt (or config.py fallback)
#  and passed through every draw call so nothing reads config
#  at draw time.
# ─────────────────────────────────────────────────────────────────

class DrawConfig:
    """
    All threshold / context parameters needed for drawing.
    Constructed once per run from config_used.txt or config.py.
    Passed explicitly to every drawing function — NO global config reads
    happen inside _draw_panel or any helper called by it.
    """
    def __init__(
        self,
        section_mode:    str   = "quadrant",
        n_sections:      int   = 4,
        window_rows:     int   = 4,
        window_cols:     int   = 3,
        high_conf_thr:   float = 0.85,
        medium_conf_thr: float = 0.75,
        anchor_conf_thr: float = 0.85,
        section_min_patches:         int   = 2,
        section_majority_ratio_thr:  float = 0.70,
        section_max_mixed_fraction:  float = 0.30,
        stronghold_min:              int   = 2,
        source:          str   = "config.py",
    ):
        self.section_mode   = section_mode.lower()
        self.n_sections     = n_sections
        self.window_rows    = window_rows
        self.window_cols    = window_cols
        self.high_conf_thr  = high_conf_thr
        self.medium_conf_thr = medium_conf_thr
        self.anchor_conf_thr = anchor_conf_thr
        self.section_min_patches        = section_min_patches
        self.section_majority_ratio_thr = section_majority_ratio_thr
        self.section_max_mixed_fraction = section_max_mixed_fraction
        self.stronghold_min = stronghold_min
        self.source         = source   # for display / logging

    @property
    def sq(self) -> int:
        return max(1, int(self.n_sections ** 0.5))

    def summary(self) -> str:
        if self.section_mode == "window":
            ctx = f"window {self.window_rows}×{self.window_cols}"
        else:
            ctx = f"quadrant N={self.n_sections} ({self.sq}×{self.sq})"
        return (
            f"DrawConfig [{self.source}]  mode={self.section_mode}  {ctx}  "
            f"HIGH={self.high_conf_thr}  anchor={self.anchor_conf_thr}  "
            f"stronghold_min={self.stronghold_min}"
        )


# ─────────────────────────────────────────────────────────────────
#  Load DrawConfig — config_used.txt → config.py fallback
# ─────────────────────────────────────────────────────────────────

def _parse_config_used_txt(path: str) -> dict:
    """
    Parse key = value lines from config_used.txt.
    Returns a dict of {PARAM_NAME: value_string}.
    Lines that are section headers, dashes, or blank are ignored.
    """
    result = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("=") or line.startswith("─") or line.startswith("-"):
                continue
            if "=" not in line:
                continue
            # Strip leading spaces / bullet chars
            line = line.lstrip(" \t•")
            key, _, val = line.partition("=")
            key = key.strip()
            val = val.split("(")[0].strip()   # drop trailing "(ignored — ...)" comments
            if key:
                result[key] = val
    return result


def load_draw_config(csv_path: str) -> DrawConfig:
    """
    Try to load DrawConfig from config_used.txt sitting next to csv_path
    (or in the parent directory).  Falls back to config.py values.

    config_used.txt is written by sweep_thresholds.py into
    best_threshold_evaluation/ so the context params are always in sync
    with the evaluation that produced the CSV.
    """
    search_dirs = [
        os.path.dirname(csv_path),
        os.path.dirname(os.path.dirname(csv_path)),
    ]
    txt_path = None
    for d in search_dirs:
        candidate = os.path.join(d, "config_used.txt")
        if os.path.exists(candidate):
            txt_path = candidate
            break

    # ── Fallback: read straight from config.py ────────────────────
    def _from_config() -> DrawConfig:
        return DrawConfig(
            section_mode    = str(getattr(config, "SECTION_MODE",   "quadrant")),
            n_sections      = int(getattr(config, "N_SECTIONS",     4)),
            window_rows     = int(getattr(config, "WINDOW_ROWS",    4)),
            window_cols     = int(getattr(config, "WINDOW_COLS",    3)),
            high_conf_thr   = float(getattr(config, "HIGH_CONF_THR",   0.85)),
            medium_conf_thr = float(getattr(config, "MEDIUM_CONF_THR", 0.75)),
            anchor_conf_thr = float(getattr(config, "ANCHOR_CONF_THR", 0.85)),
            section_min_patches        = int(getattr(config, "SECTION_MIN_PATCHES", 2)),
            section_majority_ratio_thr = float(getattr(config, "SECTION_MAJORITY_RATIO_THR",
                                                        getattr(config, "MAJORITY_RATIO_THR", 0.70))),
            section_max_mixed_fraction = float(getattr(config, "SECTION_MAX_MIXED_FRACTION", 0.30)),
            stronghold_min  = int(getattr(config, "STRONGHOLD_MIN", 2)),
            source          = "config.py",
        )

    if txt_path is None:
        dc = _from_config()
        print(f"  [DrawConfig] config_used.txt not found — using config.py values")
        print(f"  {dc.summary()}")
        return dc

    try:
        kv = _parse_config_used_txt(txt_path)

        def _get(key, default, cast):
            v = kv.get(key)
            if v is None:
                return default
            try:
                return cast(v)
            except Exception:
                return default

        dc = DrawConfig(
            section_mode    = _get("SECTION_MODE",    "quadrant", str),
            n_sections      = _get("N_SECTIONS",      4,          int),
            window_rows     = _get("WINDOW_ROWS x WINDOW_COLS", None, str),  # handled below
            window_cols     = 3,   # placeholder
            high_conf_thr   = _get("HIGH_CONF_THR",   0.85,  float),
            medium_conf_thr = float(getattr(config, "MEDIUM_CONF_THR", 0.75)),
            anchor_conf_thr = _get("ANCHOR_CONF_THR", 0.85,  float),
            section_min_patches        = _get("SECTION_MIN_PATCHES",        2,    int),
            section_majority_ratio_thr = _get("SECTION_MAJORITY_RATIO_THR", 0.70, float),
            section_max_mixed_fraction = float(getattr(config, "SECTION_MAX_MIXED_FRACTION", 0.30)),
            stronghold_min  = _get("STRONGHOLD_MIN",  2, int),
            source          = os.path.relpath(txt_path),
        )

        # WINDOW_ROWS x WINDOW_COLS is stored as "4 x 3" in the txt
        win_str = kv.get("WINDOW_ROWS x WINDOW_COLS", "")
        if "x" in win_str:
            parts = win_str.split("x")
            try:
                dc.window_rows = int(parts[0].strip())
                dc.window_cols = int(parts[1].strip())
            except Exception:
                dc.window_rows = int(getattr(config, "WINDOW_ROWS", 4))
                dc.window_cols = int(getattr(config, "WINDOW_COLS", 3))
        else:
            dc.window_rows = int(getattr(config, "WINDOW_ROWS", 4))
            dc.window_cols = int(getattr(config, "WINDOW_COLS", 3))

        print(f"  [DrawConfig] Loaded from {txt_path}")
        print(f"  {dc.summary()}")
        return dc

    except Exception as e:
        print(f"  [DrawConfig] Failed to parse {txt_path}: {e} — falling back to config.py")
        dc = _from_config()
        print(f"  {dc.summary()}")
        return dc


# ─────────────────────────────────────────────────────────────────
#  Engine mirrors  (all accept DrawConfig instead of reading config)
# ─────────────────────────────────────────────────────────────────

def assign_sections(df_slide, dc: DrawConfig):
    sq = dc.sq
    df = df_slide.copy()
    valid = df[df["row"].notna() & df["col"].notna()]
    if len(valid) == 0:
        df["section_id"] = -1
        return df, None
    rows = sorted(valid["row"].tolist())
    cols = sorted(valid["col"].tolist())
    rmin, rmax = rows[0], rows[-1]
    cmin, cmax = cols[0], cols[-1]
    row_span = max(rmax - rmin, 1)
    col_span = max(cmax - cmin, 1)

    def _sec(r, c):
        rb = min(int((r - rmin) / row_span * sq), sq - 1)
        cb = min(int((c - cmin) / col_span * sq), sq - 1)
        return rb * sq + cb

    df["section_id"] = df.apply(
        lambda row: _sec(row["row"], row["col"]) if pd.notna(row["row"]) else -1,
        axis=1)
    bounds = dict(rmin=rmin, rmax=rmax, cmin=cmin, cmax=cmax,
                  row_span=row_span, col_span=col_span, sq=sq)
    return df, bounds


def compute_section_votes(df_slide, dc: DrawConfig):
    result = {}
    for sec_id, grp in df_slide.groupby("section_id"):
        if sec_id == -1:
            continue
        firm   = grp[~grp["ml_predicted"].isin(UNCERTAIN)]
        n_firm = len(firm)
        n_gp   = (firm["ml_predicted"] == "Gplus").sum()
        n_g    = n_firm - n_gp
        n_mix  = len(grp) - n_firm

        if n_firm == 0:    maj, maj_n = "TIE", 0
        elif n_gp > n_g:   maj, maj_n = "Gplus", n_gp
        elif n_g > n_gp:   maj, maj_n = "G", n_g
        else:              maj, maj_n = "TIE", n_g

        ratio    = maj_n / n_firm if n_firm > 0 else 0.0
        mix_frac = n_mix / len(grp) if len(grp) > 0 else 0.0
        dominant = (
            n_firm >= dc.section_min_patches
            and maj != "TIE"
            and ratio >= dc.section_majority_ratio_thr
            and mix_frac <= dc.section_max_mixed_fraction
        )

        would_fire = []
        if dominant:
            for _, p in grp.iterrows():
                ml   = p["ml_predicted"]
                conf = p["ml_conf"]
                if ml not in UNCERTAIN and conf >= dc.anchor_conf_thr:
                    continue
                if ml in UNCERTAIN or ml != maj:
                    would_fire.append(p["image_name"])

        result[sec_id] = dict(
            majority=maj, ratio=ratio, n_firm=n_firm,
            n_mixed=n_mix, mix_frac=mix_frac,
            dominant=dominant, would_fire=would_fire,
        )
    return result


def compute_strongholds(df_slide, dc: DrawConfig):
    grid = {}
    for _, row in df_slide.iterrows():
        if pd.notna(row["row"]) and pd.notna(row["col"]):
            grid[(int(row["row"]), int(row["col"]))] = row
    strongholds = set()
    for _, row in df_slide.iterrows():
        if row["ml_predicted"] in UNCERTAIN or row["ml_conf_tier"] != "HIGH":
            continue
        r, c = int(row["row"]), int(row["col"])
        cls = row["ml_predicted"]
        same_nbs = sum(
            1 for dr, dc_ in [(0, 1), (0, -1), (1, 0), (-1, 0)]
            if (nb := grid.get((r + dr, c + dc_))) is not None
            and nb["ml_predicted"] == cls and nb["ml_conf_tier"] == "HIGH"
        )
        if same_nbs >= dc.stronghold_min:
            strongholds.add(row["image_name"])
    return strongholds


def _to_grid(df_slide):
    df = df_slide.copy()
    df["row"] = pd.to_numeric(df["row"], errors="coerce")
    df["col"] = pd.to_numeric(df["col"], errors="coerce")
    valid = df[df["row"].notna() & df["col"].notna()]
    if len(valid) == 0:
        df["grow"] = np.nan; df["gcol"] = np.nan
        return df, [], []
    df["grow"] = df["row"]
    df["gcol"] = df["col"]
    rows_sorted = sorted(valid["row"].unique())
    cols_sorted = sorted(valid["col"].unique())
    return df, rows_sorted, cols_sorted


# ─────────────────────────────────────────────────────────────────
#  Core panel drawing
#  mode:
#    "before"        — ML predictions, no rule borders, no window overlay
#    "before_window" — ML predictions + dashed window border on changed patches
#    "after"         — final predictions, rule borders, review dots
#
#  dc: DrawConfig — ALL context params come from here, never from config
# ─────────────────────────────────────────────────────────────────

def _draw_panel(ax, df_slide, title: str, dc: DrawConfig, mode: str = "after"):
    import matplotlib.patches as mpatches

    ax.set_facecolor(AX_COLOR)

    df_slide, bounds = assign_sections(df_slide, dc)
    if bounds is None:
        ax.text(0.5, 0.5, "No grid patches", ha="center", va="center",
                color=CELL_TEXT_COLOR, transform=ax.transAxes, fontsize=11,
                **DETAIL_FONT)
        ax.set_title(title.upper(), fontsize=11, color="#1A1A2E", **TITLE_FONT, pad=5)
        return

    df_slide, rows_sorted, cols_sorted = _to_grid(df_slide)

    sec_votes   = compute_section_votes(df_slide, dc) if dc.section_mode != "window" else {}
    strongholds = compute_strongholds(df_slide, dc)

    sq         = dc.sq
    n_sections = dc.n_sections

    # ── Section background (quadrant mode only) ───────────────────
    if dc.section_mode != "window":
        changed_sections: set = set()
        if mode == "after":
            for _, row in df_slide.iterrows():
                if bool(row.get("changed", False)):
                    sec = row.get("section_id", -1)
                    if sec != -1:
                        changed_sections.add(int(sec))

        for sec_id in range(n_sections):
            rb, cb = sec_id // sq, sec_id % sq
            c0_d = cb * GRID_COLS / sq
            c1_d = (cb + 1) * GRID_COLS / sq
            r0_d = rb * GRID_ROWS / sq
            r1_d = (rb + 1) * GRID_ROWS / sq
            sv    = sec_votes.get(sec_id, {})
            color = SECTION_PALETTE[sec_id % len(SECTION_PALETTE)]

            if mode == "after" and sec_id in changed_sections:
                alpha_fill = 0.28; alpha_edge = 0.85; lw = 1.6
            elif sv.get("dominant"):
                alpha_fill = 0.12; alpha_edge = 0.45; lw = 0.8
            else:
                alpha_fill = 0.05; alpha_edge = 0.20; lw = 0.4

            ax.add_patch(mpatches.FancyBboxPatch(
                (c0_d - 0.5, r0_d - 0.5), c1_d - c0_d, r1_d - r0_d,
                linewidth=lw, edgecolor=color, facecolor=color,
                alpha=alpha_fill, boxstyle="round,pad=0", zorder=1))

    # ── Would-fire set (before panel only) ───────────────────────
    would_fire_set: set = set()
    if mode in ("before", "before_window"):
        for sv in sec_votes.values():
            would_fire_set.update(sv.get("would_fire", []))

    # ── Window half-extents (from DrawConfig, not config) ────────
    W_ROWS = dc.window_rows
    W_COLS = dc.window_cols
    h_r = (W_ROWS - 1) / 2
    h_c = (W_COLS - 1) / 2

    # ── Draw each patch ───────────────────────────────────────────
    for _, row in df_slide.iterrows():
        r, c = row.get("grow"), row.get("gcol")
        if pd.isna(r) or pd.isna(c):
            continue
        r, c = int(r), int(c)

        ml_pred    = str(row.get("ml_predicted",    ""))
        final_pred = str(row.get("final_predicted", ""))
        gt         = str(row.get("gt_label",        ""))
        rule_raw   = str(row.get("rule_applied",    ""))
        rule       = _norm_rule(rule_raw)
        changed    = bool(row.get("changed",        False))
        needs_rev  = bool(row.get("needs_review",   False))
        iname      = str(row.get("image_name",      ""))
        ml_conf    = float(row.get("ml_conf",       0))

        is_before    = mode in ("before", "before_window")
        display_pred = ml_pred if is_before else final_pred

        has_gt   = gt and gt not in ("None", "nan", "")
        is_wrong = has_gt and (
            (ml_pred != gt) if is_before else (final_pred != gt)
        )

        fc = (WRONG_COLOR.get(display_pred, "#CCCCCC") if is_wrong
              else PRED_COLOR.get(display_pred, "#AAAAAA"))

        if is_wrong:
            txt_color = WRONG_TEXT_COLOR
        elif mode == "after" and changed:
            txt_color = CELL_TEXT_CHANGED
        else:
            txt_color = CELL_TEXT_COLOR

        if display_pred in UNCERTAIN:
            txt_color = "#1A1A2E"

        is_anchor     = (ml_pred not in UNCERTAIN and ml_conf >= dc.anchor_conf_thr)
        is_stronghold = iname in strongholds
        would_fire    = iname in would_fire_set

        # Border style
        if mode == "after" and changed:
            lw = 2.8; ls = "solid"; ec = RULE_COLOR.get(rule, "#555555")
        elif is_before and would_fire:
            lw = 1.6; ls = "dashed"; ec = "#7B5EA7"
        else:
            lw = 0.7; ls = "solid"; ec = "#BBBBBB"

        # ── Window overlay for before_window mode ─────────────────
        # Uses dc.window_rows / dc.window_cols — same values that
        # pyreason_engine.py used when it ran the evaluation.
        # fixed — only Rule 1a / 1b get the window border
        RULE1_CHANGED = {"rule1a", "rule1b", "rule1f_mixed", "rule1a_mixed",
                        "rule1f_minority", "rule1b_minority"}
        if mode == "before_window" and changed and dc.section_mode == "window" and rule in RULE1_CHANGED:
            ax.add_patch(mpatches.Rectangle(
                (c - h_c - 0.5, r - h_r - 0.5), W_COLS, W_ROWS,
                linewidth=1.8, edgecolor="#84190D", facecolor="none",
                linestyle=(0, (5, 4)), alpha=0.75, zorder=2))

        # ── After panel window border ─────────────────────────────


        if mode == "after" and changed and dc.section_mode == "window" and rule in RULE1_CHANGED:
            ax.add_patch(mpatches.Rectangle(
                (c - h_c - 0.5, r - h_r - 0.5), W_COLS, W_ROWS,
                linewidth=1.8, edgecolor="#84190D", facecolor="none",
                linestyle=(0, (5, 4)), alpha=0.75, zorder=2))

        # ── Patch cell ────────────────────────────────────────────
        ax.add_patch(mpatches.FancyBboxPatch(
            (c - 0.45, r - 0.45), 0.90, 0.90,
            linewidth=lw, linestyle=ls, edgecolor=ec, facecolor=fc,
            alpha=0.93, boxstyle="round,pad=0.03", zorder=3))

        # Hatch
        if is_anchor and not (mode == "after" and changed):
            ax.add_patch(mpatches.FancyBboxPatch(
                (c - 0.45, r - 0.45), 0.90, 0.90,
                linewidth=0, hatch=ANCHOR_HATCH,
                edgecolor="#333333", facecolor="none",
                alpha=0.20, boxstyle="round,pad=0.03", zorder=4))

        # Review dot (after mode, flagged but not changed)
        if mode == "after" and needs_rev:
            ax.plot(c + 0.36, r - 0.36, "o",
                    color=REVIEW_DOT_COLOR, markersize=4, zorder=6)

        # ── Labels ───────────────────────────────────────────────
        def _short(pred):
            if pred == "Gplus":   return "G+"
            if pred in UNCERTAIN: return "Mx"
            return pred

        tier = _conf_tier_label(ml_conf, dc.high_conf_thr, dc.medium_conf_thr)

        if mode == "after" and changed:
            ax.text(c, r - 0.18, f"{_short(ml_pred)}→{_short(final_pred)}",
                    ha="center", va="center", fontsize=11,
                    color=txt_color, **DETAIL_FONT, zorder=7)
            ax.text(c, r + 0.22, f"{tier} {ml_conf:.2f}",
                    ha="center", va="center", fontsize=9,
                    color=txt_color, alpha=0.85, **DETAIL_FONT, zorder=7)
        else:
            ax.text(c, r - 0.11, _short(display_pred),
                    ha="center", va="center", fontsize=11,
                    color=txt_color, **DETAIL_FONT, zorder=7)
            ax.text(c, r + 0.27, f"{tier} {ml_conf:.2f}",
                    ha="center", va="center", fontsize=9,
                    color=txt_color, alpha=0.88, **DETAIL_FONT, zorder=7)

        if is_wrong:
            ax.text(c - 0.37, r - 0.37, "X",
                    ha="center", va="center", fontsize=9,
                    color=WRONG_MARK_COLOR, **DETAIL_FONT, zorder=8)

    # ── Axes ─────────────────────────────────────────────────────
    ax.set_xlim(-0.5, GRID_COLS - 0.5)
    ax.set_ylim(GRID_ROWS - 0.5, -0.5)
    ax.set_xticks([]); ax.set_yticks([])

    for x in np.arange(-0.5, GRID_COLS, 1.0):
        ax.axvline(x, color="#555555", linewidth=0.6, alpha=0.20, zorder=0)
    for y in np.arange(-0.5, GRID_ROWS, 1.0):
        ax.axhline(y, color="#555555", linewidth=0.6, alpha=0.20, zorder=0)

    # Bold section boundary lines (quadrant mode only)
    if dc.section_mode != "window":
        for qi in range(1, sq):
            ax.axvline(qi * GRID_COLS / sq - 0.5,
                       color="#222222", linewidth=2.2, alpha=0.60, zorder=5)
        for qi in range(1, sq):
            ax.axhline(qi * GRID_ROWS / sq - 0.5,
                       color="#222222", linewidth=2.2, alpha=0.60, zorder=5)

    for spine in ax.spines.values():
        spine.set_edgecolor("#AAAAAA")
        spine.set_linewidth(0.8)

    ax.set_title(title.upper(), fontsize=16, color="#1A1A2E", pad=5, **TITLE_FONT)


# ─────────────────────────────────────────────────────────────────
#  Dual-panel (ML | After)
# ─────────────────────────────────────────────────────────────────

def draw_slide_dual(ax_before, ax_after, df_slide, slide_id, stats: dict, dc: DrawConfig):
    n_ch  = stats.get("n_changed",  0)
    n_r1a = stats.get("n_r1a",      0)
    n_r1b = stats.get("n_r1b",      0)
    n_r2  = stats.get("n_r2",       0)
    n_wor = stats.get("n_worsened", 0)
    n_p   = stats.get("n_patches",  len(df_slide))

    title_before = f"Slide {slide_id}  [{n_p} patches]  —  ML only"
    title_after  = (f"After PyReason  |  R1a={n_r1a}  R1b={n_r1b}  "
                    f"R2={n_r2}  chg={n_ch}  wors={n_wor}")

    _draw_panel(ax_before, df_slide, title=title_before, dc=dc, mode="before")
    _draw_panel(ax_after,  df_slide, title=title_after,  dc=dc, mode="after")


# ─────────────────────────────────────────────────────────────────
#  Legend — standalone, large font
# ─────────────────────────────────────────────────────────────────

def _build_legend_handles(dc: DrawConfig):
    import matplotlib.patches as mpatches
    ctx_label = (
        f"dashed rect = window ({dc.window_rows}×{dc.window_cols} patches)  [from {dc.source}]"
        if dc.section_mode == "window"
        else f"quadrant grid ({dc.sq}×{dc.sq})  [from {dc.source}]"
    )
    fill_items = [
        ("G — correct",        PRED_COLOR["G"],      "#888", 0.6),
        ("G+ — correct",       PRED_COLOR["Gplus"],  "#888", 0.6),
        ("Mixed — predicted",  PRED_COLOR["Mixed"],  "#888", 0.6),
        ("G — wrong pred",     WRONG_COLOR["G"],     "#888", 0.6),
        ("G+ — wrong pred",    WRONG_COLOR["Gplus"], "#888", 0.6),
        ("Mixed — wrong pred", WRONG_COLOR["Mixed"], "#888", 0.6),
    ]
    border_items = [
        ("Rule 1a: Mixed → window majority",    RULE_COLOR["rule1a"]),
        ("Rule 1b: minority → window majority", RULE_COLOR["rule1b"]),
        ("Rule 2:  neighbour agreement",        RULE_COLOR["rule2_neighbor"]),
        ("Rule 3:  cluster flag",               RULE_COLOR["rule3_cluster_flag"]),
    ]
    text_items = [
        "X  wrong prediction",
        "•  review flag (no pred change)",
        "////  anchor (HIGH conf)",
        ctx_label,
        "top = pred  |  bottom = tier + conf  (H/M/L)",
        "bright section bg = contains changed patch",
    ]
    elems = []
    for label, fc, ec, lw in fill_items:
        elems.append(mpatches.Patch(facecolor=fc, edgecolor=ec, linewidth=lw, label=label))
    elems.append(mpatches.Patch(facecolor="none", edgecolor="none", label=" "))
    for label, ec in border_items:
        elems.append(mpatches.Patch(facecolor="#E8E8E8", edgecolor=ec, linewidth=2.5, label=label))
    elems.append(mpatches.Patch(facecolor="none", edgecolor="none", label=" "))
    for txt in text_items:
        elems.append(mpatches.Patch(facecolor="none", edgecolor="none", label=txt))
    return elems


def make_legend(fig, dc: DrawConfig, large: bool = False):
    fs = 11 if large else 8
    hw = 1.8 if large else 1.4
    hh = 1.4 if large else 1.0
    fig.legend(
        handles=_build_legend_handles(dc),
        loc="lower center",
        ncol=4,
        framealpha=0.95,
        bbox_to_anchor=(0.5, 0.0),
        labelcolor="#1A1A2E",
        facecolor="#F5F5F5",
        edgecolor="#BBBBBB",
        handlelength=hw,
        handleheight=hh,
        columnspacing=1.5,
        handletextpad=0.7,
        prop={"family": "Arial", "size": fs, "weight": "normal"},
    )


def save_legend_standalone(out_dir: str, dc: DrawConfig):
    fig, ax = plt.subplots(figsize=(14, 3.5))
    ax.set_visible(False)
    fig.patch.set_facecolor(BG_COLOR)
    fig.legend(
        handles=_build_legend_handles(dc),
        loc="center",
        ncol=4,
        framealpha=0.95,
        labelcolor="#1A1A2E",
        facecolor="#F5F5F5",
        edgecolor="#AAAAAA",
        handlelength=1.8,
        handleheight=1.4,
        columnspacing=1.6,
        handletextpad=0.8,
        prop={"family": "Arial", "size": 12, "weight": "normal"},
    )
    path = os.path.join(out_dir, "legend.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close()
    print(f"  Saved legend → {path}")


# ─────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────

def _slide_stats(df_s: pd.DataFrame) -> dict:
    rule_col = df_s["rule_applied"].apply(_norm_rule)
    return dict(
        n_patches  = len(df_s),
        n_changed  = int(df_s["changed"].sum()),
        n_r1a      = int((rule_col == "rule1a").sum()),
        n_r1b      = int((rule_col == "rule1b").sum()),
        n_r2       = int((rule_col == "rule2_neighbor").sum()),
        n_worsened = int((df_s["outcome"] == "worsened").sum())
                    if "outcome" in df_s.columns else 0,
    )


def pick_interesting_slides(df: pd.DataFrame, n: int = 5) -> list:
    stats = df.groupby("slide_id").agg(
        n_patches =("image_name",   "count"),
        n_r1      =("rule_applied",
                    lambda x: x.apply(_norm_rule).isin({"rule1a","rule1b"}).sum()),
        n_worsened=("outcome",      lambda x: (x == "worsened").sum()),
        n_changed =("changed",      "sum"),
    ).reset_index()
    stats = stats.sort_values(["n_r1","n_worsened","n_changed"], ascending=False)
    return stats[stats["n_patches"] >= 4].head(n)["slide_id"].tolist()


def _find_csv(base_dir: str, filename: str = "patch_results.csv",
              subdir_priority: list[str] | None = None) -> str | None:
    """
    Search for filename under base_dir.
    Priority order is defined by subdir_priority (defaults to EVAL_SUBDIR_PRIORITY).
    Falls back to base_dir itself, then any subdirectory (sorted reverse).
    """
    if subdir_priority is None:
        subdir_priority = EVAL_SUBDIR_PRIORITY

    # 1. Check priority subdirs in order
    for sub in subdir_priority:
        p = os.path.join(base_dir, sub, filename)
        if os.path.exists(p):
            print(f"  [CSV] Found in priority subdir: {sub}/")
            return p

    # 2. Check base_dir itself
    p = os.path.join(base_dir, filename)
    if os.path.exists(p):
        print(f"  [CSV] Found in base dir")
        return p

    # 3. Fall back — any subdir, sorted reverse (newest first if timestamp-named)
    try:
        for sub in sorted(os.listdir(base_dir), reverse=True):
            p = os.path.join(base_dir, sub, filename)
            if os.path.exists(p):
                print(f"  [CSV] Found in fallback subdir: {sub}/")
                return p
    except Exception:
        pass

    return None


def load_data(source: str = "auto") -> tuple[pd.DataFrame, str]:
    if source not in ("auto", "sweep", "run_test") and os.path.exists(source):
        csv_path = source
    else:
        csv_path = _find_csv(config.OUTPUT_DIR)

    if csv_path is None:
        for root, _dirs, files in os.walk(config.OUTPUT_DIR):
            if "patch_results.csv" in files:
                csv_path = os.path.join(root, "patch_results.csv")
                break

    if csv_path is None:
        print(f"[ERROR] patch_results.csv not found under {config.OUTPUT_DIR}")
        sys.exit(1)

    print(f"  Loading: {csv_path}")
    return pd.read_csv(csv_path), csv_path


def _save_single(fig, path, facecolor=BG_COLOR, dpi=130):
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor=facecolor)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────

def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df, csv_path = load_data()
    df = df[df["image_type"] == "grid"].copy()
    print(f"Loaded {len(df)} grid patches from {df['slide_id'].nunique()} slides")
    print(f"Source: {csv_path}")

    # ── Load DrawConfig ONCE from config_used.txt (or config.py) ─
    # All drawing functions receive dc explicitly — nothing reads
    # config.py at draw time, so window borders are always consistent
    # with the evaluation that produced this CSV.
    dc = load_draw_config(csv_path)

    base = os.path.join(os.path.dirname(csv_path), "slide_grids")

    dirs = {
        "all":     os.path.join(base, "all"),
        "changed": os.path.join(base, "changed"),
        "rule1":   os.path.join(base, "by_rule", "rule1"),
        "rule2":   os.path.join(base, "by_rule", "rule2"),
        "rule3":   os.path.join(base, "by_rule", "rule3"),
        "rule4":   os.path.join(base, "by_rule", "rule4"),
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)

    print(f"Output root: {base}")

    save_legend_standalone(base, dc)

    all_slides = df["slide_id"].unique()
    n_changed  = 0

    for i, sid in enumerate(all_slides):
        df_s  = df[df["slide_id"] == sid].copy()
        stats = _slide_stats(df_s)

        rule_col    = df_s["rule_applied"].apply(_norm_rule)
        has_rule1   = bool((rule_col.isin({"rule1a","rule1b"})).any())
        has_rule2   = bool((rule_col == "rule2_neighbor").any())
        has_rule3   = bool((rule_col == "rule3_cluster_flag").any())
        has_rule4   = bool((rule_col == "rule4_review").any())
        any_changed = bool(df_s["changed"].any())

        n_p   = stats["n_patches"]
        n_ch  = stats["n_changed"]
        n_r1a = stats["n_r1a"]
        n_r1b = stats["n_r1b"]
        n_r2  = stats["n_r2"]
        n_wor = stats["n_worsened"]

        # ── (A) all/ ─────────────────────────────────────────────
        fig, (ax_b, ax_a) = plt.subplots(
            1, 2, figsize=(18, 6), gridspec_kw={"wspace": 0.08})
        fig.patch.set_facecolor(BG_COLOR)
        draw_slide_dual(ax_b, ax_a, df_s, sid, stats, dc)
        make_legend(fig, dc, large=False)
        plt.subplots_adjust(bottom=0.17, left=0.04, right=0.97)
        _save_single(fig, os.path.join(dirs["all"], f"slide_{sid}.png"))

        # ── (B) changed/ ─────────────────────────────────────────
        if any_changed:
            n_changed += 1
            fig, (ax_b, ax_a) = plt.subplots(
                1, 2, figsize=(18, 6), gridspec_kw={"wspace": 0.08})
            fig.patch.set_facecolor(BG_COLOR)
            draw_slide_dual(ax_b, ax_a, df_s, sid, stats, dc)
            make_legend(fig, dc, large=False)
            plt.subplots_adjust(bottom=0.17, left=0.04, right=0.97)
            _save_single(fig, os.path.join(dirs["changed"], f"slide_{sid}.png"))

        # ── (C) by_rule/rule1/ — THREE separate panels ───────────
        if has_rule1:
            ctx_str    = (f"window {dc.window_rows}×{dc.window_cols}"
                          if dc.section_mode == "window"
                          else f"quadrant {dc.sq}×{dc.sq}")
            title_ml   = f"Slide {sid}  [{n_p} patches]  —  ML ONLY"
            title_win  = f"Slide {sid}  —  ML ONLY + CONTEXT ({ctx_str})"
            title_after= (f"After PyReason  |  R1a={n_r1a}  R1b={n_r1b}  "
                          f"chg={n_ch}  wors={n_wor}")

            fig1, ax1 = plt.subplots(1, 1, figsize=(9, 6))
            fig1.patch.set_facecolor(BG_COLOR)
            plt.subplots_adjust(left=0.03, right=0.97, top=0.93, bottom=0.04)
            _draw_panel(ax1, df_s, title=title_ml, dc=dc, mode="before")
            _save_single(fig1, os.path.join(dirs["rule1"], f"{sid}_1_ml_only.png"))

            fig2, ax2 = plt.subplots(1, 1, figsize=(9, 6))
            fig2.patch.set_facecolor(BG_COLOR)
            plt.subplots_adjust(left=0.03, right=0.97, top=0.93, bottom=0.04)
            _draw_panel(ax2, df_s, title=title_win, dc=dc, mode="before_window")
            _save_single(fig2, os.path.join(dirs["rule1"], f"{sid}_2_ml_window.png"))

            fig3, ax3 = plt.subplots(1, 1, figsize=(9, 6))
            fig3.patch.set_facecolor(BG_COLOR)
            plt.subplots_adjust(left=0.03, right=0.97, top=0.93, bottom=0.04)
            _draw_panel(ax3, df_s, title=title_after, dc=dc, mode="after")
            _save_single(fig3, os.path.join(dirs["rule1"], f"{sid}_3_after.png"))

        # ── (D) by_rule/rule2/ ───────────────────────────────────
        if has_rule2 and not has_rule1:
            title_ml    = f"Slide {sid}  [{n_p} patches]  —  ML ONLY"
            title_after = f"After PyReason  |  R2={n_r2}  chg={n_ch}  wors={n_wor}"

            fig1, ax1 = plt.subplots(1, 1, figsize=(9, 6))
            fig1.patch.set_facecolor(BG_COLOR)
            plt.subplots_adjust(left=0.03, right=0.97, top=0.93, bottom=0.04)
            _draw_panel(ax1, df_s, title=title_ml, dc=dc, mode="before")
            _save_single(fig1, os.path.join(dirs["rule2"], f"{sid}_1_ml_only.png"))

            fig2, ax2 = plt.subplots(1, 1, figsize=(9, 6))
            fig2.patch.set_facecolor(BG_COLOR)
            plt.subplots_adjust(left=0.03, right=0.97, top=0.93, bottom=0.04)
            _draw_panel(ax2, df_s, title=title_after, dc=dc, mode="after")
            _save_single(fig2, os.path.join(dirs["rule2"], f"{sid}_2_after.png"))

        # ── (E) by_rule/rule3/ ───────────────────────────────────
        if has_rule3 and not has_rule1 and not has_rule2:
            title_ml    = f"Slide {sid}  [{n_p} patches]  —  ML ONLY"
            title_after = f"After PyReason  |  R3 cluster flagged  chg={n_ch}  wors={n_wor}"

            fig1, ax1 = plt.subplots(1, 1, figsize=(9, 6))
            fig1.patch.set_facecolor(BG_COLOR)
            plt.subplots_adjust(left=0.03, right=0.97, top=0.93, bottom=0.04)
            _draw_panel(ax1, df_s, title=title_ml, dc=dc, mode="before")
            _save_single(fig1, os.path.join(dirs["rule3"], f"{sid}_1_ml_only.png"))

            fig2, ax2 = plt.subplots(1, 1, figsize=(9, 6))
            fig2.patch.set_facecolor(BG_COLOR)
            plt.subplots_adjust(left=0.03, right=0.97, top=0.93, bottom=0.04)
            _draw_panel(ax2, df_s, title=title_after, dc=dc, mode="after")
            _save_single(fig2, os.path.join(dirs["rule3"], f"{sid}_2_after.png"))

        # ── (F) by_rule/rule4/ ───────────────────────────────────
        if has_rule4 and not has_rule1 and not has_rule2 and not has_rule3:
            title_ml    = f"Slide {sid}  [{n_p} patches]  —  ML ONLY"
            title_after = f"After PyReason  |  R4 low-conf flagged  chg={n_ch}  wors={n_wor}"

            fig1, ax1 = plt.subplots(1, 1, figsize=(9, 6))
            fig1.patch.set_facecolor(BG_COLOR)
            plt.subplots_adjust(left=0.03, right=0.97, top=0.93, bottom=0.04)
            _draw_panel(ax1, df_s, title=title_ml, dc=dc, mode="before")
            _save_single(fig1, os.path.join(dirs["rule4"], f"{sid}_1_ml_only.png"))

            fig2, ax2 = plt.subplots(1, 1, figsize=(9, 6))
            fig2.patch.set_facecolor(BG_COLOR)
            plt.subplots_adjust(left=0.03, right=0.97, top=0.93, bottom=0.04)
            _draw_panel(ax2, df_s, title=title_after, dc=dc, mode="after")
            _save_single(fig2, os.path.join(dirs["rule4"], f"{sid}_2_after.png"))

        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(all_slides)} done…")

    print(f"\n  Saved {len(all_slides)} slides → {dirs['all']}/")
    print(f"  Saved {n_changed} changed slides → {dirs['changed']}/")

    # ── 5-sample summary ─────────────────────────────────────────
    chosen = pick_interesting_slides(df, n=5) or list(df["slide_id"].unique()[:5])
    print(f"\nGenerating 5-sample summary: {chosen}")

    fig, axes = plt.subplots(
        len(chosen), 2,
        figsize=(18, 5.5 * len(chosen)),
        gridspec_kw={"wspace": 0.07, "hspace": 0.38},
    )
    if len(chosen) == 1:
        axes = [axes]
    fig.patch.set_facecolor(BG_COLOR)

    for idx, sid in enumerate(chosen):
        df_s  = df[df["slide_id"] == sid].copy()
        stats = _slide_stats(df_s)
        draw_slide_dual(axes[idx][0], axes[idx][1], df_s, sid, stats, dc)

    make_legend(fig, dc, large=False)
    fig.suptitle(
        "PYREASON — BEFORE / AFTER  (5 MOST-CHANGED SLIDES)",
        fontsize=20, color="#1A1A2E", y=1.003, **TITLE_FONT,
    )
    plt.subplots_adjust(bottom=0.06, left=0.04, right=0.97)
    summary_path = os.path.join(base, "summary_5_samples.png")
    plt.savefig(summary_path, dpi=130, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close()
    print(f"  Saved summary → {summary_path}")
    print(f"\n  Context params used: {dc.summary()}")


if __name__ == "__main__":
    main()