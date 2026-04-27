#!/usr/bin/env python
"""Visualization helpers for HPO, CV learning curves, and Confusion matrix"""

from __future__ import annotations

import os
from typing import Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix


# ---------------- optuna plots ----------------
def export_optuna_plots(study, out_dir: str) -> None:
    """Save Optuna optimization history + param importance (and extras) to out_dir."""
    os.makedirs(out_dir, exist_ok=True)
    try:
        from optuna.visualization import (
            plot_optimization_history,
            plot_param_importances,
            plot_parallel_coordinate,
            plot_slice,
        )

        plot_optimization_history(study).write_html(
            os.path.join(out_dir, "optuna_optimization_history.html")
        )
        plot_param_importances(study).write_html(
            os.path.join(out_dir, "optuna_param_importance.html")
        )
        plot_parallel_coordinate(study).write_html(
            os.path.join(out_dir, "optuna_parallel_coordinate.html")
        )
        plot_slice(study).write_html(os.path.join(out_dir, "optuna_slice.html"))
        print("[Viz] Optuna HTML plots saved.")
    except Exception as e:
        print(f"[Viz] Optuna plots failed ({e}). Ensure optuna.visualization is available.")


# ---------------- internal helpers ----------------
def _plot_mean_std_curve(
    df: pd.DataFrame,
    metric_cols: List[str],
    out_png: str,
    ylabel: str,
    title: str | None = None,
) -> None:
    """
    Plot CV mean ± std over epochs for one or more metric columns.

    Expects: columns ['epoch', 'fold', *metric_cols]
    Missing metrics are skipped.
    """
    if df.empty:
        print(f"[Viz] Empty DF for {metric_cols}. Skipping.")
        return

    present = [m for m in metric_cols if m in df.columns]
    if not present:
        print(f"[Viz] No requested metrics present: {metric_cols}")
        return

    plt.figure(figsize=(7.2, 4.8))
    g = df.groupby("epoch")

    for m in present:
        mean = g[m].mean()
        std = g[m].std(ddof=0)

        epochs = mean.index.values
        mean_v = mean.values
        std_v = np.where(np.isnan(std.values), 0.0, std.values)

        plt.plot(epochs, mean_v, label=f"{m} (mean)")
        plt.fill_between(epochs, mean_v - std_v, mean_v + std_v, alpha=0.25, label=f"{m} ±1 SD")

    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(title or f"{', '.join(present)} (CV mean ± std)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()
    print(f"[Viz] Saved curve → {out_png}")


# ---------------- auto-detect primary metric ----------------
# Priority order: whichever val_ metric is present first wins.
_PRIMARY_METRIC_CANDIDATES = [
    "val_mcc", "val_f2", "val_f1", "val_acc", "val_recall", "val_precision"
]

def _detect_primary_metric(cols: set, hint: str | None = None) -> str:
    """
    Return the best available primary metric column from `cols`.
    If `hint` is given and exists, it wins. Otherwise falls back through
    the priority list. Defaults to 'val_mcc' if nothing matches.
    """
    if hint and hint in cols:
        return hint
    for c in _PRIMARY_METRIC_CANDIDATES:
        if c in cols:
            return c
    return "val_mcc"  # safe default even if missing (plots will be skipped gracefully)


# ---------------- CV overview plots ----------------
def plot_cv_mean_std_curves(
    all_eval_excel: str,
    out_dir: str,
    monitor_metric: str | None = None,   # auto-detected if None
) -> None:
    """
    From per-epoch, per-fold metrics (Excel) produce:
      - cv_<primary>_mean_std.png   (primary metric — auto-detected)
      - cv_loss_mean_std.png

    The primary metric is resolved automatically from columns present in the
    Excel file, following the priority: val_mcc > val_f2 > val_f1 > val_acc.
    Pass `monitor_metric` to override.
    """
    if not os.path.exists(all_eval_excel):
        print(f"[Viz] No eval Excel found: {all_eval_excel}")
        return

    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_excel(all_eval_excel)
    cols = set(df.columns)

    if not {"epoch", "fold"}.issubset(cols):
        print(f"[Viz] Missing core columns in {all_eval_excel}. Need ['epoch','fold'].")
        return

    # ── auto-detect primary metric ───────────────────────────────────────────
    primary = _detect_primary_metric(cols, hint=monitor_metric)
    train_primary = primary.replace("val_", "train_")
    print(f"[Viz] Auto-detected primary metric: {primary}")

    if primary in cols:
        pm_cols = [primary] + ([train_primary] if train_primary in cols else [])
        _plot_mean_std_curve(
            df[["epoch", "fold"] + pm_cols],
            metric_cols=pm_cols,
            out_png=os.path.join(out_dir, f"cv_{primary}_mean_std.png"),
            ylabel=primary.replace("val_", "").upper(),
            title=(f"Train vs Val {primary} (CV mean ± std)"
                   if train_primary in cols
                   else f"Validation {primary} (CV mean ± std)"),
        )
    else:
        print(f"[Viz] Skipped primary metric plot: '{primary}' not found in columns.")

    # ── always plot all available val_ metrics (secondary panel) ────────────
    secondary_metrics = [
        c for c in ["val_f1", "val_f2", "val_mcc", "val_acc", "val_recall", "val_precision"]
        if c in cols and c != primary
    ]
    if secondary_metrics:
        _plot_mean_std_curve(
            df[["epoch", "fold"] + secondary_metrics],
            metric_cols=secondary_metrics,
            out_png=os.path.join(out_dir, "cv_all_val_metrics_mean_std.png"),
            ylabel="Score",
            title="All Validation Metrics (CV mean ± std)",
        )

    if "val_loss" in cols:
        loss_cols = ["val_loss"] + (["train_loss"] if "train_loss" in cols else [])
        _plot_mean_std_curve(
            df[["epoch", "fold"] + loss_cols],
            metric_cols=loss_cols,
            out_png=os.path.join(out_dir, "cv_loss_mean_std.png"),
            ylabel="Loss",
            title="Train vs Val Loss (CV mean ± std)" if "train_loss" in cols else "Validation Loss (CV mean ± std)",
        )
    else:
        print("[Viz] Skipped loss plot: 'val_loss' not found.")


def plot_per_fold_curves(
    all_eval_excel: str,
    out_dir: str,
    metric_col: str | None = None,   # auto-detected if None
) -> None:
    if not os.path.exists(all_eval_excel):
        print(f"[Viz] No eval Excel found: {all_eval_excel}")
        return
    df = pd.read_excel(all_eval_excel)
    cols = set(df.columns)

    # auto-detect if not specified
    if metric_col is None:
        metric_col = _detect_primary_metric(cols)
        print(f"[Viz] plot_per_fold_curves: auto-detected metric = {metric_col}")

    if not {"epoch", "fold", metric_col}.issubset(cols):
        print(f"[Viz] Missing columns for per-fold curves. Need 'epoch','fold','{metric_col}'.")
        return
    os.makedirs(out_dir, exist_ok=True)

    plt.figure(figsize=(7.2, 4.8))
    for f, g in df.groupby("fold"):
        g = g.sort_values("epoch")
        plt.plot(g["epoch"], g[metric_col], alpha=0.5, linewidth=1, label=f"fold {f}")

    mean = df.groupby("epoch")[metric_col].mean()
    plt.plot(mean.index, mean.values, linewidth=2.5, label="mean", zorder=10)
    plt.xlabel("Epoch")
    plt.ylabel(metric_col)
    plt.title(f"Per-fold {metric_col}")
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"per_fold_{metric_col}.png"), dpi=300)
    plt.close()
    print(f"[Viz] Saved per-fold {metric_col} curves.")


# ---------------- confusion matrix ----------------
def save_confusion_matrix(
    y_true,
    y_pred,
    out_path: str,
    title: str = "Confusion Matrix",
    labels: Iterable[str] = ("0", "1"),
    num_classes: int = 2,
    cmap: str = "Blues",
    annotate_fmt: str = "d",
    figsize: Tuple[int, int] = (8, 6),
    probs=None,
    threshold=None,
) -> None:
    """Save a confusion matrix heatmap (supports 2-class and 3-class)."""
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    class_indices = list(range(num_classes))
    cm = confusion_matrix(y_true, y_pred, labels=class_indices)

    figsize_auto = (max(6, num_classes * 2 + 2), max(5, num_classes * 2))
    plt.figure(figsize=figsize_auto)
    ax = sns.heatmap(
        cm,
        annot=True,
        fmt=annotate_fmt,
        cmap=cmap,
        xticklabels=list(labels),
        yticklabels=list(labels),
        cbar=True,
    )
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


# ---------------- combined F1+loss plots ----------------
def plot_cv_primary_and_loss_combined(
    all_eval_excel: str,
    out_dir: str,
    primary_col_val: str | None = None,    # auto-detected if None (e.g. val_mcc)
    primary_col_train: str | None = None,  # auto-detected if None
    loss_col_val: str = "val_loss",
    loss_col_train: str = "train_loss",
    out_name: str | None = None,           # auto-named after primary metric
) -> None:
    """Single figure with CV mean±std curves: left axis = primary metric (auto-detected), right axis = Loss."""
    if not os.path.exists(all_eval_excel):
        print(f"[Viz] No eval Excel found: {all_eval_excel}")
        return

    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_excel(all_eval_excel)
    cols = set(df.columns)
    if not {"epoch", "fold"}.issubset(cols):
        print("[Viz] Missing core columns. Need ['epoch','fold'].")
        return

    # auto-detect
    if primary_col_val is None:
        primary_col_val = _detect_primary_metric(cols)
    if primary_col_train is None:
        primary_col_train = primary_col_val.replace("val_", "train_")
    if out_name is None:
        out_name = f"cv_{primary_col_val}_and_loss_combined.png"

    metric_label = primary_col_val.replace("val_", "").upper()
    print(f"[Viz] plot_cv_primary_and_loss_combined: primary = {primary_col_val}")

    g = df.groupby("epoch")
    epochs = g.size().index.values

    def _mean_std(col: str):
        if col not in df.columns:
            return None, None
        m = g[col].mean().values
        s = g[col].std(ddof=0).values
        return m, np.where(np.isnan(s), 0.0, s)

    pm_val_m, pm_val_s     = _mean_std(primary_col_val)
    pm_tr_m,  pm_tr_s      = _mean_std(primary_col_train)
    loss_val_m, loss_val_s = _mean_std(loss_col_val)
    loss_tr_m,  loss_tr_s  = _mean_std(loss_col_train)

    if not any(x is not None for x in (pm_val_m, pm_tr_m, loss_val_m, loss_tr_m)):
        print(f"[Viz] No {primary_col_val} or Loss columns found — nothing to plot.")
        return

    fig, ax_pm = plt.subplots(figsize=(7.6, 4.8))
    lines, labels = [], []

    ax_pm.set_xlabel("Epoch")
    ax_pm.set_ylabel(metric_label)
    ax_pm.set_ylim(0.0, 1.0)

    if pm_val_m is not None:
        ln, = ax_pm.plot(epochs, pm_val_m, label=f"{primary_col_val} (mean)")
        ax_pm.fill_between(epochs, pm_val_m - pm_val_s, pm_val_m + pm_val_s, alpha=0.25)
        lines.append(ln)
        labels.append(f"{primary_col_val} (mean ± sd)")
    if pm_tr_m is not None:
        ln, = ax_pm.plot(epochs, pm_tr_m, linestyle="--", label=f"{primary_col_train} (mean)")
        ax_pm.fill_between(epochs, pm_tr_m - pm_tr_s, pm_tr_m + pm_tr_s, alpha=0.20)
        lines.append(ln)
        labels.append(f"{primary_col_train} (mean ± sd)")

    ax_loss = ax_pm.twinx()
    ax_loss.set_ylabel("Loss")

    if loss_val_m is not None:
        ln, = ax_loss.plot(epochs, loss_val_m, label=f"{loss_col_val} (mean)")
        ax_loss.fill_between(epochs, loss_val_m - loss_val_s, loss_val_m + loss_val_s, alpha=0.25)
        lines.append(ln)
        labels.append(f"{loss_col_val} (mean ± sd)")
    if loss_tr_m is not None:
        ln, = ax_loss.plot(epochs, loss_tr_m, linestyle="--", label=f"{loss_col_train} (mean)")
        ax_loss.fill_between(epochs, loss_tr_m - loss_tr_s, loss_tr_m + loss_tr_s, alpha=0.20)
        lines.append(ln)
        labels.append(f"{loss_col_train} (mean ± sd)")

    title_bits = []
    if (pm_val_m is not None) or (pm_tr_m is not None):
        title_bits.append(metric_label)
    if (loss_val_m is not None) or (loss_tr_m is not None):
        title_bits.append("Loss")
    plt.title(f"CV Mean ± Std: {' & '.join(title_bits)}")

    ax_pm.legend(lines, labels, loc="best")
    fig.tight_layout()
    out_png = os.path.join(out_dir, out_name)
    plt.savefig(out_png, dpi=300)
    plt.close(fig)
    print(f"[Viz] Saved combined curve → {out_png}")


def _group_by_epoch_mean(df: pd.DataFrame, col: str):
    g = df.groupby("epoch")
    return g[col].mean().index.values, g[col].mean().values


def plot_cv_mean_primary(
    all_eval_excel: str,
    out_dir: str,
    primary_col_val: str | None = None,   # auto-detected if None (e.g. val_mcc)
    primary_col_train: str | None = None,
    out_name: str | None = None,
) -> None:
    """Mean curves only for the auto-detected primary metric (+ optional train counterpart)."""
    if not os.path.exists(all_eval_excel):
        print(f"[Viz] No eval Excel found: {all_eval_excel}")
        return
    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_excel(all_eval_excel)
    cols = set(df.columns)
    if not {"epoch", "fold"}.issubset(cols):
        print("[Viz] Need columns ['epoch','fold'].")
        return

    if primary_col_val is None:
        primary_col_val = _detect_primary_metric(cols)
    if primary_col_train is None:
        primary_col_train = primary_col_val.replace("val_", "train_")
    if out_name is None:
        out_name = f"cv_{primary_col_val}_mean_only.png"

    metric_label = primary_col_val.replace("val_", "").upper()

    plt.figure(figsize=(7.6, 4.8))
    lines, labels = [], []

    if primary_col_val in cols:
        e, y = _group_by_epoch_mean(df, primary_col_val)
        ln, = plt.plot(e, y, label=f"{primary_col_val} (mean)")
        lines.append(ln)
        labels.append(f"{primary_col_val} (mean)")
    if primary_col_train in cols:
        e, y = _group_by_epoch_mean(df, primary_col_train)
        ln, = plt.plot(e, y, label=f"{primary_col_train} (mean)")
        lines.append(ln)
        labels.append(f"{primary_col_train} (mean)")

    if not lines:
        print(f"[Viz] No {primary_col_val} columns found to plot.")
        return

    plt.xlabel("Epoch")
    plt.ylabel(metric_label)
    plt.ylim(0.0, 1.0)
    plt.title(f"Train vs Val {metric_label} (CV mean)")
    plt.legend(lines, labels, loc="best")
    plt.tight_layout()
    out_png = os.path.join(out_dir, out_name)
    plt.savefig(out_png, dpi=300)
    plt.close()
    print(f"[Viz] Saved → {out_png}")


def plot_cv_mean_loss(
    all_eval_excel: str,
    out_dir: str,
    loss_col_val: str = "val_loss",
    loss_col_train: str = "train_loss",
    out_name: str = "cv_loss_mean_only.png",
) -> None:
    """Mean curves only: val_loss (+ optional train_loss)."""
    if not os.path.exists(all_eval_excel):
        print(f"[Viz] No eval Excel found: {all_eval_excel}")
        return
    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_excel(all_eval_excel)
    if not {"epoch", "fold"}.issubset(df.columns):
        print("[Viz] Need columns ['epoch','fold'].")
        return

    plt.figure(figsize=(7.6, 4.8))
    lines, labels = [], []

    if loss_col_val in df.columns:
        e, y = _group_by_epoch_mean(df, loss_col_val)
        ln, = plt.plot(e, y, label=f"{loss_col_val} (mean)")
        lines.append(ln)
        labels.append(f"{loss_col_val} (mean)")
    if loss_col_train in df.columns:
        e, y = _group_by_epoch_mean(df, loss_col_train)
        ln, = plt.plot(e, y, label=f"{loss_col_train} (mean)")
        lines.append(ln)
        labels.append(f"{loss_col_train} (mean)")

    if not lines:
        print("[Viz] No Loss columns found to plot.")
        return

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Train vs Val Loss (CV mean)")
    plt.legend(lines, labels, loc="best")
    plt.tight_layout()
    out_png = os.path.join(out_dir, out_name)
    plt.savefig(out_png, dpi=300)
    plt.close()
    print(f"[Viz] Saved → {out_png}")


def plot_cv_primary_and_loss_mean_only(
    all_eval_excel: str,
    out_dir: str,
    primary_col_val: str | None = None,
    primary_col_train: str | None = None,
    loss_col_val: str = "val_loss",
    loss_col_train: str = "train_loss",
    out_name: str | None = None,
) -> None:
    """
    Plot CV mean curves (no std) on one figure:
      - Left axis: primary metric (auto-detected, e.g. val_mcc)
      - Right axis: Loss
    """
    if not os.path.exists(all_eval_excel):
        print(f"[Viz] No eval Excel found: {all_eval_excel}")
        return

    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_excel(all_eval_excel)
    cols = set(df.columns)
    if not {"epoch", "fold"}.issubset(cols):
        print("[Viz] Missing core columns. Need ['epoch','fold'].")
        return

    if primary_col_val is None:
        primary_col_val = _detect_primary_metric(cols)
    if primary_col_train is None:
        primary_col_train = primary_col_val.replace("val_", "train_")
    if out_name is None:
        out_name = f"cv_{primary_col_val}_and_loss_mean_only.png"

    metric_label = primary_col_val.replace("val_", "").upper()

    if not any(c in cols for c in [primary_col_val, primary_col_train, loss_col_val, loss_col_train]):
        print("[Viz] No plottable columns found.")
        return

    g = df.groupby("epoch")
    epochs = g.size().index.values

    def _mean(col: str):
        return g[col].mean().values if col in cols else None

    pm_val_m   = _mean(primary_col_val)
    pm_tr_m    = _mean(primary_col_train)
    loss_val_m = _mean(loss_col_val)
    loss_tr_m  = _mean(loss_col_train)

    fig, ax_pm = plt.subplots(figsize=(7.6, 4.8))
    lines, labels = [], []

    ax_pm.set_xlabel("Epoch")
    ax_pm.set_ylabel(metric_label)
    ax_pm.set_ylim(0.0, 1.0)

    if pm_val_m is not None:
        ln, = ax_pm.plot(epochs, pm_val_m, label=f"{primary_col_val} (mean)")
        lines.append(ln); labels.append(f"{primary_col_val} (mean)")
    if pm_tr_m is not None:
        ln, = ax_pm.plot(epochs, pm_tr_m, label=f"{primary_col_train} (mean)")
        lines.append(ln); labels.append(f"{primary_col_train} (mean)")

    ax_loss = ax_pm.twinx()
    ax_loss.set_ylabel("Loss")

    if loss_val_m is not None:
        ln, = ax_loss.plot(epochs, loss_val_m, label=f"{loss_col_val} (mean)")
        lines.append(ln); labels.append(f"{loss_col_val} (mean)")
    if loss_tr_m is not None:
        ln, = ax_loss.plot(epochs, loss_tr_m, label=f"{loss_col_train} (mean)")
        lines.append(ln); labels.append(f"{loss_col_train} (mean)")

    title_bits = []
    if pm_val_m is not None or pm_tr_m is not None:
        title_bits.append(metric_label)
    if loss_val_m is not None or loss_tr_m is not None:
        title_bits.append("Loss")
    plt.title("CV Mean: " + " & ".join(title_bits) if title_bits else "CV Mean")

    ax_pm.legend(lines, labels, loc="best")
    fig.tight_layout()
    out_png = os.path.join(out_dir, out_name)
    plt.savefig(out_png, dpi=300)
    plt.close(fig)
    print(f"[Viz] Saved mean-only curve → {out_png}")

# --- Boxplot: best epoch per fold pulled from Excel ---
def plot_cv_boxplot(metric_dict: dict[str, list[float]], out_png: str) -> None:
    """
    Render a boxplot for {metric_name: [values per fold]} and save to out_png.
    """
    if not metric_dict:
        print("[Viz] Boxplot: empty metric_dict.")
        return

    metrics = list(metric_dict.keys())
    vals = [metric_dict[m] for m in metrics]
    means = [np.mean(v) for v in vals]

    pos = np.arange(1, len(metrics) + 1)
    plt.figure(figsize=(7.6, 4.8))
    bp = plt.boxplot(vals, positions=pos, widths=0.45, patch_artist=True, showfliers=False)

    # style
    for b in bp["boxes"]:
        b.set_facecolor("#5fa8d3"); b.set_alpha(0.45); b.set_edgecolor("black")
    for w in bp["whiskers"]: w.set_color("black")
    for c in bp["caps"]:     c.set_color("black")
    for m in bp["medians"]:  m.set_color("black"); m.set_linewidth(1.5)

    # mean dots + labels
    for i, mu in enumerate(means, start=1):
        plt.scatter(i, mu, color="black", s=20, zorder=5)

    all_vals = [v for vs in vals for v in vs]
    y_min = max(0.0, float(np.min(all_vals)) - 0.05) if all_vals else 0.0
    y_max = min(1.0, float(np.max(all_vals)) + 0.05) if all_vals else 1.0
    plt.xlim(0.5, len(metrics) + 0.5)
    plt.ylim(y_min, y_max)
    plt.xticks(pos, metrics, fontsize=12)
    plt.ylabel("Score", fontsize=13)
    plt.title("Cross-Validation Results Across Metrics", fontsize=14, pad=16)
    for s in ["top", "right", "left", "bottom"]:
        plt.gca().spines[s].set_visible(False)
    plt.gca().yaxis.grid(True, linestyle="-", alpha=0.2)
    plt.tight_layout()

    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[Viz] Saved CV boxplot → {out_png}")


def plot_cv_boxplot_from_excel(
    all_eval_excel: str,
    out_dir: str,
    monitor_col: str | None = None,   # auto-detected if None
    out_name: str = "cv_boxplot.png",
) -> None:
    """
    Read all_eval_metrics.xlsx, select the best row per fold using `monitor_col`
    (auto-detected if None, e.g. val_mcc), and plot a boxplot for all key metrics.
    Robust to truncated column names like 'val_precisio'.
    """
    if not os.path.exists(all_eval_excel):
        print(f"[Viz] Boxplot skipped: not found → {all_eval_excel}")
        return

    df = pd.read_excel(all_eval_excel)
    if df.empty:
        print("[Viz] Boxplot skipped: empty dataframe.")
        return

    def norm(s: str) -> str:
        return str(s).strip().lower().replace(" ", "_")
    df.columns = [norm(c) for c in df.columns]

    def find_col(cands: list[str]) -> str | None:
        for c in cands:
            if c in df.columns:
                return c
        for c in cands:
            for cc in df.columns:
                if cc.startswith(c):
                    return cc
        return None

    col_fold   = find_col(["fold"])
    col_acc    = find_col(["val_acc", "val_accuracy"])
    col_prec   = find_col(["val_precision", "val_precisio", "val_prec"])
    col_recall = find_col(["val_recall"])
    col_f1     = find_col(["val_f1"])
    col_mcc    = find_col(["val_mcc"])

    # auto-detect monitor col — prefer val_mcc
    if monitor_col is None:
        monitor_col = _detect_primary_metric(set(df.columns))
    col_mon = find_col([monitor_col]) or col_mcc or col_f1

    needed = [col_fold, col_mon]
    if any(c is None for c in needed):
        print(f"[Viz] Boxplot skipped: required columns missing. Have: {df.columns.tolist()}")
        return

    best_idx = df.groupby(col_fold)[col_mon].idxmax()
    best = df.loc[best_idx].sort_values(by=col_fold)

    # Build metric dict with whatever columns are available
    metric_dict = {}
    if col_acc:    metric_dict["Accuracy"]  = best[col_acc].astype(float).tolist()
    if col_prec:   metric_dict["Precision"] = best[col_prec].astype(float).tolist()
    if col_recall: metric_dict["Recall"]    = best[col_recall].astype(float).tolist()
    if col_f1:     metric_dict["F1"]        = best[col_f1].astype(float).tolist()
    if col_mcc:    metric_dict["MCC"]       = best[col_mcc].astype(float).tolist()

    out_png = os.path.join(out_dir, out_name)
    plot_cv_boxplot(metric_dict, out_png)


# ---------------- driver ----------------
def _run_cv_plots(all_eval_path: str, out_dir: str, monitor_metric: str | None = None) -> None:
    """
    Run all CV plots. Primary metric is auto-detected from the Excel columns
    (or pass monitor_metric to override, e.g. 'val_mcc').
    """
    if not os.path.exists(all_eval_path):
        print(f"[Viz] Skipped CV curves: not found → {all_eval_path}")
        return

    plot_cv_mean_std_curves(all_eval_path, out_dir, monitor_metric=monitor_metric)
    plot_per_fold_curves(all_eval_path, out_dir, metric_col=monitor_metric)
    plot_per_fold_curves(all_eval_path, out_dir, metric_col="val_loss")
    plot_cv_primary_and_loss_combined(all_eval_path, out_dir, primary_col_val=monitor_metric)
    plot_cv_primary_and_loss_mean_only(all_eval_path, out_dir, primary_col_val=monitor_metric)
    plot_cv_mean_loss(all_eval_path, out_dir)
    plot_cv_mean_primary(all_eval_path, out_dir, primary_col_val=monitor_metric)
    plot_cv_boxplot_from_excel(all_eval_path, out_dir, monitor_col=monitor_metric, out_name="cv_boxplot.png")


# ---------------- backward-compat aliases ----------------
# Keep old names working if anything imports them by the old name
plot_cv_mean_f2               = plot_cv_mean_primary
plot_cv_f1_and_loss_combined  = plot_cv_primary_and_loss_combined
plot_cv_f1_and_loss_mean_only = plot_cv_primary_and_loss_mean_only