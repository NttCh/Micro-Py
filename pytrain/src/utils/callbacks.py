#!/usr/bin/env python
import pytrain.src._path_fix  # noqa: F401  — ensures repo root is on sys.path
"""Custom callbacks: progress tracking, evaluation, Excel export, Optuna reporting."""

import math
import os
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
)
from torch import nn

import config as config


# ---------------- progress ----------------
class CleanTQDMProgressBar(TQDMProgressBar):
    def init_train_tqdm(self):
        bar = super().init_train_tqdm()
        bar.leave = False
        return bar


class TrialFoldProgressCallback(pl.Callback):
    def __init__(
        self,
        trial_number: Optional[int] = None,
        total_trials: Optional[int] = None,
        fold_number: Optional[int] = None,
        total_folds: Optional[int] = None,
    ):
        super().__init__()
        self.trial_number = trial_number
        self.total_trials = total_trials
        self.fold_number = fold_number
        self.total_folds = total_folds

    def on_train_start(self, trainer, pl_module) -> None:  # noqa: ANN001
        msgs = []
        if self.trial_number is not None and self.total_trials is not None:
            msgs.append(f"Trial {self.trial_number}/{self.total_trials}")
        if self.fold_number is not None and self.total_folds is not None:
            msgs.append(f"Fold {self.fold_number}/{self.total_folds}")

        if msgs:
            print(" | ".join(msgs))


class OverallProgressCallback(pl.Callback):
    def on_train_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self.total_epochs = trainer.max_epochs

    def on_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        e = trainer.current_epoch + 1
        rem = self.total_epochs - trainer.current_epoch
        print(f"[OverallProgress] Epoch {e}/{self.total_epochs} - Remaining: {rem}")


# ---------------- evaluation ----------------
@torch.no_grad()
def _eval_loader(
    model: torch.nn.Module,
    loader: Iterable,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    ce = nn.CrossEntropyLoss(reduction="sum")
    n, total_loss = 0, 0.0
    y_true, y_pred = [], []

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        total_loss += float(ce(logits, labels).item())
        n += int(labels.numel())
        preds = torch.argmax(logits, dim=1)
        y_pred.extend(preds.detach().cpu().tolist())
        y_true.extend(labels.detach().cpu().tolist())

    if n == 0:
        return dict(loss=0.0, acc=0.0, precision=0.0, recall=0.0, f1=0.0, f2=0.0)

    avg_loss = total_loss / n
    num_classes = len(set(y_true) | set(y_pred))
    avg = "macro" if num_classes > 2 else "binary"
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average=avg, zero_division=0)
    rec = recall_score(y_true, y_pred, average=avg, zero_division=0)
    f1 = f1_score(y_true, y_pred, average=avg, zero_division=0)
    f2 = fbeta_score(y_true, y_pred, beta=2, average=avg, zero_division=0)
    return dict(loss=avg_loss, acc=acc, precision=prec, recall=rec, f1=f1, f2=f2)


class EpochEndEvalCallback(pl.Callback):
    """Evaluate train_eval and val loaders at epoch end; log F2 on the progress bar."""

    def __init__(self, train_eval_loader, val_loader):
        super().__init__()
        self.train_eval_loader = train_eval_loader
        self.val_loader = val_loader

    def on_validation_epoch_end(self, trainer, pl_module) -> None:  # noqa: ANN001
        device = pl_module.device
        tr = _eval_loader(pl_module, self.train_eval_loader, device)
        va = _eval_loader(pl_module, self.val_loader, device)

        for k, v in tr.items():
            v = 0.0 if (isinstance(v, float) and math.isnan(v)) else float(v)
            trainer.callback_metrics[f"train_{k}"] = torch.tensor(v)
        for k, v in va.items():
            v = 0.0 if (isinstance(v, float) and math.isnan(v)) else float(v)
            trainer.callback_metrics[f"val_{k}"] = torch.tensor(v)

        pl_module.log("train_f1", trainer.callback_metrics["train_f1"], prog_bar=True, on_epoch=True, logger=True)
        pl_module.log("val_f1", trainer.callback_metrics["val_f1"], prog_bar=True, on_epoch=True, logger=True)


class TrainSetEvalCallback(pl.Callback):
    """Run a clean evaluation on the training set (using val transforms) after each epoch."""

    def __init__(self, train_eval_loader):
        super().__init__()
        self.train_eval_loader = train_eval_loader
        self.criterion = nn.CrossEntropyLoss()

    @torch.no_grad()
    def on_validation_epoch_end(self, trainer, pl_module) -> None:  # noqa: ANN001
        pl_module.eval()

        all_preds, all_labels = [], []
        total_loss, batches = 0.0, 0

        for images, labels in self.train_eval_loader:
            images, labels = images.to(pl_module.device), labels.to(pl_module.device)
            logits = pl_module(images)
            total_loss += float(self.criterion(logits, labels).item())
            batches += 1
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.detach().cpu().numpy())
            all_labels.extend(labels.detach().cpu().numpy())

        if batches == 0:
            pl_module.train()
            return

        avg_loss = total_loss / batches
        num_classes = len(set(all_labels) | set(all_preds))
        avg = "macro" if num_classes > 2 else "binary"
        acc = accuracy_score(all_labels, all_preds)
        prec = precision_score(all_labels, all_preds, average=avg, zero_division=0)
        rec = recall_score(all_labels, all_preds, average=avg, zero_division=0)
        f1 = fbeta_score(all_labels, all_preds, beta=1, average=avg, zero_division=0)
        f2 = fbeta_score(all_labels, all_preds, beta=2, average=avg, zero_division=0)

        pl_module.log("train_loss", avg_loss, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        pl_module.log("train_acc", acc, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        pl_module.log("train_precision", prec, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        pl_module.log("train_recall", rec, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        pl_module.log("train_f1", f1, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        pl_module.log("train_f2", f2, on_step=False, on_epoch=True, prog_bar=False, logger=True)

        pl_module.train()


# ---------------- excel export ----------------
class MasterValidationMetricsCallback(pl.Callback):
    """
    Write per-epoch metrics to <BASE_SAVE_DIR>/eval/all_eval_metrics_<head_tag>.xlsx:
      - logs (per-epoch, per-fold)
      - per_fold_best (best epoch by val_mcc)
      - summary (mean ± std and 95% CI across folds, using per-fold best)
      - epoch_mean (mean/std across folds for each epoch, for learning-curve plots)
    """

    def __init__(self, fold_number: int = 0, head_tag: str = "cnn"):
        super().__init__()
        eval_root = os.path.join(config.BASE_SAVE_DIR, "eval")
        os.makedirs(eval_root, exist_ok=True)

        # --- CHANGED: filename includes head_tag only (no stage) ---
        safe_head = str(head_tag).strip().lower() if head_tag else "cnn"
        self.excel_path = os.path.join(eval_root, f"all_eval_metrics_{safe_head}.xlsx")

        self.fold_number = fold_number
        self.head_tag = safe_head
        self.rows = []

    def on_validation_end(self, trainer, pl_module) -> None:  # noqa: ANN001
        cm = trainer.callback_metrics

        def _tf(name: str):
            v = cm.get(name)
            try:
                return float(v.item()) if hasattr(v, "item") else float(v)
            except Exception:
                return None

        epoch = int(trainer.current_epoch + 1)
        row = {"fold": self.fold_number, "epoch": epoch, "head_type": self.head_tag}

        for prefix in ["train", "val"]:
            for m in ["loss", "acc", "precision", "recall", "f1", "f2", "mcc"]:
                key = f"{prefix}_{m}"
                val = _tf(key)
                if val is not None:
                    row[key] = val

        self.rows.append(row)

    def on_train_end(self, trainer, pl_module) -> None:  # noqa: ANN001
        run_df = pd.DataFrame(self.rows)

        # merge with existing if present
        if os.path.exists(self.excel_path):
            try:
                old = pd.read_excel(self.excel_path, sheet_name=None)
                logs_df = pd.concat([old.get("logs", pd.DataFrame()), run_df], ignore_index=True)
            except Exception:
                logs_df = run_df.copy()
        else:
            logs_df = run_df.copy()

        # ensure columns exist (keep your originals + a few common ones)
        must_cols = [
            "fold", "epoch", "head_type",
            "val_mcc", "val_f2", "val_recall", "val_precision", "val_loss", "val_acc",
            "train_mcc", "train_f2", "train_loss",
        ]
        for c in must_cols:
            if c not in logs_df.columns:
                logs_df[c] = np.nan

        # --- per-fold best by val_mcc ---
        tmp = logs_df.dropna(subset=["val_mcc"]).copy()
        if "fold" not in tmp.columns:
            tmp["fold"] = 0
        tmp = tmp.sort_values(["fold", "val_mcc", "epoch"], ascending=[True, False, False])
        per_fold_best = tmp.groupby("fold", as_index=False).first().sort_values("fold")

        # --- summary with t-based 95% CI (fallback z=1.96) ---
        def _tcrit(df_deg: int) -> float:
            try:
                from scipy import stats as _st  # optional
                return float(_st.t.ppf(0.975, df=df_deg)) if df_deg > 0 else np.nan
            except Exception:
                return 1.96 if df_deg > 0 else np.nan

        k = int(per_fold_best["fold"].nunique()) if len(per_fold_best) else 0
        tcrit = _tcrit(k - 1)

        metrics = ["val_mcc", "val_f2", "val_recall", "val_precision", "val_loss", "val_acc"]
        summary_rows = []
        for m in metrics:
            vals = per_fold_best[m].astype(float).to_numpy() if m in per_fold_best.columns else np.array([])
            if len(vals) == 0:
                mean = std = ci_l = ci_u = np.nan
            else:
                mean = float(np.mean(vals))
                std = float(np.std(vals, ddof=1)) if k > 1 else 0.0
                se = (std / np.sqrt(k)) if k > 1 else np.nan
                ci_l = (mean - tcrit * se) if k > 1 else np.nan
                ci_u = (mean + tcrit * se) if k > 1 else np.nan

            summary_rows.append(
                {"metric": m, "k_folds": k, "mean": mean, "std": std, "95%_CI_lower": ci_l, "95%_CI_upper": ci_u}
            )
        summary_df = pd.DataFrame(summary_rows)

        # --- NEW: epoch-wise mean across folds for learning curves ---
        # Uses available folds per epoch (no forward-fill).
        metric_cols = [c for c in logs_df.columns if c.startswith("train_") or c.startswith("val_")]
        metric_cols = [c for c in metric_cols if c not in ["train_epoch", "val_epoch"]]

        g = logs_df.groupby("epoch", as_index=False)

        epoch_mean = g[metric_cols].mean(numeric_only=True)
        epoch_std = g[metric_cols].std(numeric_only=True, ddof=1)

        # number of folds contributing per epoch (based on val_f2 presence if exists else any metric)
        if "val_mcc" in logs_df.columns:
            n_per_epoch = logs_df.groupby("epoch")["val_mcc"].apply(lambda s: int(s.notna().sum())).reset_index(name="n_folds")
        else:
            n_per_epoch = logs_df.groupby("epoch").size().reset_index(name="n_folds")

        # flatten mean/std into one table: <metric>_mean, <metric>_std
        epoch_df = epoch_mean.copy()
        for c in metric_cols:
            epoch_df[f"{c}_std"] = epoch_std[c].to_numpy()
        epoch_df = epoch_df.merge(n_per_epoch, on="epoch", how="left")
        epoch_df.insert(1, "head_type", self.head_tag)

        # atomic write
        tmp_path = self.excel_path + ".writing.xlsx"
        try:
            with pd.ExcelWriter(tmp_path, engine="openpyxl", mode="w") as w:
                logs_df.to_excel(w, sheet_name="logs", index=False)
                per_fold_best.to_excel(w, sheet_name="per_fold_best", index=False)
                summary_df.to_excel(w, sheet_name="summary", index=False)
                epoch_df.to_excel(w, sheet_name="epoch_mean", index=False)  # NEW
            os.replace(tmp_path, self.excel_path)
            print(f"[MasterValidationMetricsCallback] wrote → {self.excel_path}")
        finally:
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass

# ---------------- optuna reporting ----------------
class OptunaCompositeReportingCallback(pl.Callback):
    """Report a chosen validation metric to Optuna once per epoch and allow pruning."""

    def __init__(self, trial, cfg, metric_name: str = "val_f2"):
        super().__init__()
        self.trial = trial
        self.cfg = cfg
        self.metric_name = metric_name
        self._last_reported_step: Optional[int] = None

    @staticmethod
    def _as_float(x) -> Optional[float]:  # noqa: ANN001
        try:
            if isinstance(x, torch.Tensor):
                return float(x.detach().cpu().item())
        except Exception:
            pass
        try:
            return float(x)
        except Exception:
            return None

    def on_validation_end(self, trainer, pl_module) -> None:  # noqa: ANN001
        if getattr(trainer, "sanity_checking", False):
            return

        step = int(trainer.current_epoch)
        if self._last_reported_step == step:
            return

        m = trainer.callback_metrics.get(self.metric_name)
        val = self._as_float(m)
        if val is None:
            return

        self.trial.report(val, step=step)
        self._last_reported_step = step

        if self.trial.should_prune():
            from optuna.exceptions import TrialPruned
            raise TrialPruned()


# ---------------- compatibility aliases ----------------
LocalFairEvalCallback = EpochEndEvalCallback
LocalTrainEvalCallback = TrainSetEvalCallback

# ---------------- TrainingResourceLogger ----------------

import time
import torch
import pandas as pd
import os
from datetime import datetime

class TrainingResourceLogger:
    def __init__(self):
        self.start_time = None
        self.peak_gpu_mb = None

    def start(self):
        self.start_time = time.time()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def end(self):
        elapsed = time.time() - self.start_time
        if torch.cuda.is_available():
            self.peak_gpu_mb = torch.cuda.max_memory_allocated() / 1024**2
        else:
            self.peak_gpu_mb = 0.0
        return elapsed

    @staticmethod
    def count_params(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    def to_row(
        self,
        *,
        run_id: str,
        is_tune: bool,
        optuna_stage: str,
        optuna_study: str,
        trial_number: int | None,
        fold_number: int | None,
        cv_run: int | None,
        model_name: str,
        stage: str,
        head_type: str,
        num_params: int,
        epochs_run: int,
        train_time_sec: float,
        peak_gpu_mb: float,
        val_score: float | None,
    ):
        import datetime
        return {
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "run_id": run_id,
            "is_tune": int(is_tune),

            "optuna_stage": optuna_stage,
            "optuna_study": optuna_study,
            "trial_number": trial_number if trial_number is not None else "",
            "fold_number": fold_number if fold_number is not None else "",
            "cv_run": cv_run if cv_run is not None else "",

            "model_name": model_name,
            "stage": stage,
            "head_type": head_type,
            "num_params": num_params,
            "epochs_run": epochs_run,
            "train_time_sec": train_time_sec,
            "peak_gpu_mb": peak_gpu_mb,
            "val_score": val_score if val_score is not None else "",
        }

def append_resource_log(row: dict, eval_dir: str):
    os.makedirs(eval_dir, exist_ok=True)

    # 🔹 NEW: use head_type in filename
    head_type = str(row.get("head_type", "unknown")).lower()
    path = os.path.join(eval_dir, f"training_resources_{head_type}.xlsx")

    if os.path.exists(path):
        old = pd.read_excel(path)
        df = pd.concat([old, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])

    df.to_excel(path, index=False)
