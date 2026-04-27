#!/usr/bin/env python
"""Model definitions (including the LightningModule)."""

from typing import Any

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from sklearn.metrics import accuracy_score, matthews_corrcoef, precision_recall_fscore_support

from src.utils.utils import load_obj, read_cfg_snapshot_from_weights


# ---------------- model builder ----------------
def build_classifier(cfg: Any, num_classes: int) -> nn.Module:
    """
    Build a classification model and replace the final FC layer with
    a plain nn.Linear matching num_classes.
    """
    backbone_cls = load_obj(cfg.model.backbone.class_name)
    model = backbone_cls(**cfg.model.backbone.params)

    if hasattr(model, "fc"):
        head_name = "fc"
        old_head  = model.fc
    elif hasattr(model, "classifier"):
        head_name = "classifier"
        old_head  = model.classifier
    else:
        raise ValueError(
            f"Don't know how to replace the head for {cfg.model.backbone.class_name}"
        )

    if isinstance(old_head, nn.Sequential):
        in_features = next(
            m for m in reversed(old_head) if isinstance(m, nn.Linear)
        ).in_features
    else:
        in_features = old_head.in_features

    new_head = nn.Linear(in_features, num_classes)
    setattr(model, head_name, new_head)

    print(f"[build_classifier] Linear head | in={in_features} → num_classes={num_classes}")
    return model


def _build_model_for_weights(default_cfg: Any, weights_path: str, num_classes: int = 3):
    """
    If the weights file contains a cfg snapshot, rebuild from it.
    Otherwise fall back to the current cfg.
    """
    snap = read_cfg_snapshot_from_weights(weights_path)
    if snap:
        try:
            merged = OmegaConf.merge(default_cfg, OmegaConf.create(snap))
            return build_classifier(merged, num_classes=num_classes)
        except Exception as e:
            print(f"[Load] Snapshot rebuild failed ({e}); using current cfg.")
    return build_classifier(default_cfg, num_classes=num_classes)


# ---------------- lightning module ----------------
class LitClassifier(pl.LightningModule):
    """
    LightningModule wrapping a CNN classifier.

    Supports:
      - 2-class (binary) and 3-class (Gminus / Gplus / mixed)
      - label smoothing (via cfg.training.label_smoothing or explicit arg)
      - class-weighted CrossEntropyLoss (via cfg.training.ce_class_weights)
      - freeze strategies: "none" | "feature_extractor" | "warmup_unfreeze"
    """

    def __init__(
        self,
        cfg: Any,
        model: nn.Module,
        num_classes: int,
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__()
        self.cfg        = cfg
        self.model      = model
        self.num_classes = int(num_classes)

        # label_smoothing: explicit arg wins; otherwise read from cfg
        _ls_cfg = float(getattr(getattr(cfg, "training", {}), "label_smoothing", 0.0))
        self._label_smoothing = float(label_smoothing) if label_smoothing > 0.0 else _ls_cfg

        # ── class-weighted loss ───────────────────────────────────────────────
        tr_cfg = getattr(self.cfg, "training", {})
        w_list = getattr(tr_cfg, "ce_class_weights", None)
        w = None
        if w_list is not None:
            try:
                w = torch.tensor(list(w_list), dtype=torch.float32)
            except Exception:
                w = None

        self.register_buffer(
            "_ce_weight",
            w if w is not None else torch.tensor([]),
            persistent=False,
        )
        self.criterion = nn.CrossEntropyLoss(
            weight=(self._ce_weight if self._ce_weight.numel() else None),
            label_smoothing=self._label_smoothing,
        )
        if self._label_smoothing > 0.0:
            print(
                f"[LitClassifier] label_smoothing={self._label_smoothing:.3f}"
                f"  num_classes={self.num_classes}"
            )

        # ── freeze config ─────────────────────────────────────────────────────
        self._freeze_epochs:   int  = int(getattr(tr_cfg, "freeze_backbone_epochs", 0))
        self._freeze_strategy: str  = str(getattr(tr_cfg, "freeze_strategy", "none")).lower()
        self._froze = False

        # ── epoch-level accumulators ──────────────────────────────────────────
        # Store raw preds+targets each epoch; compute all metrics in epoch_end.
        # Using lists avoids shape issues with variable batch sizes.
        self._val_preds_list:   list = []
        self._val_targets_list: list = []
        self._train_preds_list: list = []
        self._train_targets_list: list = []

    # ── helpers ───────────────────────────────────────────────────────────────
    def _get_head(self):
        return getattr(self.model, "fc", None) or getattr(self.model, "classifier", None)

    def _set_trainable(self, head_only: bool):
        head     = self._get_head()
        head_ids = {id(p) for p in head.parameters()} if head is not None else set()
        for p in self.model.parameters():
            p.requires_grad = (id(p) in head_ids) if head_only else True

    def _freeze_bn_running_stats(self):
        for m in self.model.modules():
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                m.eval()

    # ── lightning hooks ───────────────────────────────────────────────────────
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def on_train_epoch_start(self) -> None:
        # BUG FIX: Freeze logic removed from here.
        # FreezeBackboneCallback in train.py is the single authority for backbone
        # freezing/unfreezing (strategy="initial"). Having both active caused
        # them to fight each other — FreezeBackboneCallback would unfreeze at
        # epoch N, then LitClassifier would re-freeze on the next epoch start.
        # Only reset the per-epoch accumulator lists here.
        self._train_preds_list   = []
        self._train_targets_list = []

    def on_validation_epoch_start(self) -> None:
        self._val_preds_list   = []
        self._val_targets_list = []

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        images, labels = batch
        images = images.to(self.device)
        labels = labels.to(self.device)

        logits = self(images)
        loss   = self.criterion(logits, labels)
        preds  = torch.argmax(logits, dim=1)

        self._train_preds_list.extend(preds.detach().cpu().tolist())
        self._train_targets_list.extend(labels.detach().cpu().tolist())

        correct = (preds == labels).sum().float()
        total   = float(labels.numel())
        acc     = correct / total if total > 0 else torch.tensor(0.0, device=self.device)

        # Use sklearn (already imported) — same as on_validation_epoch_end
        mcc = float(matthews_corrcoef(
            labels.detach().cpu().numpy(),
            preds.detach().cpu().numpy()
        ))

        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log("train_acc",  acc,  on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log("train_mcc",  mcc,  on_step=False, on_epoch=True, prog_bar=False, logger=True)
        return loss

    def validation_step(self, batch, batch_idx: int) -> torch.Tensor:
        images, labels = batch
        images = images.to(self.device)
        labels = labels.to(self.device)

        logits = self(images)
        loss   = self.criterion(logits, labels)
        preds  = torch.argmax(logits, dim=1)

        self._val_preds_list.extend(preds.detach().cpu().tolist())
        self._val_targets_list.extend(labels.detach().cpu().tolist())

        self.log("val_loss", loss, on_step=False, on_epoch=True,
                 prog_bar=True, logger=True, sync_dist=True)
        return loss

    def on_validation_epoch_end(self) -> None:
        y_true = np.array(self._val_targets_list, dtype=np.int64)
        y_pred = np.array(self._val_preds_list,   dtype=np.int64)

        if len(y_true) == 0:
            for name in ("val_acc", "val_recall", "val_precision", "val_f1", "val_f2", "val_mcc"):
                self.log(name, 0.0, on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
            return

        device = self.device

        # ── accuracy ─────────────────────────────────────────────────────────
        acc = float(accuracy_score(y_true, y_pred))

        # ── macro precision / recall / F1 / F2 ───────────────────────────────
        # average="macro" treats every class equally — important for imbalanced 3-class
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="macro", zero_division=0
        )
        # F2 weights recall twice as much as precision
        _, _, f2, _ = precision_recall_fscore_support(
            y_true, y_pred, average="macro", zero_division=0, beta=2.0
        )

        # ── MCC ───────────────────────────────────────────────────────────────
        mcc = float(matthews_corrcoef(y_true, y_pred))

        self.log("val_acc",       acc,        on_step=False, on_epoch=True, prog_bar=True,  logger=True, sync_dist=True)
        self.log("val_recall",    float(rec),  on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        self.log("val_precision", float(prec), on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        self.log("val_f1",        float(f1),   on_step=False, on_epoch=True, prog_bar=True,  logger=True, sync_dist=True)
        self.log("val_f2",        float(f2),   on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        self.log("val_mcc",       mcc,         on_step=False, on_epoch=True, prog_bar=True,  logger=True, sync_dist=True)

    def configure_optimizers(self):
        optimizer_cls    = load_obj(self.cfg.optimizer.class_name)
        optimizer_params = dict(getattr(self.cfg, "optimizer").params)
        optimizer_params.pop("gradient_clip_val", None)
        optimizer_params.pop("dropout", None)

        optimizer = optimizer_cls(self.model.parameters(), **optimizer_params)

        scheduler_cls    = load_obj(self.cfg.scheduler.class_name)
        scheduler_params = dict(getattr(self.cfg, "scheduler").params)

        monitor  = getattr(self.cfg.scheduler, "monitor", "val_mcc")
        maximize = monitor in {
            "val_f2", "val_f1", "val_mcc", "val_recall", "val_precision", "val_acc", "val_auc"
        }

        if scheduler_cls is torch.optim.lr_scheduler.ReduceLROnPlateau:
            scheduler_params["mode"] = "max" if maximize else "min"

        scheduler = scheduler_cls(optimizer, **scheduler_params)

        return [optimizer], [{
            "scheduler": scheduler,
            "interval":  self.cfg.scheduler.step,
            "monitor":   monitor,
        }]