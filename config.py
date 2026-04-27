"""
config.py
=========
Single configuration file for the entire pipeline:
  - Pytrain  : CNN training / tuning / evaluation
  - PyReason : post-processing rule engine + test evaluation

Edit the paths in Section 1 and thresholds in Section 9.
Everything else can stay as defaults until you have a reason to change it.

Sections:
  1.  Paths
  2.  Model
  3.  Training
  4.  Optimizer
  5.  Scheduler
  6.  Augmentation
  7.  Cross-validation
  8.  Label mapping
  9.  PyReason thresholds
  10. Pytrain (OmegaConf config — used by pytrain/main.py)
"""

import os
from pathlib import Path

# ─────────────────────────────────────────────────────────────────
#  1. PATHS  ← edit these
# ─────────────────────────────────────────────────────────────────

# Root of the repository (auto-detected; don't change)
REPO_ROOT = Path(__file__).resolve().parent

# Training images — subfolders = class names
#   data/train/
#     neg/    ← gram-negative  (label 0 = G)
#     pos/    ← gram-positive  (label 1 = Gplus)
#     mixed/  ← uncertain      (label 2 = Mixed)
TRAIN_DATA_ROOT  = os.getenv("MICROCLF_DATA_ROOT",
                             str(REPO_ROOT / "data" / "train"))

# Test images — same subfolder layout for metrics,
# or flat folder (no GT, PyReason still runs)
TEST_FOLDER      = os.getenv("MICROCLF_TEST_DIR",
                             str(REPO_ROOT / "data" / "test"))

# Where pytrain outputs go (checkpoints, logs, Excel)
TRAIN_OUTPUT_DIR = os.getenv("MICROCLF_SAVE_DIR",
                             str(REPO_ROOT / "outputs" / "pytrain"))

# Where pyreason test / eval results go
OUTPUT_DIR       = os.getenv("MICROCLF_OUTPUT_DIR",
                             str(REPO_ROOT / "outputs" / "pyreason"))

# Checkpoint for run_test_grid.py
# Leave "" to look for best_model.ckpt inside TRAIN_OUTPUT_DIR automatically
CKPT_PATH        = os.getenv("MICROCLF_CKPT_PATH", "")


# ─────────────────────────────────────────────────────────────────
#  2. MODEL
# ─────────────────────────────────────────────────────────────────

# Any torchvision backbone with a .fc or .classifier head:
#   torchvision.models.resnet50      (default — matches pytrain backbone)
#   torchvision.models.resnet101
#   torchvision.models.efficientnet_b0
#   torchvision.models.densenet121
BACKBONE         = "torchvision.models.resnet50"

# ImageNet weights to initialise from:
#   "IMAGENET1K_V2"  (best for ResNet50)
#   "IMAGENET1K_V1"
#   "DEFAULT"
#   None             (random init)
BACKBONE_WEIGHTS = "IMAGENET1K_V2"

# Input image size (height = width)
IMG_SIZE         = 224

# Optional warm-start checkpoint (loaded before training, not used by run_test).
# Leave "" to start from ImageNet weights only.
PRETRAINED_CKPT  = ""


# ─────────────────────────────────────────────────────────────────
#  3. TRAINING
# ─────────────────────────────────────────────────────────────────

SEED             = 42
TRAIN_EPOCHS     = 30
BATCH_SIZE       = 8
NUM_WORKERS      = 0        # 0 = safe on Windows; set >0 on Linux/Mac
VAL_SPLIT        = 0.2      # fraction used for validation (no-CV mode only)
EARLY_STOP_PAT   = 7        # stop if val_mcc doesn't improve for N epochs

# Freeze backbone for N epochs, then unfreeze everything.
# 0 = train the whole network from epoch 1.
FREEZE_BACKBONE_EPOCHS = 10


# ─────────────────────────────────────────────────────────────────
#  4. OPTIMIZER
# ─────────────────────────────────────────────────────────────────

OPTIMIZER        = "AdamW"   # "AdamW" | "Adam" | "SGD"

LR               = 1e-4
WEIGHT_DECAY     = 1e-3

# SGD only — ignored for Adam/AdamW
SGD_MOMENTUM     = 0.9
SGD_NESTEROV     = True

# Gradient clipping (0 = disabled)
GRAD_CLIP        = 1.0


# ─────────────────────────────────────────────────────────────────
#  5. SCHEDULER
# ─────────────────────────────────────────────────────────────────

SCHEDULER          = "ReduceLROnPlateau"  # "ReduceLROnPlateau" | "CosineAnnealingLR" | "StepLR" | "none"

SCHEDULER_FACTOR   = 0.3   # ReduceLROnPlateau: multiply LR on plateau
SCHEDULER_PATIENCE = 3     # ReduceLROnPlateau: epochs before reducing

COSINE_T_MAX       = TRAIN_EPOCHS
COSINE_ETA_MIN     = 1e-6

STEP_SIZE          = 10
STEP_GAMMA         = 0.1


# ─────────────────────────────────────────────────────────────────
#  6. AUGMENTATION FLAGS
# ─────────────────────────────────────────────────────────────────

AUG_HFLIP          = True;  AUG_HFLIP_P          = 0.5
AUG_VFLIP          = True;  AUG_VFLIP_P          = 0.5
AUG_ROTATE         = True;  AUG_ROTATE_LIMIT      = 20;   AUG_ROTATE_P      = 0.6
AUG_BRIGHTNESS     = True;  AUG_BRIGHTNESS_LIMIT  = 0.2;  AUG_CONTRAST_LIMIT= 0.2; AUG_BRIGHTNESS_P  = 0.5
AUG_HUE_SAT        = True;  AUG_HUE_SHIFT         = 10;   AUG_SAT_SHIFT     = 20;  AUG_VAL_SHIFT     = 10; AUG_HUE_P = 0.3


# ─────────────────────────────────────────────────────────────────
#  7. CROSS-VALIDATION
# ─────────────────────────────────────────────────────────────────

USE_CV    = True
NUM_FOLDS = 5


# ─────────────────────────────────────────────────────────────────
#  8. LABEL MAPPING  (change only if your folder names change)
# ─────────────────────────────────────────────────────────────────

# 3-class: G (gram-negative=0), Gplus (gram-positive=1), Mixed (uncertain=2)
SUBFOLDER_TO_INT    = {"neg": 0, "pos": 1, "mixed": 2}
INT_TO_LABEL        = {0: "G",   1: "Gplus", 2: "Mixed"}
LABEL_TO_INT        = {"G": 0,   "Gplus": 1, "Mixed": 2}
LABELLED_SUBFOLDERS = ["pos", "neg", "mixed"]
SUBFOLDER_TO_LABEL  = {"pos": "Gplus", "neg": "G", "mixed": "Mixed"}

# Legacy aliases (used by evaluate.py)
LABEL_MAP     = INT_TO_LABEL
LABEL_MAP_INV = LABEL_TO_INT

# Class weights for CrossEntropyLoss.
# None = auto-compute from class frequencies in the training data (recommended).
# Override example: [1.8, 1.0, 1.2]  (index order: neg=0, pos=1, mixed=2)
MANUAL_CLASS_WEIGHTS = [1.8, 1.0, 1.2]

# Smoothing power for auto class-weight computation (used when MANUAL_CLASS_WEIGHTS is None).
#   1.0 = raw inverse frequency  (aggressive)
#   0.5 = sqrt smoothing         (gentler)
#   0.0 = uniform weights
CLASS_WEIGHT_POWER = 0.75


# ─────────────────────────────────────────────────────────────────
#  9. PYREASON THRESHOLDS
# ─────────────────────────────────────────────────────────────────

HIGH_CONF_THR        = 0.85   # >= this → HIGH confidence (firm patch for vote)
MEDIUM_CONF_THR      = 0.75   # >= this → MEDIUM

MAJORITY_RATIO_THR   = 0.70   # slide needs >= this fraction to be "dominant"
MIN_PATCHES_FOR_VOTE = 3      # slide needs >= this many firm patches for Rule 1

NEIGHBOR_AGREE_MIN   = 3      # Rule 2: neighbors needed to resolve uncertain patch
CLUSTER_MIN          = 3      # Rule 3: min adjacent minority patches = cluster flag

# Rule 1c/1d — Mixed patch resolution threshold
# Mixed patches in a dominant slide are resolved to the majority class ONLY when
# the slide ratio >= this. Set higher than MAJORITY_RATIO_THR to be more
# conservative about overriding Mixed predictions.
MIXED_RESOLVE_RATIO_THR = 0.85

# Rule 1c/1d — Maximum allowed Mixed fraction on a slide before Rule 1c is blocked.
# If more than this fraction of a slide predicts Mixed, treat as genuinely heterogeneous.
MAX_MIXED_FRACTION_FOR_RESOLVE = 0.20

# Rule 1f — Section (quadrant/window) vote parameters
N_SECTIONS                  = 4      # 2×2 quadrant grid  (or 9 for 3×3)
SECTION_MIN_PATCHES         = 2      # min firm patches in a section to be trusted
SECTION_MAJORITY_RATIO_THR  = 0.75   # section needs >= 75% agreement to be dominant
SECTION_MAX_MIXED_FRACTION  = 0.40   # if > 40% Mixed in section → skip

# Stricter thresholds for minority firm patch flipping (Rule 1b)
SECTION_MINORITY_MIN_PATCHES  = 3
SECTION_MINORITY_RATIO_THR    = 0.85

# Rule 0 (Anchor) + Rule 5 (Stronghold)
# ANCHOR_CONF_THR: patches with confidence >= this are immutable in Rule 1.
ANCHOR_CONF_THR  = 0.85   # same as HIGH_CONF_THR by default

# STRONGHOLD_MIN: patch with >= this many HIGH-conf same-class neighbours
# is a "stronghold" and cannot be flipped by Rule 1, even if LOW/MEDIUM conf.
STRONGHOLD_MIN   = 2

# Sweep compatibility
NEIGHBOUR_VOTE_MIN      = 2
NEIGHBOUR_VOTE_MAJORITY = 0.6

SECTION_MODE = "window"   # "quadrant" | "window"
WINDOW_ROWS  = 3          # height of sliding window
WINDOW_COLS  = 5          # width of sliding window


# ─────────────────────────────────────────────────────────────────
#  10. PYTRAIN — OmegaConf config (used by pytrain/main.py)
# ─────────────────────────────────────────────────────────────────
# This block builds the OmegaConf DictConfig that pytrain/main.py uses.
# All path and model values are derived from sections 1–8 above so that
# there is ONE place to edit for the whole pipeline.

def _build_pytrain_cfg():
    """Build OmegaConf DictConfig for the pytrain pipeline."""
    from omegaconf import OmegaConf

    _train_root = TRAIN_DATA_ROOT

    cfg_dict = {
        "run_mode": "train",    # "train" | "tune" | "test"
        "use_cv": USE_CV,

        "reuse": {
            "enable": False,
            "fixed": [
                # Example: reuse best trial from a previous Optuna study
                # {"from": "study", "study_name": "bac_clinical_optuna_A_objMCC_XXXXXXXX", "trial_id": -1},
            ],
        },

        "general": {
            "save_dir":    TRAIN_OUTPUT_DIR,
            "project_name": "bac_clinical",
        },

        "trainer": {
            "devices":           1,
            "accelerator":       "auto",
            "precision":         "16-mixed",
            "gradient_clip_val": GRAD_CLIP,
        },

        "training": {
            "seed":                      SEED,
            "mode":                      "max",
            "tuning_epochs_detection":   TRAIN_EPOCHS,
            "num_folds":                 NUM_FOLDS,
            "only_fold":                 None,
            "repeated_cv":               1,
            "early_stopping": {
                "enabled":           True,
                "early_stop_metric": "val_mcc",
                "patience_hpo":      EARLY_STOP_PAT,
                "patience_final":    EARLY_STOP_PAT,
            },
            "freeze_strategy":        "initial",   # "initial" | "none"
            "freeze_backbone_epochs": FREEZE_BACKBONE_EPOCHS,
            "final_unfreeze":         True,
            "finalize_after_hpo":     True,
            "clear_cuda_before_stage": True,
            "label_smoothing":        0.05,
            "min_fold_score":         0.75,
        },

        "xai": {
            "enabled":              False,
            "target_layer":         ["layer2", "layer3", "layer4"],
            "k_patches":            3,
            "patch_size":           96,
            "save_heatmap":         True,
            "save_overlay":         True,
            "save_original_overlay": True,
            "panel":                False,
            "class_names":          ["Gminus", "Gplus", "mixed"],
            "tta":                  True,
            "smooth":               True,
            "smooth_n":             6,
            "smooth_sigma":         0.05,
            "mask_scale_bar":       False,
            "mask_rect_wh":         [0.18, 0.08],
        },

        "optimizer": {
            "class_name": f"torch.optim.{OPTIMIZER}",
            "params": {
                "lr":           LR,
                "weight_decay": WEIGHT_DECAY,
            },
        },

        "scheduler": {
            "class_name": "torch.optim.lr_scheduler.ReduceLROnPlateau",
            "step":       "epoch",
            "monitor":    "val_mcc",
            "params": {
                "mode":     "max",
                "factor":   SCHEDULER_FACTOR,
                "patience": SCHEDULER_PATIENCE,
            },
        },

        "model": {
            "backbone": {
                "class_name": BACKBONE,
                "params": {
                    "weights": BACKBONE_WEIGHTS,
                },
            },
            "num_classes": 3,   # G=0, Gplus=1, Mixed=2
        },

        "data": {
            "negative_dir":  os.path.join(_train_root, "neg"),
            "positive_dir":  os.path.join(_train_root, "pos"),
            "mixed_dir":     os.path.join(_train_root, "mixed"),
            "detection_csv": os.path.join(_train_root, "train_model.csv"),
            "folder_path":   _train_root,
            "num_workers":   NUM_WORKERS,
            "generate_csv":  True,
            "batch_size":    BATCH_SIZE,
            "label_col":     "label",
            "valid_split":   VAL_SPLIT,
            "mixed_cap_ratio": None,
        },

        "augmentation": {
            "train": {
                "augs": [
                    {"class_name": "albumentations.Resize",         "params": {"height": IMG_SIZE, "width": IMG_SIZE, "p": 1.0}},
                    {"class_name": "albumentations.Rotate",         "params": {"limit": AUG_ROTATE_LIMIT, "p": AUG_ROTATE_P}},
                    {"class_name": "albumentations.HorizontalFlip", "params": {"p": AUG_HFLIP_P}},
                    {"class_name": "albumentations.VerticalFlip",   "params": {"p": AUG_VFLIP_P}},
                    {"class_name": "albumentations.Normalize",      "params": {}},
                    {"class_name": "albumentations.pytorch.transforms.ToTensorV2", "params": {"p": 1.0}},
                ]
            },
            "valid": {
                "augs": [
                    {"class_name": "albumentations.Resize",    "params": {"height": IMG_SIZE, "width": IMG_SIZE, "p": 1.0}},
                    {"class_name": "albumentations.Normalize", "params": {}},
                    {"class_name": "albumentations.pytorch.transforms.ToTensorV2", "params": {"p": 1.0}},
                ]
            },
            "hpo_resize":   {"height": IMG_SIZE, "width": IMG_SIZE},
            "final_resize": {"height": IMG_SIZE, "width": IMG_SIZE},
        },

        "test": {
            "folder_path": [TEST_FOLDER],
        },

        # Path to a pretrained checkpoint to initialize from (optional).
        "pretrained_ckpt": PRETRAINED_CKPT,

        # ── Optuna (Stage A: CNN hyperparameter tuning only) ──────
        "optuna": {
            "stage": "A",
            "n_trials_by_stage": {"A": 15},

            "sampler": {
                "class_name": "optuna.samplers.TPESampler",
                "params": {"multivariate": True},
            },
            "pruner": {
                "class_name": "optuna.pruners.MedianPruner",
                "params": {"n_startup_trials": 5, "n_warmup_steps": 15},
            },

            "params_A": {
                "lr":                {"type": "loguniform", "low": 3e-6,  "high": 1e-4},
                "batch_size":        {"type": "categorical", "choices": [8, 16, 32]},
                "weight_decay":      {"type": "loguniform", "low": 1e-6,  "high": 1e-3},
                "gradient_clip_val": {"type": "float",      "low": 0.1,   "high": 1.0,  "step": 0.05},
                "label_smoothing":   {"type": "float",      "low": 0.0,   "high": 0.2,  "step": 0.05},
            },
        },
    }

    return OmegaConf.create(cfg_dict)


# Build the OmegaConf cfg object (imported by pytrain/main.py as `from config import cfg`)
try:
    cfg = _build_pytrain_cfg()
    BASE_SAVE_DIR = TRAIN_OUTPUT_DIR
except Exception as _e:
    # OmegaConf not installed — pyreason-only usage still works
    cfg = None
    BASE_SAVE_DIR = TRAIN_OUTPUT_DIR
