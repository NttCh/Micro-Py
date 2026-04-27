# Gram-Stain Classification Pipeline

CNN-based gram-stain classification with PyReason post-processing.

## Repository structure

```
├── config.py                 ← SINGLE config for the whole pipeline (edit this)
│
├── run_test_grid.py          ← Run PyReason on grid-eligible test patches
├── visualize_slides.py       ← Draw before/after slide grids
│
├── src/                      ← PyReason components
│   ├── ml_inference.py       ← Load images, run CNN, return raw predictions
│   ├── model_loader.py       ← Build backbone + load Lightning checkpoint
│   ├── pyreason_engine.py    ← Rule engine (Rules 1a/1b, 2, 3, 4)
│   └── evaluate.py           ← Metrics + Excel export
│
└── pytrain/                  ← CNN training pipeline
    ├── main.py               ← Entry point  (run_mode: train | tune | test)
    └── src/
        ├── model/model.py    ← LightningModule + build_classifier
        └── utils/
            ├── train.py
            ├── callbacks.py
            ├── data.py
            ├── utils.py
            ├── inference.py
            ├── viz.py
            ├── config_logger.py
            └── optuna_tuner.py
```

## Quickstart

### 1. Install dependencies

```bash
pip install torch torchvision pytorch-lightning albumentations \
            optuna omegaconf scikit-learn openpyxl pandas matplotlib
```

### 2. Set your data paths

Edit **`config.py`** — only Section 1 (Paths) needs to change:

```python
TRAIN_DATA_ROOT = "/path/to/data/train"   # subfolders: neg/ pos/ mixed/
TEST_FOLDER     = "/path/to/data/test"
TRAIN_OUTPUT_DIR = "/path/to/outputs/pytrain"
OUTPUT_DIR       = "/path/to/outputs/pyreason"
CKPT_PATH        = ""  # leave "" to auto-find after training
```

Or set environment variables:
```bash
export MICROCLF_DATA_ROOT=/path/to/data/train
export MICROCLF_TEST_DIR=/path/to/data/test
export MICROCLF_SAVE_DIR=/path/to/outputs/pytrain
```

### 3. Train the CNN

```bash
python pytrain/main.py
```

Set `run_mode` in `config.py`:
- `"train"` — standard training (with or without CV)
- `"tune"`  — Optuna hyperparameter search (Stage A)
- `"test"`  — evaluate a saved checkpoint

### 4. Run PyReason on test data

```bash
python run_test_grid.py
```

Set `CKPT_PATH` in `config.py` to the best checkpoint from training,
or leave it empty to auto-find the newest checkpoint in `TRAIN_OUTPUT_DIR`.

### 5. Visualise results

```bash
python visualize_slides.py
```

Reads `patch_results.csv` from the PyReason output directory and saves
before/after slide grid images under `outputs/pyreason/grid_only/slide_grids/`.

## Data layout

```
data/
  train/
    neg/      ← gram-negative patches   (label 0 = G)
    pos/      ← gram-positive patches   (label 1 = Gplus)
    mixed/    ← uncertain patches       (label 2 = Mixed)
  test/
    neg/      ← (optional, needed for metrics)
    pos/
    mixed/    ← (optional)
```

Patch filenames must follow the pattern `{slide_id}_{row}_{col}.jpg`
(e.g. `000299_1_4.jpg`) for PyReason grid rules to fire.
Individual whole-slide images (`000299.jpg`) are also supported —
PyReason will skip them for grid rules but still run the ML prediction.

## Configuration reference

All settings live in `config.py`. Key sections:

| Section | What it controls |
|---------|-----------------|
| 1. Paths | Data, output, checkpoint locations |
| 2. Model | Backbone, ImageNet weights, image size |
| 3. Training | Epochs, batch size, early stopping |
| 4. Optimizer | AdamW / Adam / SGD, LR, weight decay |
| 5. Scheduler | ReduceLROnPlateau / Cosine / Step |
| 6. Augmentation | Flip, rotate, brightness/contrast, hue |
| 7. Cross-validation | K-fold vs simple split |
| 8. Label mapping | Folder names → class indices |
| 9. PyReason thresholds | Confidence tiers, rule parameters |
| 10. Pytrain OmegaConf | Auto-built from sections 1–8 |
