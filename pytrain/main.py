# main.py
#!/usr/bin/env python
"""
Run training, testing, or tuning (Optuna Stage A) based on the configuration.

`cfg.run_mode` ∈ {"train", "test", "tune"}.

Optuna:
  Stage A → tune CNN hyperparams (lr, batch_size, label_smoothing, etc.)

Outputs are stored under:
<save_dir>/<YYYYMMDD>/<HHMMSS>_<run_mode>/{best_model, eval, multi_predictions}/
"""

from pathlib import Path
import sys as _sys
import os as _os
# Add repo root to path so `import config` always finds the root config.py
_REPO_ROOT = str(Path(__file__).resolve().parents[1])
if _REPO_ROOT not in _sys.path:
    _sys.path.insert(0, _REPO_ROOT)
# Add pytrain/ to path so `from src.*` resolves to pytrain/src/
_PYTRAIN_ROOT = str(Path(__file__).resolve().parent)
if _PYTRAIN_ROOT not in _sys.path:
    _sys.path.insert(0, _PYTRAIN_ROOT)
import os
import sys
import warnings
import optuna
import torch
from pytorch_lightning import seed_everything
import pandas as pd

# ---------------- GPU and Environment Settings ----------------
torch.set_float32_matmul_precision("high")
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

warnings.filterwarnings(
    "ignore",
    message=(r".*reported value is ignored because this `step` is already reported.*"),
    category=UserWarning,
    module="optuna.trial",
)

# -------- Project imports --------
import config as config
from config import cfg

from src.utils.data import _build_valid_transform

from src.utils.inference import (
    evaluate_model,
    predict_folders_to_combined_workbook,
    save_multi_ckpt_comparison,
    _parse_test_folds,
    _run_multi_ckpt_tests,
)

from src.model.model import build_classifier, _build_model_for_weights

from src.utils.optuna_tuner import (
    run_optuna_search,
    apply_best_params_to_cfg,
    _choose_reuse_trial,
    _apply_trial_params_to_cfg,
    record_trial_history,
    PARAM_PATHS_A,
)

from src.utils.train import (
    train_stage,
    repeated_cross_validation,
    print_trial_thai_callback,
)

from src.utils.utils import (
    set_seed,
    thai_time,
    generate_detection_csv,
    make_run_dirs,
    _resolve_pretrained_ckpt_for_training,
    _print_training_plan,
    _reuse_enabled,
    load_weights_into_model,
    assert_train_data_available,
)

from src.utils.viz import export_optuna_plots, _run_cv_plots
from src.utils.config_logger import save_config_snapshot

# ---------------- Mode Runners ----------------
def run_test_only(cfg, dirs):
    print("[Main] TEST ONLY MODE")
    if not cfg.pretrained_ckpt:
        print("Please provide cfg.pretrained_ckpt for testing.")
        sys.exit(1)

    raw_ckpts  = getattr(cfg.test, "ckpt_paths", None) or cfg.pretrained_ckpt
    ckpt_paths = [raw_ckpts] if isinstance(raw_ckpts, str) else list(raw_ckpts)
    ckpt_paths = list(dict.fromkeys(ckpt_paths))

    test_folds = _parse_test_folds(cfg)
    if test_folds:
        print(f"[Main] Test folders: {test_folds}")

    valid_tf = _build_valid_transform(cfg)
    ALL_RESULTS_FOR_COMPARE = {}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if test_folds:
        for ckpt in ckpt_paths:
            print(f"\n=== Loading weights: {ckpt} ===")

            num_classes = int(getattr(getattr(cfg, "model", {}), "num_classes", 3))
            model = _build_model_for_weights(cfg, ckpt, num_classes=num_classes)
            miss, unexp = load_weights_into_model(model, ckpt)
            if miss or unexp:
                print(f"[Warn] loaded with gaps: missing={miss} unexpected={unexp}")
            model.eval()
            model = model.to(device)

            run_tag   = Path(ckpt).parent.parent.name
            ckpt_tag  = Path(ckpt).with_suffix("").name
            ckpt_name = f"{run_tag}__{ckpt_tag}"
            out_dir   = os.path.join(dirs["multi_predictions"], ckpt_name)
            os.makedirs(out_dir, exist_ok=True)

            combined_xlsx = os.path.join(out_dir, f"predictions_{ckpt_name}_ALL_FOLDERS.xlsx")
            results_dict  = predict_folders_to_combined_workbook(
                model=model,
                folders=test_folds,
                transform=valid_tf,
                combined_xlsx_path=combined_xlsx,
                ckpt_print_prefix=f"[{ckpt_name}] ",
            )
            ALL_RESULTS_FOR_COMPARE[ckpt_name] = results_dict

    if len(ALL_RESULTS_FOR_COMPARE) > 1:
        compare_xlsx = os.path.join(dirs["multi_predictions"], "predictions_COMPARE_ALL_CKPTS.xlsx")
        save_multi_ckpt_comparison(ALL_RESULTS_FOR_COMPARE, compare_xlsx)
        print(f"[Test] Wrote cross-ckpt comparison workbook → {compare_xlsx}")

    save_config_snapshot(cfg, dirs["eval"], extra={
        "ckpt_paths_used": str(ckpt_paths),
        "test_folders":    str(test_folds),
        "monitor_metric":  str(getattr(cfg.training, "early_stop_metric", "val_mcc")),
    })

    print("[Main] TEST ONLY complete.")


def run_tune(cfg, dirs):
    print("[Main] TUNING MODE (Optuna Stage A — CNN hyperparams)")

    eval_folder = dirs["eval"]

    db_path    = os.path.join(cfg.general.save_dir, "optuna.db")
    storage    = f"sqlite:///{db_path}"
    study_name = getattr(getattr(cfg, "optuna", {}), "study_name", None)

    best_params, study = run_optuna_search(
        cfg=cfg,
        study_name=study_name,
        direction="maximize",
        storage=storage,
        load_if_exists=True,
        show_progress_bar=True,
        callbacks=[print_trial_thai_callback, record_trial_history(eval_folder)],
    )

    try:
        out_path = os.path.join(eval_folder, "optuna_trials.xlsx")
        df_all   = study.trials_dataframe()
        if os.path.exists(out_path):
            with pd.ExcelWriter(out_path, engine="openpyxl", mode="a", if_sheet_exists="replace") as w:
                df_all.to_excel(w, sheet_name="trials_dataframe", index=False)
        else:
            with pd.ExcelWriter(out_path, engine="openpyxl", mode="w") as w:
                df_all.to_excel(w, sheet_name="trials_dataframe", index=False)
    except Exception as e:
        print("[Optuna] trials_dataframe export failed:", e)

    export_optuna_plots(study, eval_folder)

    # Apply best params to cfg
    apply_best_params_to_cfg(cfg, best_params, target_nodes=PARAM_PATHS_A)

    best_ckpt = None

    if getattr(cfg.training, "finalize_after_hpo", False):
        print("[Finalize] Retraining with best CNN hyperparameters on CV...")
        num_classes = int(getattr(getattr(cfg, "model", {}), "num_classes", 3))
        detection_model, _ = repeated_cross_validation(
            cfg, cfg.data.detection_csv,
            num_classes=num_classes,
            stage_name="final_best",
            repeats=1,
        )
        best_ckpt = os.path.join(dirs["best_model"], "best_detection.ckpt")
        torch.save({"state_dict": detection_model.state_dict()}, best_ckpt)
        print(f"[Main] Saved ckpt → {best_ckpt}")
    else:
        print("[Main] Tuning finished (no finalize — set finalize_after_hpo=True to retrain).")

    save_config_snapshot(cfg, dirs["eval"], extra={
        "best_ckpt":      best_ckpt or "(not produced — set finalize_after_hpo=True to save)",
        "optuna_stage":   "A",
        "monitor_metric": str(getattr(cfg.training, "early_stop_metric", "val_mcc")),
        "best_params":    str(best_params),
    })


# ---------------- run_train ----------------
def run_train(cfg, dirs):
    print("[Main] TRAINING MODE")
    detection_csv = cfg.data.detection_csv
    num_classes   = int(getattr(getattr(cfg, "model", {}), "num_classes", 3))

    ckpt_to_use = _resolve_pretrained_ckpt_for_training(cfg)
    _print_training_plan(cfg, ckpt_to_use)

    metrics_xlsx = os.path.join(dirs["eval"], "all_eval_metrics_cnn.xlsx")

    # ---------------- REUSE path ----------------
    if _reuse_enabled(cfg):
        reuse_cfg = getattr(cfg, "reuse", None)
        _pretrained_backup = getattr(cfg, "pretrained_ckpt", None)
        cfg.pretrained_ckpt = None

        try:
            db_path = os.path.join(cfg.general.save_dir, "optuna.db")
            storage = f"sqlite:///{db_path}"
            fixed_blocks = getattr(reuse_cfg, "fixed", None) if reuse_cfg is not None else None

            def _apply_reuse_fixed_blocks(_cfg, _fixed_blocks, _storage: str) -> str:
                if not _fixed_blocks:
                    return "retrain_fixed_blocks"

                try:
                    from omegaconf import OmegaConf
                    blocks = OmegaConf.to_container(_fixed_blocks, resolve=True)
                except Exception:
                    blocks = list(_fixed_blocks) if isinstance(_fixed_blocks, (list, tuple)) else _fixed_blocks

                if isinstance(blocks, dict):
                    blocks = [blocks]
                if not isinstance(blocks, list):
                    blocks = []

                applied = []
                for i, block in enumerate(blocks):
                    if not isinstance(block, dict):
                        print(f"[Reuse] WARNING: fixed[{i}] is not dict -> skipped")
                        continue

                    src = str(block.get("from", "")).lower().strip()
                    if src != "study":
                        print(f"[Reuse] WARNING: fixed[{i}] unsupported from='{src}' -> skipped")
                        continue

                    study_name = str(block.get("study_name", "") or "").strip()
                    if not study_name:
                        print(f"[Reuse] WARNING: fixed[{i}] missing study_name -> skipped")
                        continue

                    trial_id = block.get("trial_id", -1)
                    try:
                        trial_id = int(trial_id) if trial_id is not None else None
                    except Exception:
                        trial_id = None

                    try:
                        study_obj = optuna.load_study(study_name=study_name, storage=_storage)
                    except Exception as e:
                        print(f"[Reuse] WARNING: cannot load study '{study_name}' -> {e} (skipped)")
                        continue

                    try:
                        t = _choose_reuse_trial(study_obj, trial_id)
                    except Exception as e:
                        print(f"[Reuse] WARNING: got None trial -> {e} (skipped)")
                        continue

                    print(f"[Reuse] Apply fixed[{i}] study='{study_obj.study_name}' "
                          f"trial={t.number} value={t.value} params={t.params}")
                    exclude_keys = block.get("exclude_keys", None) or block.get("exclude", None)
                    _apply_trial_params_to_cfg(_cfg, t.params, exclude_keys=exclude_keys)
                    applied.append(f"block{i}")

                tag = "retrain_fixed_" + "_".join(applied) if applied else "retrain_fixed_blocks"
                return tag

            if fixed_blocks:
                stage_tag = _apply_reuse_fixed_blocks(cfg, fixed_blocks, storage)
            else:
                study_name = str(getattr(reuse_cfg, "study_name", "") or "").strip()
                if not study_name:
                    raise RuntimeError("Reuse requested but no reuse source configured.")

                try:
                    study_obj = optuna.load_study(study_name=study_name, storage=storage)
                except KeyError:
                    names = optuna.get_all_study_names(storage=storage)
                    candidates = [n for n in names if "optuna" in n]
                    if not candidates:
                        raise
                    study_obj = optuna.load_study(study_name=candidates[-1], storage=storage)
                    print(f"[Reuse] Falling back to '{study_obj.study_name}'")

                trial_id = getattr(reuse_cfg, "retrain_trial_id", None)
                try:
                    trial_id = int(trial_id) if trial_id is not None else None
                except Exception:
                    trial_id = None

                t = _choose_reuse_trial(study_obj, trial_id)
                print(f"[Reuse] Using Trial #{t.number} | value: {t.value} | params: {t.params}")
                _apply_trial_params_to_cfg(cfg, t.params)
                stage_tag = f"retrain_trial{t.number}" if (trial_id is not None and trial_id >= 0) else "retrain_best_trial"

            cfg.use_cv = bool(getattr(cfg, "use_cv", True))
            repeats    = int(getattr(cfg.training, "repeated_cv", 1))

            if cfg.use_cv:
                detection_model, _ = repeated_cross_validation(
                    cfg=cfg, csv_path=detection_csv,
                    num_classes=num_classes, stage_name=stage_tag, repeats=repeats,
                )
            else:
                detection_model, _ = train_stage(
                    cfg=cfg, csv_path=detection_csv,
                    num_classes=num_classes, stage_name=stage_tag,
                )

            best_ckpt = os.path.join(dirs["best_model"], f"best_{stage_tag}.ckpt")
            torch.save({"state_dict": detection_model.state_dict()}, best_ckpt)
            print(f"[Reuse] Saved ckpt → {best_ckpt}")

            evaluate_model(detection_model, detection_csv, cfg, stage=stage_tag)
            _run_cv_plots(metrics_xlsx, dirs["eval"])
            _run_multi_ckpt_tests(detection_model, cfg, dirs, best_ckpt)

            save_config_snapshot(cfg, dirs["eval"], extra={
                "best_ckpt":       best_ckpt,
                "reuse_stage_tag": stage_tag,
                "monitor_metric":  str(getattr(cfg.training, "early_stop_metric", "val_mcc")),
                "num_classes":     str(num_classes),
                "class_weights":   str(getattr(cfg.training, "class_weights", "N/A")),
            })
            print("[Reuse] DONE.")
            return

        finally:
            cfg.pretrained_ckpt = _pretrained_backup

    # ---------------- Normal training path ----------------
    if bool(getattr(cfg, "use_cv", True)):
        detection_model, _ = repeated_cross_validation(
            cfg=cfg, csv_path=detection_csv,
            num_classes=num_classes, stage_name="detection",
            repeats=getattr(cfg.training, "repeated_cv", 1),
        )
    else:
        detection_model, _ = train_stage(
            cfg=cfg, csv_path=detection_csv,
            num_classes=num_classes, stage_name="detection",
        )

    best_ckpt = os.path.join(dirs["best_model"], "best_detection.ckpt")
    torch.save({"state_dict": detection_model.state_dict()}, best_ckpt)
    print(f"[Main] Saved ckpt → {best_ckpt}")

    evaluate_model(detection_model, cfg.data.detection_csv, cfg, stage="detection_cv")
    _run_cv_plots(metrics_xlsx, dirs["eval"])
    _run_multi_ckpt_tests(detection_model, cfg, dirs, best_ckpt)

    save_config_snapshot(cfg, dirs["eval"], extra={
        "best_ckpt":      best_ckpt,
        "monitor_metric": str(getattr(cfg.training, "early_stop_metric", "val_mcc")),
        "num_classes":    str(num_classes),
        "use_cv":         str(getattr(cfg, "use_cv", True)),
        "num_folds":      str(getattr(cfg.training, "num_folds", "N/A")),
        "class_weights":  str(getattr(cfg.training, "class_weights", "N/A")),
    })


# ---------------- main() ----------------
def main():
    set_seed(cfg.training.seed)
    seed_everything(cfg.training.seed, workers=True)

    run_mode = str(getattr(cfg, "run_mode", "train")).lower()
    if run_mode not in {"train", "test", "tune"}:
        print(f"[Main] Unknown run_mode='{run_mode}', defaulting to 'train'")
        run_mode = "train"
    cfg.run_mode = run_mode

    now      = thai_time()
    date_str = now.strftime("%Y%m%d")
    time_mode = f"{now.strftime('%H%M%S')}_{run_mode}_cnn"

    config.BASE_SAVE_DIR = os.path.join(cfg.general.save_dir, date_str, time_mode)
    dirs = make_run_dirs(config.BASE_SAVE_DIR)

    # ---------------- TRAIN or TUNE pre-checks ----------------
    if run_mode in {"train", "tune"}:
        assert_train_data_available(cfg)

        csv_path     = cfg.data.detection_csv
        want_autogen = bool(getattr(cfg.data, "generate_csv", False))

        if want_autogen:
            os.makedirs(os.path.dirname(csv_path), exist_ok=True)
            generate_detection_csv(
                cfg.data.negative_dir,
                cfg.data.positive_dir,
                csv_path,
                mixed_dir=getattr(cfg.data, "mixed_dir", None),
            )
        else:
            if not os.path.exists(csv_path):
                raise FileNotFoundError(
                    f"[Main] detection_csv not found: {csv_path}\n"
                    "Set cfg.data.generate_csv=True to build it automatically."
                )

    # ---------------- RUN MODE SWITCH ----------------
    if run_mode == "test":
        run_test_only(cfg, dirs)
    elif run_mode == "tune":
        run_tune(cfg, dirs)
    else:
        run_train(cfg, dirs)

    print("[Main] ALL DONE.")


if __name__ == "__main__":
    main()
