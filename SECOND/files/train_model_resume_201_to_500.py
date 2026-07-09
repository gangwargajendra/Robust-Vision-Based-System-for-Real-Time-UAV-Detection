"""
Resume YOLO11 training from existing checkpoint and continue up to epoch 500.
- Resumes from FIRST/files/last.pt (does not restart)
- Appends epoch-wise metrics to existing FIRST/results/epoch_progress_*.txt
- Appends training logs to existing FIRST/results/training_*.txt
- Keeps early stopping enabled via patience
"""

from __future__ import annotations

import csv
import logging
import os
import sys
import traceback
from pathlib import Path

import torch
from ultralytics import YOLO


class ResumeConfig:
    MODEL_NAME = "yolo11n"
    RESUME_START_EPOCH = 201
    TARGET_EPOCHS = 500
    BATCH_SIZE = 2
    IMG_SIZE = 320
    PATIENCE = 30
    CONFIDENCE_THRESHOLD = 0.25
    DEVICE = int(os.getenv("YOLO_DEVICE", "0"))
    NUM_WORKERS = 1

    WORK_DIR = Path(__file__).parent.resolve()
    ROOT_DIR = WORK_DIR.parent.parent.resolve()

    FIRST_FILES_DIR = ROOT_DIR / "FIRST" / "files"
    SECOND_RESULTS_DIR = ROOT_DIR / "second" / "results"

    EPOCH_LOG = SECOND_RESULTS_DIR / "epoch_progress_201_to_500.txt"
    TRAINING_LOG = SECOND_RESULTS_DIR / "training_resume_201_to_500.txt"


def _first_existing(candidates: list[Path]) -> Path:
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def _parse_last_logged_epoch(epoch_log_path: Path) -> int:
    if not epoch_log_path.exists():
        return 0

    last_epoch = 0
    with epoch_log_path.open("r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            if row[0].strip() == "epoch_step":
                continue
            if len(row) < 2:
                continue
            try:
                epoch_value = int(float(row[1]))
                if epoch_value > last_epoch:
                    last_epoch = epoch_value
            except Exception:
                continue
    return last_epoch


def _setup_logger(training_log_file: Path) -> logging.Logger:
    logger = logging.getLogger("YOLO_Resume_Training")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if logger.handlers:
        for handler in list(logger.handlers):
            logger.removeHandler(handler)

    file_handler = logging.FileHandler(training_log_file, encoding="utf-8", mode="a")
    console_handler = logging.StreamHandler(sys.stdout)

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger


def _build_epoch_logger_callback(
    epoch_log_file: Path,
    logger: logging.Logger,
    effective_last_epoch: int,
):
    header = (
        "epoch_step,epoch,total_epochs,box_loss,cls_loss,dfl_loss,"
        "precision,recall,map50,map50_95\n"
    )

    if (not epoch_log_file.exists()) or epoch_log_file.stat().st_size == 0:
        with epoch_log_file.open("w", encoding="utf-8") as f:
            f.write(header)

    def on_fit_epoch_end(trainer):
        epoch = int(getattr(trainer, "epoch", 0)) + 1
        total_epochs = int(getattr(trainer, "epochs", ResumeConfig.TARGET_EPOCHS))

        # Skip boundary duplicates and anything before the intended resume window.
        if epoch <= effective_last_epoch or epoch < ResumeConfig.RESUME_START_EPOCH:
            return

        box_loss = cls_loss = dfl_loss = 0.0
        tloss = getattr(trainer, "tloss", None)
        if tloss is not None:
            try:
                losses = tloss.detach().cpu().tolist() if hasattr(tloss, "detach") else list(tloss)
                if len(losses) >= 3:
                    box_loss, cls_loss, dfl_loss = [float(x) for x in losses[:3]]
            except Exception:
                pass

        metrics = getattr(trainer, "metrics", {}) or {}
        precision = float(metrics.get("metrics/precision(B)", 0.0) or 0.0)
        recall = float(metrics.get("metrics/recall(B)", 0.0) or 0.0)
        map50 = float(metrics.get("metrics/mAP50(B)", 0.0) or 0.0)
        map50_95 = float(metrics.get("metrics/mAP50-95(B)", 0.0) or 0.0)

        line = (
            f"{epoch}/{ResumeConfig.TARGET_EPOCHS},{epoch},{ResumeConfig.TARGET_EPOCHS},"
            f"{box_loss:.6f},{cls_loss:.6f},{dfl_loss:.6f},"
            f"{precision:.6f},{recall:.6f},{map50:.6f},{map50_95:.6f}\n"
        )
        with epoch_log_file.open("a", encoding="utf-8") as f:
            f.write(line)

        logger.info(
            "Epoch %s/%s | precision=%.4f recall=%.4f mAP50=%.4f mAP50-95=%.4f",
            epoch,
            ResumeConfig.TARGET_EPOCHS,
            precision,
            recall,
            map50,
            map50_95,
        )

    return on_fit_epoch_end


def _check_environment(logger: logging.Logger) -> None:
    logger.info("=" * 70)
    logger.info("ENVIRONMENT CHECK (RESUME)")
    logger.info("=" * 70)
    logger.info("PyTorch Version: %s", torch.__version__)
    logger.info("CUDA Available: %s", torch.cuda.is_available())

    if torch.cuda.is_available():
        logger.info("GPU Count: %s", torch.cuda.device_count())
        for i in range(torch.cuda.device_count()):
            name = torch.cuda.get_device_name(i)
            mem_gb = torch.cuda.get_device_properties(i).total_memory / 1e9
            logger.info("  GPU %s: %s", i, name)
            logger.info("    Memory: %.2f GB", mem_gb)
        logger.info("Configured device index: %s", ResumeConfig.DEVICE)


def _validate_paths(
    logger: logging.Logger,
    epoch_log_file: Path,
    training_log_file: Path,
    data_yaml_path: Path,
    resume_last_pt: Path,
) -> bool:
    required_paths = [
        data_yaml_path,
        resume_last_pt,
        ResumeConfig.SECOND_RESULTS_DIR,
    ]

    ok = True
    for path in required_paths:
        if not path.exists():
            logger.error("Missing required path: %s", path)
            ok = False

    if not epoch_log_file.exists():
        logger.warning("Epoch log file not found. It will be created: %s", epoch_log_file)
    else:
        logger.info("Using existing epoch log file: %s", epoch_log_file)

    logger.info("Using training log file: %s", training_log_file)
    return ok


def _evaluate_best_model(logger: logging.Logger, best_model_path: Path, data_yaml_path: Path) -> None:
    if not best_model_path.exists():
        logger.warning("Best model not found for evaluation: %s", best_model_path)
        return

    try:
        model = YOLO(str(best_model_path))
        metrics = model.val(
            data=str(data_yaml_path),
            imgsz=ResumeConfig.IMG_SIZE,
            conf=ResumeConfig.CONFIDENCE_THRESHOLD,
            verbose=False,
        )
        logger.info("mAP50: %.4f", float(getattr(metrics.box, "map50", 0.0)))
        logger.info("mAP50-95: %.4f", float(getattr(metrics.box, "map", 0.0)))
    except Exception as exc:
        logger.warning("Best-model evaluation skipped due to: %s", exc)


def _create_resume_compatible_checkpoint(
    logger: logging.Logger,
    source_ckpt: Path,
    target_epochs: int,
    resume_start_epoch: int,
    output_ckpt: Path,
) -> Path | None:
    try:
        # Torch >=2.6 defaults to weights_only=True which blocks full checkpoint dict loading.
        ckpt = torch.load(str(source_ckpt), map_location="cpu", weights_only=False)
    except TypeError:
        # Backward compatibility for torch versions without weights_only argument.
        ckpt = torch.load(str(source_ckpt), map_location="cpu")
    except Exception as exc:
        logger.error("Could not load checkpoint %s: %s", source_ckpt, exc)
        logger.error("Checkpoint load traceback:\n%s", traceback.format_exc())
        return None

    if not isinstance(ckpt, dict):
        logger.error("Unexpected checkpoint format in %s (not a dict)", source_ckpt)
        return None

    train_args = ckpt.get("train_args")
    if isinstance(train_args, dict):
        train_args["epochs"] = target_epochs
    elif train_args is not None and hasattr(train_args, "epochs"):
        setattr(train_args, "epochs", target_epochs)
    else:
        ckpt["train_args"] = {"epochs": target_epochs}

    args_obj = ckpt.get("args")
    if isinstance(args_obj, dict):
        args_obj["epochs"] = target_epochs
    elif args_obj is not None and hasattr(args_obj, "epochs"):
        setattr(args_obj, "epochs", target_epochs)

    # Ultralytics resume requires ckpt['epoch'] >= 0 and < target epochs.
    # Completed runs can store epoch=-1, which triggers "nothing to resume".
    forced_epoch = max(0, int(resume_start_epoch) - 2)
    ckpt_epoch = ckpt.get("epoch", None)
    if not isinstance(ckpt_epoch, int):
        try:
            ckpt_epoch = int(ckpt_epoch)
        except Exception:
            ckpt_epoch = None
    if ckpt_epoch is None or ckpt_epoch < 0 or ckpt_epoch >= target_epochs:
        ckpt["epoch"] = forced_epoch
    else:
        ckpt["epoch"] = min(ckpt_epoch, target_epochs - 1)

    try:
        torch.save(ckpt, str(output_ckpt))
    except Exception as exc:
        logger.error("Failed saving patched checkpoint %s: %s", output_ckpt, exc)
        logger.error("Checkpoint save traceback:\n%s", traceback.format_exc())
        return None

    logger.warning(
        "Created patched resume checkpoint with epochs=%s epoch=%s at %s",
        target_epochs,
        ckpt.get("epoch"),
        output_ckpt,
    )
    return output_ckpt


def main() -> int:
    ResumeConfig.SECOND_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    data_yaml_path = _first_existing(
        [
            ResumeConfig.FIRST_FILES_DIR / "data.yaml",
            ResumeConfig.ROOT_DIR / "data.yaml",
        ]
    )
    resume_last_pt = _first_existing(
        [
            ResumeConfig.FIRST_FILES_DIR / "last.pt",
            ResumeConfig.ROOT_DIR / "runs" / "detect" / "weights" / "last.pt",
            ResumeConfig.ROOT_DIR / "last.pt",
        ]
    )
    best_pt = _first_existing(
        [
            ResumeConfig.FIRST_FILES_DIR / "best.pt",
            ResumeConfig.ROOT_DIR / "runs" / "detect" / "weights" / "best.pt",
            ResumeConfig.ROOT_DIR / "best.pt",
        ]
    )

    epoch_log_file = ResumeConfig.EPOCH_LOG
    training_log_file = ResumeConfig.TRAINING_LOG

    logger = _setup_logger(training_log_file)

    logger.info("\n" + "=" * 70)
    logger.info("YOLO11 RESUME TRAINING PIPELINE")
    logger.info("Target epochs: %s", ResumeConfig.TARGET_EPOCHS)
    logger.info("Patience (early stopping): %s", ResumeConfig.PATIENCE)
    logger.info("Resume checkpoint: %s", resume_last_pt)
    logger.info("Best checkpoint (for eval): %s", best_pt)
    logger.info("Data yaml: %s", data_yaml_path)
    logger.info("Epoch log file: %s", epoch_log_file)
    logger.info("Training log file: %s", training_log_file)
    logger.info("=" * 70)

    _check_environment(logger)

    if not _validate_paths(logger, epoch_log_file, training_log_file, data_yaml_path, resume_last_pt):
        logger.error("Path validation failed. Exiting.")
        return 1

    initial_last_epoch = _parse_last_logged_epoch(epoch_log_file)
    effective_last_epoch = max(initial_last_epoch, ResumeConfig.RESUME_START_EPOCH - 1)
    logger.info("Detected last logged epoch in 201-500 file: %s", initial_last_epoch)
    logger.info("Effective resume lower bound: %s", effective_last_epoch)

    if effective_last_epoch >= ResumeConfig.TARGET_EPOCHS:
        logger.info(
            "No training required. Existing log already has epoch %s which is >= target %s.",
            effective_last_epoch,
            ResumeConfig.TARGET_EPOCHS,
        )
        return 0

    model = YOLO(str(resume_last_pt))
    model.add_callback(
        "on_fit_epoch_end",
        _build_epoch_logger_callback(epoch_log_file, logger, effective_last_epoch),
    )

    train_kwargs = dict(
        resume=True,
        epochs=ResumeConfig.TARGET_EPOCHS,
        patience=ResumeConfig.PATIENCE,
        device=ResumeConfig.DEVICE,
        workers=ResumeConfig.NUM_WORKERS,
        imgsz=ResumeConfig.IMG_SIZE,
        batch=ResumeConfig.BATCH_SIZE,
        save=True,
        verbose=False,
        plots=True,
        conf=ResumeConfig.CONFIDENCE_THRESHOLD,
        amp=True,
    )

    try:
        model.train(**train_kwargs)
    except Exception as exc:
        err_text = str(exc)
        if "nothing to resume" not in err_text.lower():
            logger.error("Resume training failed: %s", exc)
            logger.error("Resume training traceback:\n%s", traceback.format_exc())
            return 1

        logger.warning("Ultralytics reported finished checkpoint; attempting metadata-patched resume retry.")
        patched_ckpt = _create_resume_compatible_checkpoint(
            logger=logger,
            source_ckpt=resume_last_pt,
            target_epochs=ResumeConfig.TARGET_EPOCHS,
            resume_start_epoch=ResumeConfig.RESUME_START_EPOCH,
            output_ckpt=ResumeConfig.SECOND_RESULTS_DIR / "last_resume_201_to_500.pt",
        )
        if patched_ckpt is None:
            logger.error("Could not create patched checkpoint for resume retry.")
            return 1

        retry_model = YOLO(str(patched_ckpt))
        retry_model.add_callback(
            "on_fit_epoch_end",
            _build_epoch_logger_callback(epoch_log_file, logger, effective_last_epoch),
        )
        try:
            retry_model.train(**train_kwargs)
        except Exception as retry_exc:
            logger.error("Resume retry failed: %s", retry_exc)
            logger.error("Resume retry traceback:\n%s", traceback.format_exc())
            return 1

    logger.info("Resume training finished. Running evaluation on best checkpoint if available.")
    _evaluate_best_model(logger, best_pt, data_yaml_path)

    logger.info("=" * 70)
    logger.info("RESUME PIPELINE COMPLETED")
    logger.info("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
