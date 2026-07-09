"""
YOLO11 training script for shared GPU servers.
- Stable defaults for crowded GPUs
- Clean per-epoch logging to a dedicated text file
- Resume support from last checkpoint
"""

from __future__ import annotations

import json
import logging
import os
import sys
import warnings
from datetime import datetime
from pathlib import Path

import torch
from ultralytics import YOLO

warnings.filterwarnings("ignore")


class TrainingConfig:
    """Configuration for YOLO11 training."""

    MODEL_NAME = "yolo11n"
    EPOCHS = 200
    BATCH_SIZE = 2  # Lowered strictly to 2 to prevent CUBLAS_STATUS_ALLOC_FAILED
    IMG_SIZE = 320  # Lowered image size to help with stringent memory constraints
    PATIENCE = 30
    CONFIDENCE_THRESHOLD = 0.25
    DEVICE = int(os.getenv("YOLO_DEVICE", "0"))
    NUM_WORKERS = 1 # Lowered from 2 to 1 to reduce memory usage during dataloading

    DATA_YAML = "data.yaml"

    WORK_DIR = Path(__file__).parent.resolve()
    RUNS_DIR = WORK_DIR / "runs"
    LOGS_DIR = WORK_DIR / "logs"
    CHECKPOINTS_DIR = WORK_DIR / "checkpoints"

    TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
    TRAINING_LOG = LOGS_DIR / f"training_{TIMESTAMP}.txt"
    EPOCH_LOG = LOGS_DIR / f"epoch_progress_{TIMESTAMP}.txt"
    PARAMS_FILE = CHECKPOINTS_DIR / "best_params.json"

    @classmethod
    def create_directories(cls) -> None:
        cls.RUNS_DIR.mkdir(exist_ok=True)
        cls.LOGS_DIR.mkdir(exist_ok=True)
        cls.CHECKPOINTS_DIR.mkdir(exist_ok=True)


class CheckpointManager:
    """Saves best-metric summary for quick resume/inspection."""

    def __init__(self, params_file: Path):
        self.params_file = params_file

    def save_params(self, params: dict) -> None:
        with self.params_file.open("w", encoding="utf-8") as f:
            json.dump(params, f, indent=2)


def setup_logger(log_file: Path) -> logging.Logger:
    logger = logging.getLogger("YOLO_Training")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    # Avoid duplicate handlers if script is re-imported in a session.
    if logger.handlers:
        for handler in list(logger.handlers):
            logger.removeHandler(handler)

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
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


def check_environment(logger: logging.Logger) -> None:
    logger.info("=" * 70)
    logger.info("ENVIRONMENT CHECK")
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
        logger.info("Configured device index: %s", TrainingConfig.DEVICE)
    else:
        logger.warning("GPU not available. Training will be slow on CPU.")

    logger.info("=" * 70)


def validate_data(logger: logging.Logger) -> bool:
    logger.info("\n" + "=" * 70)
    logger.info("DATASET VALIDATION")
    logger.info("=" * 70)

    data_yaml_path = TrainingConfig.WORK_DIR / TrainingConfig.DATA_YAML
    if not data_yaml_path.exists():
        logger.error("data.yaml not found at %s", data_yaml_path)
        return False

    logger.info("data.yaml found at %s", data_yaml_path)

    import yaml

    with data_yaml_path.open("r", encoding="utf-8") as f:
        data_cfg = yaml.safe_load(f)

    for key in ["path", "train", "val"]:
        if key not in data_cfg:
            logger.error("Missing key in data.yaml: %s", key)
            return False

    data_path = Path(data_cfg["path"])
    train_path = data_path / data_cfg["train"]
    val_path = data_path / data_cfg["val"]
    nc = int(data_cfg.get("nc", 0) or 0)

    logger.info("Configured dataset path: %s", data_path)

    if not train_path.exists():
        logger.error("Training data path not found: %s", train_path)
        return False
    if not val_path.exists():
        logger.error("Validation data path not found: %s", val_path)
        return False

    logger.info("Training path found: %s", train_path)
    logger.info("Validation path found: %s", val_path)

    # Early class-id validation avoids confusing runtime warnings later.
    if nc > 0:
        label_roots = [data_path / "labels" / "train", data_path / "labels" / "val"]
        found_max_class = -1
        checked_files = 0
        for root in label_roots:
            if not root.exists():
                continue
            for txt_file in root.glob("*.txt"):
                checked_files += 1
                try:
                    with txt_file.open("r", encoding="utf-8") as f:
                        for line in f:
                            line = line.strip()
                            if not line:
                                continue
                            cls_id = int(float(line.split()[0]))
                            if cls_id > found_max_class:
                                found_max_class = cls_id
                except Exception:
                    continue

        if found_max_class >= nc:
            logger.error(
                "Label class mismatch: max class id %s found but nc=%s (valid ids: 0..%s)",
                found_max_class,
                nc,
                nc - 1,
            )
            logger.error("Please update data.yaml nc/names to match dataset labels.")
            return False
        logger.info("Class-id check passed across %s label files (max class id: %s)", checked_files, found_max_class)

    logger.info("Dataset validation passed")
    logger.info("=" * 70)
    return True


def _latest_results_csv() -> Path | None:
    candidates = sorted(
        TrainingConfig.RUNS_DIR.glob("detect*/results.csv"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def _save_best_params_from_csv(logger: logging.Logger, checkpoint_manager: CheckpointManager) -> None:
    results_csv = _latest_results_csv()
    if not results_csv:
        logger.warning("Could not find results.csv to extract best metrics")
        return

    try:
        import csv

        rows = []
        with results_csv.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        if not rows:
            logger.warning("results.csv is empty: %s", results_csv)
            return

        best_row = max(rows, key=lambda r: float(r.get("metrics/mAP50-95(B)", 0.0) or 0.0))
        best_epoch = int(float(best_row.get("epoch", 0))) + 1
        best_map = float(best_row.get("metrics/mAP50-95(B)", 0.0) or 0.0)

        params = {
            "best_epoch": best_epoch,
            "best_mAP50_95": best_map,
            "results_csv": str(results_csv),
            "best_model": str((results_csv.parent / "weights" / "best.pt").resolve()),
            "last_model": str((results_csv.parent / "weights" / "last.pt").resolve()),
            "updated_at": datetime.now().isoformat(timespec="seconds"),
        }
        checkpoint_manager.save_params(params)
        logger.info("Best parameters saved to %s", TrainingConfig.PARAMS_FILE)
    except Exception as exc:
        logger.warning("Failed to parse/save best params from results.csv: %s", exc)


def _build_epoch_logger_callback(epoch_log_file: Path, logger: logging.Logger):
    header = (
        "epoch_step,epoch,total_epochs,box_loss,cls_loss,dfl_loss,"
        "precision,recall,map50,map50_95\n"
    )
    with epoch_log_file.open("w", encoding="utf-8") as f:
        f.write(header)

    def on_fit_epoch_end(trainer):
        epoch = int(getattr(trainer, "epoch", 0)) + 1
        total_epochs = int(getattr(trainer, "epochs", TrainingConfig.EPOCHS))

        # Train losses from trainer.tloss tensor: box, cls, dfl
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
            f"{epoch}/{total_epochs},{epoch},{total_epochs},{box_loss:.6f},{cls_loss:.6f},{dfl_loss:.6f},"
            f"{precision:.6f},{recall:.6f},{map50:.6f},{map50_95:.6f}\n"
        )
        with epoch_log_file.open("a", encoding="utf-8") as f:
            f.write(line)

        logger.info(
            "Epoch %s/%s | precision=%.4f recall=%.4f mAP50=%.4f mAP50-95=%.4f",
            epoch,
            total_epochs,
            precision,
            recall,
            map50,
            map50_95,
        )

    return on_fit_epoch_end


def train_model(logger: logging.Logger, checkpoint_manager: CheckpointManager) -> bool:
    logger.info("\n" + "=" * 70)
    logger.info("STARTING YOLO11 TRAINING")
    logger.info("=" * 70)

    logger.info("Model: %s", TrainingConfig.MODEL_NAME)
    logger.info("Epochs: %s", TrainingConfig.EPOCHS)
    logger.info("Batch Size: %s", TrainingConfig.BATCH_SIZE)
    logger.info("Image Size: %s", TrainingConfig.IMG_SIZE)
    logger.info("Workers: %s", TrainingConfig.NUM_WORKERS)
    logger.info("Device: %s", TrainingConfig.DEVICE)
    logger.info("Epoch Log: %s", TrainingConfig.EPOCH_LOG)

    model = YOLO(f"{TrainingConfig.MODEL_NAME}.pt")

    run_last = TrainingConfig.RUNS_DIR / "detect" / "weights" / "last.pt"
    resume_mode = run_last.exists()
    logger.info("Resume mode: %s", resume_mode)

    model.add_callback("on_fit_epoch_end", _build_epoch_logger_callback(TrainingConfig.EPOCH_LOG, logger))

    try:
        model.train(
            data=str(TrainingConfig.WORK_DIR / TrainingConfig.DATA_YAML),
            epochs=TrainingConfig.EPOCHS,
            imgsz=TrainingConfig.IMG_SIZE,
            batch=TrainingConfig.BATCH_SIZE,
            patience=TrainingConfig.PATIENCE,
            device=TrainingConfig.DEVICE,
            workers=TrainingConfig.NUM_WORKERS,
            resume=resume_mode,
            save=True,
            project=str(TrainingConfig.RUNS_DIR),
            name="detect",
            exist_ok=True,
            verbose=False,
            plots=True,
            conf=TrainingConfig.CONFIDENCE_THRESHOLD,
            amp=True,
        )

        logger.info("=" * 70)
        logger.info("Training completed successfully")
        _save_best_params_from_csv(logger, checkpoint_manager)
        logger.info("=" * 70)
        return True

    except Exception as exc:
        logger.error("Training failed with error: %s", exc)
        logger.error("=" * 70)
        return False


def evaluate_model(logger: logging.Logger) -> bool:
    logger.info("\n" + "=" * 70)
    logger.info("MODEL EVALUATION")
    logger.info("=" * 70)

    best_model_path = TrainingConfig.RUNS_DIR / "detect" / "weights" / "best.pt"
    if not best_model_path.exists():
        logger.warning("Best model not found for evaluation: %s", best_model_path)
        return False

    model = YOLO(str(best_model_path))
    metrics = model.val(
        data=str(TrainingConfig.WORK_DIR / TrainingConfig.DATA_YAML),
        imgsz=TrainingConfig.IMG_SIZE,
        conf=TrainingConfig.CONFIDENCE_THRESHOLD,
        verbose=False,
    )

    logger.info("mAP50: %.4f", float(getattr(metrics.box, "map50", 0.0)))
    logger.info("mAP50-95: %.4f", float(getattr(metrics.box, "map", 0.0)))
    logger.info("=" * 70)
    return True


def main() -> int:
    TrainingConfig.create_directories()
    logger = setup_logger(TrainingConfig.TRAINING_LOG)
    checkpoint_manager = CheckpointManager(TrainingConfig.PARAMS_FILE)

    logger.info("=" * 70)
    logger.info("YOLO11 TRAINING PIPELINE")
    logger.info("Training log: %s", TrainingConfig.TRAINING_LOG)
    logger.info("Epoch log: %s", TrainingConfig.EPOCH_LOG)
    logger.info("=" * 70)

    check_environment(logger)

    if not validate_data(logger):
        logger.error("Dataset validation failed. Exiting")
        return 1

    if not train_model(logger, checkpoint_manager):
        logger.error("Training failed. Exiting")
        return 1

    evaluate_model(logger)

    logger.info("\n" + "=" * 70)
    logger.info("ALL STEPS COMPLETED")
    logger.info("Training log: %s", TrainingConfig.TRAINING_LOG)
    logger.info("Epoch log: %s", TrainingConfig.EPOCH_LOG)
    logger.info("Best model: %s", TrainingConfig.RUNS_DIR / "detect" / "weights" / "best.pt")
    logger.info("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
