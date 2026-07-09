"""
GPU-accelerated analysis of validation images into environment conditions:
- bright_clear
- night_lowlight
- cluttered_ambiguous

Uses PyTorch CUDA for fast batch processing of image statistics.

Outputs in intermediate directory:
- val_images_bright_clear.txt
- val_images_night_lowlight.txt
- val_images_cluttered_ambiguous.txt
- val_environment_analysis_report.txt
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn.functional as F
from torchvision.io import read_image, ImageReadMode
import yaml


# Server-first defaults with local fallback.
BASE_DIR = Path(__file__).resolve().parent
ENV_WISE_DIR = BASE_DIR.parent
CODE_DIR = ENV_WISE_DIR.parent
SERVER_DATA_YAML = Path("/home/btech/2023/piyush.kumar23b/gajendra/code/data.yaml")
LOCAL_DATA_YAML = CODE_DIR / "data.yaml"
DATA_YAML = SERVER_DATA_YAML if SERVER_DATA_YAML.exists() else LOCAL_DATA_YAML

# Output directory
OUTPUT_DIR = ENV_WISE_DIR / "intermediate"

# Batch size for GPU processing (tune based on GPU memory)
BATCH_SIZE = 32


@dataclass
class ImageStats:
    stem: str
    mean_luma: float
    std_luma: float
    p90_luma: float
    edge_density: float
    blue_ratio: float


def _resolve_val_image_dir(data_yaml: Path) -> Path:
    if not data_yaml.exists():
        raise FileNotFoundError(f"data.yaml not found: {data_yaml}")

    cfg = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    dataset_root = Path(cfg["path"])
    val_rel = Path(cfg["val"])
    val_dir = dataset_root / val_rel
    if not val_dir.exists():
        raise FileNotFoundError(f"Validation image directory not found: {val_dir}")
    return val_dir


def _collect_image_paths(val_dir: Path) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return sorted(p for p in val_dir.iterdir() if p.is_file() and p.suffix.lower() in exts)


def _load_image_gpu(path: Path, device: torch.device) -> torch.Tensor | None:
    """Load image as float32 tensor on GPU. Returns None if failed."""
    try:
        # read_image returns uint8 [C, H, W]
        img = read_image(str(path), mode=ImageReadMode.RGB)
        return img.to(device=device, dtype=torch.float32)
    except Exception:
        return None


def _compute_stats_gpu(img: torch.Tensor, stem: str) -> ImageStats:
    """Compute all statistics on GPU using PyTorch operations."""
    # img shape: [3, H, W], values 0-255 float32 on GPU
    
    # Perceptual luminance: 0.299*R + 0.587*G + 0.114*B
    luma = 0.299 * img[0] + 0.587 * img[1] + 0.114 * img[2]  # [H, W]
    
    mean_luma = float(luma.mean().item())
    std_luma = float(luma.std().item())
    
    # 90th percentile using quantile
    p90_luma = float(torch.quantile(luma.flatten(), 0.9).item())
    
    # Edge density via Sobel-like gradient approximation
    # Pad for gradient computation
    luma_4d = luma.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
    
    # Sobel kernels
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                           dtype=torch.float32, device=img.device).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                           dtype=torch.float32, device=img.device).view(1, 1, 3, 3)
    
    gx = F.conv2d(luma_4d, sobel_x, padding=1)
    gy = F.conv2d(luma_4d, sobel_y, padding=1)
    grad_mag = torch.sqrt(gx * gx + gy * gy).squeeze()
    
    # Edge threshold scaled for Sobel (roughly 12 * 4 = 48 due to Sobel scaling)
    edge_density = float((grad_mag > 48.0).float().mean().item())
    
    # Blue-sky proxy: blue > red + 8 AND blue > green + 8
    red, green, blue = img[0], img[1], img[2]
    blue_dominant = (blue > red + 8.0) & (blue > green + 8.0)
    blue_ratio = float(blue_dominant.float().mean().item())
    
    return ImageStats(
        stem=stem,
        mean_luma=mean_luma,
        std_luma=std_luma,
        p90_luma=p90_luma,
        edge_density=edge_density,
        blue_ratio=blue_ratio,
    )


def _classify(stats: ImageStats) -> str:
    """Classification logic - UNCHANGED from original."""
    # 1) Night / low-light: globally dark or weak highlights.
    if stats.mean_luma < 85.0 or stats.p90_luma < 125.0:
        return "night_lowlight"

    # 2) Bright clear-sky: bright, relatively simple background and visible sky tendency.
    if (
        stats.mean_luma >= 130.0
        and stats.std_luma <= 58.0
        and stats.edge_density <= 0.16
        and stats.blue_ratio >= 0.12
    ):
        return "bright_clear"

    # 3) Remaining difficult scenes.
    return "cluttered_ambiguous"


def _write_list(path: Path, items: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(items), encoding="utf-8")


def main() -> None:
    start_time = time.time()
    
    # Setup device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("CUDA not available, using CPU (will be slower)")
    
    val_dir = _resolve_val_image_dir(DATA_YAML)
    print(f"Using data.yaml: {DATA_YAML}")
    print(f"Validation image dir: {val_dir}")
    print(f"Output dir: {OUTPUT_DIR}")
    
    # Collect all image paths first
    image_paths = _collect_image_paths(val_dir)
    total_images = len(image_paths)
    print(f"Found {total_images} images to process")
    
    # Results
    bright_clear: List[str] = []
    night_lowlight: List[str] = []
    cluttered_ambiguous: List[str] = []
    stats_rows: List[ImageStats] = []
    failed = 0
    
    # Process images one by one but with GPU acceleration
    # (Batch loading is complex due to variable image sizes)
    print("\nProcessing images...")
    last_report = time.time()
    
    for idx, img_path in enumerate(image_paths):
        img = _load_image_gpu(img_path, device)
        
        if img is None:
            failed += 1
            cluttered_ambiguous.append(img_path.stem)
            continue
        
        stats = _compute_stats_gpu(img, img_path.stem)
        label = _classify(stats)
        
        if label == "bright_clear":
            bright_clear.append(stats.stem)
        elif label == "night_lowlight":
            night_lowlight.append(stats.stem)
        else:
            cluttered_ambiguous.append(stats.stem)
        
        stats_rows.append(stats)
        
        # Progress report every 2 seconds
        now = time.time()
        if now - last_report >= 2.0:
            elapsed = now - start_time
            rate = (idx + 1) / elapsed
            remaining = (total_images - idx - 1) / rate if rate > 0 else 0
            print(f"  [{idx + 1}/{total_images}] {rate:.1f} img/sec, ETA: {remaining:.0f}s")
            last_report = now
    
    # Write output files
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _write_list(OUTPUT_DIR / "val_images_bright_clear.txt", bright_clear)
    _write_list(OUTPUT_DIR / "val_images_night_lowlight.txt", night_lowlight)
    _write_list(OUTPUT_DIR / "val_images_cluttered_ambiguous.txt", cluttered_ambiguous)
    
    total = len(bright_clear) + len(night_lowlight) + len(cluttered_ambiguous)
    elapsed_total = time.time() - start_time
    
    lines = [
        "ENVIRONMENT-WISE VALIDATION ANALYSIS REPORT",
        "===========================================",
        f"data.yaml: {DATA_YAML}",
        f"val dir: {val_dir}",
        f"processed images: {total}",
        f"failed reads: {failed}",
        f"processing time: {elapsed_total:.1f} seconds",
        f"throughput: {total / elapsed_total:.1f} images/sec",
        "",
        "CATEGORY COUNTS",
        "---------------",
        f"bright_clear: {len(bright_clear)}",
        f"night_lowlight: {len(night_lowlight)}",
        f"cluttered_ambiguous: {len(cluttered_ambiguous)}",
    ]
    
    if total > 0:
        lines.extend([
            "",
            "CATEGORY PERCENTAGES",
            "--------------------",
            f"bright_clear: {100.0 * len(bright_clear) / total:.2f}%",
            f"night_lowlight: {100.0 * len(night_lowlight) / total:.2f}%",
            f"cluttered_ambiguous: {100.0 * len(cluttered_ambiguous) / total:.2f}%",
        ])
    
    if stats_rows:
        mean_luma = sum(s.mean_luma for s in stats_rows) / len(stats_rows)
        mean_std = sum(s.std_luma for s in stats_rows) / len(stats_rows)
        mean_edge = sum(s.edge_density for s in stats_rows) / len(stats_rows)
        lines.extend([
            "",
            "GLOBAL IMAGE STATISTICS (MEAN)",
            "------------------------------",
            f"mean_luma: {mean_luma:.2f}",
            f"std_luma: {mean_std:.2f}",
            f"edge_density: {mean_edge:.4f}",
        ])
    
    (OUTPUT_DIR / "val_environment_analysis_report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    
    print(f"\nDone in {elapsed_total:.1f} seconds ({total / elapsed_total:.1f} img/sec)")
    print(f"Saved to {OUTPUT_DIR}:")
    print("- val_images_bright_clear.txt")
    print("- val_images_night_lowlight.txt")
    print("- val_images_cluttered_ambiguous.txt")
    print("- val_environment_analysis_report.txt")


if __name__ == "__main__":
    main()
