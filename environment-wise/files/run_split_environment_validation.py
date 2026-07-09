"""
Run environment-wise validation on split datasets and save text reports.

Expected YAML files in current working directory:
- data_val_bright_clear.yaml
- data_val_night_lowlight.yaml
- data_val_cluttered_ambiguous.yaml

Outputs:
- val_bright_clear_metrics.txt
- val_night_lowlight_metrics.txt
- val_cluttered_ambiguous_metrics.txt
- environment_split_validation_summary.txt
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from ultralytics import YOLO


BASE_DIR = Path(__file__).resolve().parent
ENV_WISE_DIR = BASE_DIR.parent
CODE_DIR = ENV_WISE_DIR.parent
INTERMEDIATE_DIR = ENV_WISE_DIR / "intermediate"
RESULTS_DIR = ENV_WISE_DIR / "results"
RUNS_DIR = ENV_WISE_DIR / "runs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Environment-wise validation runner")
    parser.add_argument("--model", default=str(CODE_DIR / "runs" / "detect" / "weights" / "best.pt"), help="Path to model weights")
    parser.add_argument("--bright-yaml", default=str(INTERMEDIATE_DIR / "data_val_bright_clear.yaml"), help="YAML for bright-clear split")
    parser.add_argument("--night-yaml", default=str(INTERMEDIATE_DIR / "data_val_night_lowlight.yaml"), help="YAML for night-lowlight split")
    parser.add_argument(
        "--cluttered-yaml",
        default=str(INTERMEDIATE_DIR / "data_val_cluttered_ambiguous.yaml"),
        help="YAML for cluttered-ambiguous split",
    )
    parser.add_argument("--device", default="0", help="Validation device, e.g. 0 or cpu")
    parser.add_argument("--save-prefix", default="val_env", help="Prefix for per-split metric files")
    return parser.parse_args()


def validate_split(model: YOLO, yaml_path: str, split_name: str, device: str, prefix: str) -> dict[str, float | str]:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    metrics = model.val(
        data=yaml_path,
        split="val",
        device=device,
        verbose=False,
        project=str(RUNS_DIR),
        name=f"val_{split_name}",
        exist_ok=True,
    )

    result = {
        "split": split_name,
        "yaml": yaml_path,
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "map50": float(metrics.box.map50),
        "map50_95": float(metrics.box.map),
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_file = RESULTS_DIR / f"{prefix}_{split_name}_metrics.txt"
    lines = [
        f"{split_name.upper()} VALIDATION",
        "=" * (len(split_name) + 11),
        f"YAML: {yaml_path}",
        f"Precision: {result['precision']:.6f}",
        f"Recall: {result['recall']:.6f}",
        f"mAP50: {result['map50']:.6f}",
        f"mAP50-95: {result['map50_95']:.6f}",
        "",
    ]
    out_file.write_text("\n".join(lines), encoding="utf-8")

    return result


def main() -> None:
    args = parse_args()

    required = [args.bright_yaml, args.night_yaml, args.cluttered_yaml]
    missing = [p for p in required if not Path(p).exists()]
    if missing:
        raise FileNotFoundError(f"Missing YAML files: {', '.join(missing)}")

    if not Path(args.model).exists():
        raise FileNotFoundError(f"Model not found: {args.model}")

    model = YOLO(args.model)

    split_defs = [
        ("bright_clear", args.bright_yaml),
        ("night_lowlight", args.night_yaml),
        ("cluttered_ambiguous", args.cluttered_yaml),
    ]

    results = []
    for split_name, yaml_file in split_defs:
        print(f"Running {split_name} validation using {yaml_file}...")
        results.append(validate_split(model, yaml_file, split_name, args.device, args.save_prefix))

    summary_lines = [
        "ENVIRONMENT-WISE VALIDATION (SPLIT DATASETS)",
        "============================================",
        f"Model: {args.model}",
        f"Device: {args.device}",
        f"Generated at: {datetime.now().isoformat(timespec='seconds')}",
        "",
    ]

    for r in results:
        summary_lines.extend(
            [
                f"{str(r['split']).upper()} VALIDATION",
                "--------------------------------",
                f"YAML: {r['yaml']}",
                f"Precision: {r['precision']:.6f}",
                f"Recall: {r['recall']:.6f}",
                f"mAP50: {r['map50']:.6f}",
                f"mAP50-95: {r['map50_95']:.6f}",
                "",
            ]
        )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summary_file = RESULTS_DIR / "environment_split_validation_summary.txt"
    summary_file.write_text("\n".join(summary_lines), encoding="utf-8")

    print("Done. Created files:")
    print(f"- {args.save_prefix}_bright_clear_metrics.txt")
    print(f"- {args.save_prefix}_night_lowlight_metrics.txt")
    print(f"- {args.save_prefix}_cluttered_ambiguous_metrics.txt")
    print(f"- {summary_file}")


if __name__ == "__main__":
    main()
