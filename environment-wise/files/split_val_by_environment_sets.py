"""
Split YOLO val set into environment-based subsets using val_images_*.txt,
copy images and labels to a new dataset folder 'environment-wise', and
create YAML files for split-wise validation.

Required list files in current working directory:
- val_images_bright_clear.txt
- val_images_night_lowlight.txt
- val_images_cluttered_ambiguous.txt

Output under dataset root:
- environment-wise/images/val_bright_clear
- environment-wise/images/val_night_lowlight
- environment-wise/images/val_cluttered_ambiguous
- environment-wise/labels/val_bright_clear
- environment-wise/labels/val_night_lowlight
- environment-wise/labels/val_cluttered_ambiguous

YAML outputs in current working directory:
- data_val_bright_clear.yaml
- data_val_night_lowlight.yaml
- data_val_cluttered_ambiguous.yaml
"""

from __future__ import annotations

from pathlib import Path
import shutil
import yaml


BASE_DIR = Path(__file__).resolve().parent
ENV_WISE_DIR = BASE_DIR.parent
CODE_DIR = ENV_WISE_DIR.parent

DATA_YAML = CODE_DIR / "data.yaml"
LIST_DIR = ENV_WISE_DIR / "intermediate"
YAML_OUT_DIR = ENV_WISE_DIR / "intermediate"
SPLIT_BASE_NAME = "environment-wise"
USE_SYMLINK = False


def _read_stems(path: Path) -> list[str]:
    if not path.exists():
        print(f"[WARN] Missing list file: {path}")
        return []
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _find_image(stem: str, img_dir: Path) -> Path | None:
    for ext in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]:
        p = img_dir / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def _copy_or_link(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if USE_SYMLINK:
        if dst.exists():
            dst.unlink()
        dst.symlink_to(src)
    else:
        shutil.copy2(src, dst)


def main() -> None:
    if not DATA_YAML.exists():
        raise FileNotFoundError(f"{DATA_YAML} not found")

    LIST_DIR.mkdir(parents=True, exist_ok=True)
    YAML_OUT_DIR.mkdir(parents=True, exist_ok=True)

    cfg = yaml.safe_load(DATA_YAML.read_text(encoding="utf-8"))
    dataset_root = Path(cfg["path"])
    train_rel = Path(cfg["train"])
    val_rel = Path(cfg["val"])
    nc = cfg["nc"]
    names = cfg["names"]

    images_val_dir = dataset_root / val_rel
    labels_val_dir = dataset_root / "labels" / "val"

    split_base_dir = dataset_root / SPLIT_BASE_NAME
    split_base_dir.mkdir(parents=True, exist_ok=True)

    print(f"data.yaml: {DATA_YAML}")
    print(f"Dataset root: {dataset_root}")
    print(f"Val images dir: {images_val_dir}")
    print(f"Val labels dir: {labels_val_dir}")
    print(f"Split output dir: {split_base_dir}")
    print(f"List dir: {LIST_DIR}")
    print(f"YAML output dir: {YAML_OUT_DIR}")

    subset_to_stems = {
        "val_bright_clear": _read_stems(LIST_DIR / "val_images_bright_clear.txt"),
        "val_night_lowlight": _read_stems(LIST_DIR / "val_images_night_lowlight.txt"),
        "val_cluttered_ambiguous": _read_stems(LIST_DIR / "val_images_cluttered_ambiguous.txt"),
    }

    for subset_name, stems in subset_to_stems.items():
        if not stems:
            print(f"\n[INFO] No entries for {subset_name}; skipping copy.")
            continue

        img_out_dir = split_base_dir / "images" / subset_name
        lbl_out_dir = split_base_dir / "labels" / subset_name
        img_out_dir.mkdir(parents=True, exist_ok=True)
        lbl_out_dir.mkdir(parents=True, exist_ok=True)

        missing_images = 0
        missing_labels = 0

        print(f"\n=== Processing {subset_name} ({len(stems)} images) ===")
        for stem in stems:
            img_src = _find_image(stem, images_val_dir)
            if img_src is None:
                missing_images += 1
                print(f"  [WARN] Missing image for stem: {stem}")
                continue

            lbl_src = labels_val_dir / f"{stem}.txt"
            img_dst = img_out_dir / img_src.name
            lbl_dst = lbl_out_dir / f"{stem}.txt"

            _copy_or_link(img_src, img_dst)
            if lbl_src.exists():
                _copy_or_link(lbl_src, lbl_dst)
            else:
                missing_labels += 1

        out_img_count = len([p for p in img_out_dir.glob("*") if p.is_file()])
        out_lbl_count = len([p for p in lbl_out_dir.glob("*.txt") if p.is_file()])
        print(f"  Missing images: {missing_images}")
        print(f"  Missing labels: {missing_labels}")
        print(f"  Output images: {out_img_count}")
        print(f"  Output labels: {out_lbl_count}")

    base_cfg = {
        "path": str(dataset_root),
        "train": str(train_rel),
        "nc": nc,
        "names": names,
    }

    split_rel = Path(SPLIT_BASE_NAME)
    yaml_defs = {
        "data_val_bright_clear.yaml": str(split_rel / "images" / "val_bright_clear"),
        "data_val_night_lowlight.yaml": str(split_rel / "images" / "val_night_lowlight"),
        "data_val_cluttered_ambiguous.yaml": str(split_rel / "images" / "val_cluttered_ambiguous"),
    }

    print("\n=== Writing environment-wise YAML files ===")
    for yaml_name, val_path_rel in yaml_defs.items():
        out_cfg = dict(base_cfg)
        out_cfg["val"] = val_path_rel
        (YAML_OUT_DIR / yaml_name).write_text(yaml.dump(out_cfg, sort_keys=False), encoding="utf-8")
        print(f"  Wrote {yaml_name} with val='{val_path_rel}'")

    print("\nAll done. Environment-wise split prepared.")


if __name__ == "__main__":
    main()
