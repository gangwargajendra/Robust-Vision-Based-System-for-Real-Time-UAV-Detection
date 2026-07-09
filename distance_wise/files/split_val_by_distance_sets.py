"""
Split YOLO val set into distance-based subsets using val_images_*.txt,
then create new YAMLs for separate validation.

Assumes:
- data.yaml exists in the current directory
- val_images_pure_close.txt, val_images_pure_medium.txt,
  val_images_pure_far.txt, val_images_mixed.txt, val_images_empty.txt
  are in the current directory (with image *stems* only).

Result:
- New dirs under dataset root/distance-wise:
    distance-wise/images/val_close,  distance-wise/labels/val_close
    distance-wise/images/val_medium, distance-wise/labels/val_medium
    distance-wise/images/val_far,    distance-wise/labels/val_far
    distance-wise/images/val_mixed,  distance-wise/labels/val_mixed
    distance-wise/images/val_empty,  distance-wise/labels/val_empty
- New YAMLs in distance-wise/intermediate:
    data_val_close.yaml
    data_val_medium.yaml
    data_val_far.yaml
    data_val_mixed.yaml
    data_val_empty.yaml
"""

from pathlib import Path
import shutil
import yaml

# ------------- CONFIG -----------------
BASE_DIR = Path(__file__).resolve().parent
DISTANCE_WISE_DIR = BASE_DIR.parent
CODE_DIR = DISTANCE_WISE_DIR.parent

DATA_YAML = CODE_DIR / "data.yaml"     # existing data.yaml in code/
LIST_DIR = DISTANCE_WISE_DIR / "intermediate"
YAML_OUT_DIR = DISTANCE_WISE_DIR / "intermediate"
USE_SYMLINK = False                    # set True to symlink instead of copy
SPLIT_BASE_NAME = "distance-wise"      # folder created under dataset root
# --------------------------------------


def read_stems(txt_path: Path) -> list[str]:
    if not txt_path.exists():
        print(f"[WARN] {txt_path} not found, skipping.")
        return []
    stems = [l.strip() for l in txt_path.read_text().splitlines() if l.strip()]
    return stems


def find_image_path(stem: str, img_dir: Path) -> Path | None:
    exts = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]
    for ext in exts:
        p = img_dir / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def copy_or_link(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if USE_SYMLINK:
        if dst.exists():
            dst.unlink()
        dst.symlink_to(src)
    else:
        shutil.copy2(src, dst)


def main():
    assert DATA_YAML.exists(), f"{DATA_YAML} not found"
    LIST_DIR.mkdir(parents=True, exist_ok=True)
    YAML_OUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- Load dataset config from data.yaml ---
    cfg = yaml.safe_load(DATA_YAML.read_text())
    dataset_root = Path(cfg["path"])                 # /.../dataset/nondrone_copy
    train_rel = Path(cfg["train"])                  # images/train
    val_rel = Path(cfg["val"])                      # images/val
    nc = cfg["nc"]
    names = cfg["names"]

    images_val_dir = dataset_root / val_rel         # .../images/val
    labels_val_dir = dataset_root / "labels" / "val"
    split_base_dir = dataset_root / SPLIT_BASE_NAME

    print(f"data.yaml          : {DATA_YAML}")
    print(f"Dataset root       : {dataset_root}")
    print(f"Val images dir     : {images_val_dir}")
    print(f"Val labels dir     : {labels_val_dir}")
    print(f"Split output dir   : {split_base_dir}")
    print(f"List dir           : {LIST_DIR}")
    print(f"YAML output dir    : {YAML_OUT_DIR}")

    # Create split root directory if needed
    split_base_dir.mkdir(parents=True, exist_ok=True)

    # --- Read stems from analysis lists ---
    close_stems  = read_stems(LIST_DIR / "val_images_pure_close.txt")
    medium_stems = read_stems(LIST_DIR / "val_images_pure_medium.txt")
    far_stems    = read_stems(LIST_DIR / "val_images_pure_far.txt")
    mixed_stems  = read_stems(LIST_DIR / "val_images_mixed.txt")
    empty_stems  = read_stems(LIST_DIR / "val_images_empty.txt")

    print(f"\nStems loaded:")
    print(f"  close :  {len(close_stems)}")
    print(f"  medium:  {len(medium_stems)}")
    print(f"  far   :  {len(far_stems)}")
    print(f"  mixed :  {len(mixed_stems)}")
    print(f"  empty :  {len(empty_stems)}")

    # --- Define target subdirs relative to split_validation ---
    subsets = {
        "val_close":  close_stems,
        "val_medium": medium_stems,
        "val_far":    far_stems,
        "val_mixed":  mixed_stems,
        "val_empty":  empty_stems,
    }

    for subset_name, stems in subsets.items():
        if not stems:
            print(f"\n[INFO] No stems for {subset_name}, skipping.")
            continue

        img_out_dir = split_base_dir / "images" / subset_name
        lbl_out_dir = split_base_dir / "labels" / subset_name
        img_out_dir.mkdir(parents=True, exist_ok=True)
        lbl_out_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n=== Processing {subset_name} ({len(stems)} images) ===")
        missing_images = 0
        missing_labels = 0

        for stem in stems:
            # locate source image
            img_src = find_image_path(stem, images_val_dir)
            if img_src is None:
                missing_images += 1
                print(f"  [WARN] image not found for stem: {stem}")
                continue

            # label path
            lbl_src = labels_val_dir / f"{stem}.txt"
            if not lbl_src.exists():
                missing_labels += 1
                print(f"  [WARN] label not found for stem: {stem}")
                # still copy image so you can inspect later if needed
            # destination paths
            img_dst = img_out_dir / img_src.name
            lbl_dst = lbl_out_dir / f"{stem}.txt"

            copy_or_link(img_src, img_dst)
            if lbl_src.exists():
                copy_or_link(lbl_src, lbl_dst)

        print(f"  Done {subset_name}:")
        print(f"    Missing images: {missing_images}")
        print(f"    Missing labels: {missing_labels}")
        print(f"    Output images:  {len(list(img_out_dir.glob('*')))}")
        print(f"    Output labels:  {len(list(lbl_out_dir.glob('*.txt')))}")

    # --- Create 5 YAML files for Close / Medium / Far / Mixed / Empty validation ---
    base_cfg = {
        "path": str(dataset_root),
        "train": str(train_rel),   # keep same train
        "nc": nc,
        "names": names,
    }

    split_base_rel = Path(SPLIT_BASE_NAME)
    yaml_defs = {
        "data_val_close.yaml":  str(split_base_rel / "images" / "val_close"),
        "data_val_medium.yaml": str(split_base_rel / "images" / "val_medium"),
        "data_val_far.yaml":    str(split_base_rel / "images" / "val_far"),
        "data_val_mixed.yaml":  str(split_base_rel / "images" / "val_mixed"),
        "data_val_empty.yaml":  str(split_base_rel / "images" / "val_empty"),
    }

    print("\n=== Writing split-specific YAML files ===")
    for yaml_name, val_path_rel in yaml_defs.items():
        cfg_out = base_cfg.copy()
        cfg_out["val"] = val_path_rel
        out_path = YAML_OUT_DIR / yaml_name
        out_path.write_text(yaml.dump(cfg_out, sort_keys=False), encoding="utf-8")
        print(f"  Wrote {yaml_name} with val='{val_path_rel}'")

    print("\nAll done. You can now run model.val() separately on each YAML.")


if __name__ == "__main__":
    main()