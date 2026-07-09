"""
Analyze val split labels by distance bucket.
Identifies: pure-close, pure-medium, pure-far, mixed, and empty images.
Uses the SAME distance formula as distance_based_validation.py
"""

import math
from pathlib import Path
from collections import defaultdict
import yaml

# ─────────────────────────────────────────────
# CONFIG — derive paths from script location
BASE_DIR = Path(__file__).resolve().parent
DISTANCE_WISE_DIR = BASE_DIR.parent
CODE_DIR = DISTANCE_WISE_DIR.parent
INTERMEDIATE_DIR = DISTANCE_WISE_DIR / "intermediate"

SERVER_DATA_YAML = CODE_DIR / "data.yaml"
LOCAL_DATA_YAML = DISTANCE_WISE_DIR / "data.yaml"
DATA_YAML = SERVER_DATA_YAML if SERVER_DATA_YAML.exists() else LOCAL_DATA_YAML

if not DATA_YAML.exists():
    raise FileNotFoundError(f"data.yaml not found: {DATA_YAML}")

cfg = yaml.safe_load(DATA_YAML.read_text(encoding="utf-8"))
DATASET_ROOT = Path(cfg["path"])
LABEL_DIR = DATASET_ROOT / "labels" / "val"
# ─────────────────────────────────────────────

CLOSE_MAX  = 2.5
MEDIUM_MAX = 5.0

def rel_distance(w: float, h: float) -> float:
    return 1.0 / math.sqrt(w * w + h * h + 1e-12)

def bucket(d: float) -> str:
    if d < CLOSE_MAX:
        return "close"
    elif d <= MEDIUM_MAX:
        return "medium"
    else:
        return "far"

label_dir = Path(LABEL_DIR)
if not label_dir.exists():
    raise FileNotFoundError(f"Label directory not found: {label_dir}")
all_label_files = sorted(label_dir.glob("*.txt"))

print(f"\nLabel directory  : {label_dir}")
print(f"Total label files: {len(all_label_files)}\n")
print(f"Intermediate dir : {INTERMEDIATE_DIR}\n")

INTERMEDIATE_DIR.mkdir(parents=True, exist_ok=True)

# Counters
counts = {
    "pure_close":  0,
    "pure_medium": 0,
    "pure_far":    0,
    "mixed":       0,
    "empty":       0,
}

# For mixed images: which combinations exist?
mix_combos = defaultdict(int)   # e.g. {"close+far": 12, "close+medium+far": 3}

# Per-bucket object counts (across all images)
obj_counts = {"close": 0, "medium": 0, "far": 0}

# Store image lists per category (useful for later splitting)
image_lists = defaultdict(list)

for lf in all_label_files:
    lines = [l.strip() for l in lf.read_text().splitlines() if l.strip()]

    if not lines:
        counts["empty"] += 1
        image_lists["empty"].append(lf.stem)
        continue

    buckets_in_image = set()
    for line in lines:
        parts = line.split()
        if len(parts) != 5:
            continue
        _, x, y, w, h = parts
        w, h = float(w), float(h)
        d  = rel_distance(w, h)
        b  = bucket(d)
        buckets_in_image.add(b)
        obj_counts[b] += 1

    if len(buckets_in_image) == 0:
        counts["empty"] += 1
        image_lists["empty"].append(lf.stem)
    elif buckets_in_image == {"close"}:
        counts["pure_close"] += 1
        image_lists["pure_close"].append(lf.stem)
    elif buckets_in_image == {"medium"}:
        counts["pure_medium"] += 1
        image_lists["pure_medium"].append(lf.stem)
    elif buckets_in_image == {"far"}:
        counts["pure_far"] += 1
        image_lists["pure_far"].append(lf.stem)
    else:
        counts["mixed"] += 1
        combo_key = "+".join(sorted(buckets_in_image))
        mix_combos[combo_key] += 1
        image_lists["mixed"].append(lf.stem)

# ─── REPORT ───────────────────────────────────────────
total = len(all_label_files)
print("=" * 50)
print("IMAGE DISTRIBUTION BY DISTANCE BUCKET")
print("=" * 50)
if total == 0:
    print("  No label files found in the configured directory.")
else:
    print(f"  Pure Close Range    : {counts['pure_close']:>6}  ({counts['pure_close']/total*100:.1f}%)")
    print(f"  Pure Medium Range   : {counts['pure_medium']:>6}  ({counts['pure_medium']/total*100:.1f}%)")
    print(f"  Pure Very Far Range : {counts['pure_far']:>6}  ({counts['pure_far']/total*100:.1f}%)")
    print(f"  MIXED (multi-bucket): {counts['mixed']:>6}  ({counts['mixed']/total*100:.1f}%)")
    print(f"  Empty (no GT boxes) : {counts['empty']:>6}  ({counts['empty']/total*100:.1f}%)")
print(f"  ─────────────────────────────")
print(f"  TOTAL               : {total:>6}")

print("\n")
print("=" * 50)
print("MIXED IMAGE COMBINATIONS")
print("=" * 50)
if mix_combos:
    for combo, cnt in sorted(mix_combos.items(), key=lambda x: -x[1]):
        print(f"  {combo:<30} : {cnt:>5}  ({cnt/total*100:.1f}%)")
else:
    print("  None — no mixed images found!")

print("\n")
print("=" * 50)
print("TOTAL GT OBJECTS PER DISTANCE BUCKET")
print("=" * 50)
total_obj = sum(obj_counts.values())
for b, cnt in obj_counts.items():
    pct = (cnt / total_obj * 100) if total_obj else 0.0
    print(f"  {b.capitalize():<12}: {cnt:>6} objects  ({pct:.1f}%)")
print(f"  {'Total':<12}: {total_obj:>6} objects")

# ─── SAVE IMAGE NAME LISTS ────────────────────────────
print("\n")
print("=" * 50)
print("SAVING IMAGE NAME LISTS TO TXT FILES")
print("=" * 50)
for category, names in image_lists.items():
    out_path = INTERMEDIATE_DIR / f"val_images_{category}.txt"
    out_path.write_text("\n".join(names), encoding="utf-8")
    print(f"  Saved {len(names):>5} entries → {out_path}")

report_lines = [
    "DISTANCE-WISE ANALYSIS REPORT",
    "============================",
    f"data.yaml: {DATA_YAML}",
    f"dataset root: {DATASET_ROOT}",
    f"label dir: {label_dir}",
    "",
    "IMAGE DISTRIBUTION",
    "------------------",
    f"pure_close: {counts['pure_close']}",
    f"pure_medium: {counts['pure_medium']}",
    f"pure_far: {counts['pure_far']}",
    f"mixed: {counts['mixed']}",
    f"empty: {counts['empty']}",
    "",
    "OBJECT COUNTS",
    "-------------",
    f"close: {obj_counts['close']}",
    f"medium: {obj_counts['medium']}",
    f"far: {obj_counts['far']}",
    f"total: {total_obj}",
]

report_path = INTERMEDIATE_DIR / "val_distance_analysis_report.txt"
report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
print(f"\n  Saved report → {report_path}")

print("\nDone.\n")
