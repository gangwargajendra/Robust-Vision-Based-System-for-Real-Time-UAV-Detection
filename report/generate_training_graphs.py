"""
Generate training graphs for YOLO11 drone detection report.
Combines Phase 1 (epochs 1-200) and Phase 2 (epochs 201-500) data.
Outputs: training loss curves, mAP curves, precision/recall curves.
"""

import csv
import os
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# ─── Paths ───
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
PHASE1_CSV = os.path.join(ROOT_DIR, "FIRST", "results", "epoch_progress_20260320_015316.txt")
PHASE2_CSV = os.path.join(ROOT_DIR, "SECOND", "results", "epoch_progress_201_to_500.txt")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "graphs")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_csv(path):
    epochs, box, cls, dfl, prec, rec, map50, map50_95 = [], [], [], [], [], [], [], []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or row[0].strip() == "epoch_step":
                continue
            try:
                e = int(float(row[1]))
                epochs.append(e)
                box.append(float(row[3]))
                cls.append(float(row[4]))
                dfl.append(float(row[5]))
                prec.append(float(row[6]))
                rec.append(float(row[7]))
                map50.append(float(row[8]))
                map50_95.append(float(row[9]))
            except (ValueError, IndexError):
                continue
    return {
        "epoch": epochs, "box_loss": box, "cls_loss": cls, "dfl_loss": dfl,
        "precision": prec, "recall": rec, "map50": map50, "map50_95": map50_95,
    }


def merge_phases(p1, p2):
    merged = {}
    for key in p1:
        merged[key] = p1[key] + p2[key]
    return merged


def plot_training_losses(data, output_dir):
    """Plot box, cls, dfl losses over all epochs."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    losses = [("box_loss", "Box Loss"), ("cls_loss", "Classification Loss"), ("dfl_loss", "DFL Loss")]
    colors = ["#2196F3", "#FF5722", "#4CAF50"]

    for ax, (key, title), color in zip(axes, losses, colors):
        ax.plot(data["epoch"], data[key], color=color, linewidth=0.8, alpha=0.7)
        ax.axvline(x=200, color="gray", linestyle="--", linewidth=1, alpha=0.6, label="Phase boundary")
        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_ylabel("Loss", fontsize=12)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "training_losses.png"), dpi=300, bbox_inches="tight")
    plt.savefig(os.path.join(output_dir, "training_losses.pdf"), bbox_inches="tight")
    plt.close()
    print("Saved: training_losses.png / .pdf")


def plot_combined_loss(data, output_dir):
    """Plot all three losses on one chart."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(data["epoch"], data["box_loss"], label="Box Loss", color="#2196F3", linewidth=1)
    ax.plot(data["epoch"], data["cls_loss"], label="Classification Loss", color="#FF5722", linewidth=1)
    ax.plot(data["epoch"], data["dfl_loss"], label="DFL Loss", color="#4CAF50", linewidth=1)
    ax.axvline(x=200, color="gray", linestyle="--", linewidth=1, alpha=0.6, label="Phase 1 → Phase 2")
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.set_title("Training Losses Over 500 Epochs", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "training_losses_combined.png"), dpi=300, bbox_inches="tight")
    plt.savefig(os.path.join(output_dir, "training_losses_combined.pdf"), bbox_inches="tight")
    plt.close()
    print("Saved: training_losses_combined.png / .pdf")


def plot_map_curves(data, output_dir):
    """Plot mAP50 and mAP50-95 over epochs."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(data["epoch"], data["map50"], label="mAP@50", color="#1976D2", linewidth=1.2)
    ax.plot(data["epoch"], data["map50_95"], label="mAP@50-95", color="#E64A19", linewidth=1.2)
    ax.axvline(x=200, color="gray", linestyle="--", linewidth=1, alpha=0.6, label="Phase 1 → Phase 2")
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("mAP", fontsize=12)
    ax.set_title("mAP Progression Over 500 Epochs", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.3, 1.0)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "map_curves.png"), dpi=300, bbox_inches="tight")
    plt.savefig(os.path.join(output_dir, "map_curves.pdf"), bbox_inches="tight")
    plt.close()
    print("Saved: map_curves.png / .pdf")


def plot_precision_recall(data, output_dir):
    """Plot Precision and Recall over epochs."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(data["epoch"], data["precision"], label="Precision", color="#388E3C", linewidth=1.2)
    ax.plot(data["epoch"], data["recall"], label="Recall", color="#F57C00", linewidth=1.2)
    ax.axvline(x=200, color="gray", linestyle="--", linewidth=1, alpha=0.6, label="Phase 1 → Phase 2")
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Precision and Recall Over 500 Epochs", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.4, 1.0)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "precision_recall_curves.png"), dpi=300, bbox_inches="tight")
    plt.savefig(os.path.join(output_dir, "precision_recall_curves.pdf"), bbox_inches="tight")
    plt.close()
    print("Saved: precision_recall_curves.png / .pdf")


def plot_phase_comparison_summary(p1, p2, output_dir):
    """Bar chart comparing final metrics of Phase 1 vs Phase 2."""
    metrics = ["Precision", "Recall", "mAP@50", "mAP@50-95"]
    phase1_vals = [p1["precision"][-1], p1["recall"][-1], p1["map50"][-1], p1["map50_95"][-1]]
    phase2_vals = [p2["precision"][-1], p2["recall"][-1], p2["map50"][-1], p2["map50_95"][-1]]

    x = range(len(metrics))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar([i - width/2 for i in x], phase1_vals, width, label="Phase 1 (Epoch 200)", color="#42A5F5")
    bars2 = ax.bar([i + width/2 for i in x], phase2_vals, width, label="Phase 2 (Epoch 500)", color="#EF5350")

    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Phase 1 vs Phase 2: Final Metrics Comparison", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=11)
    ax.legend(fontsize=10)
    ax.set_ylim(0.5, 1.0)
    ax.grid(axis="y", alpha=0.3)

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{bar.get_height():.4f}", ha="center", va="bottom", fontsize=8)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{bar.get_height():.4f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "phase_comparison.png"), dpi=300, bbox_inches="tight")
    plt.savefig(os.path.join(output_dir, "phase_comparison.pdf"), bbox_inches="tight")
    plt.close()
    print("Saved: phase_comparison.png / .pdf")


if __name__ == "__main__":
    print("Loading Phase 1 data...")
    p1 = load_csv(PHASE1_CSV)
    print(f"  Phase 1: {len(p1['epoch'])} epochs loaded (1-{p1['epoch'][-1]})")

    print("Loading Phase 2 data...")
    p2 = load_csv(PHASE2_CSV)
    print(f"  Phase 2: {len(p2['epoch'])} epochs loaded ({p2['epoch'][0]}-{p2['epoch'][-1]})")

    merged = merge_phases(p1, p2)
    print(f"  Combined: {len(merged['epoch'])} epochs\n")

    print("Generating graphs...")
    plot_training_losses(merged, OUTPUT_DIR)
    plot_combined_loss(merged, OUTPUT_DIR)
    plot_map_curves(merged, OUTPUT_DIR)
    plot_precision_recall(merged, OUTPUT_DIR)
    plot_phase_comparison_summary(p1, p2, OUTPUT_DIR)
    print(f"\nAll training graphs saved to: {OUTPUT_DIR}")
