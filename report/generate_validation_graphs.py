"""
Generate validation comparison graphs for distance-wise and environment-wise evaluation.
Compares Phase 1 (1-200) vs Phase 2 (201-500) results.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "graphs")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ─── Distance-wise validation results ───

dist_phase1 = {
    "Close":  {"Precision": 0.966914, "Recall": 0.962347, "mAP@50": 0.983146, "mAP@50-95": 0.854760},
    "Medium": {"Precision": 0.974502, "Recall": 0.954746, "mAP@50": 0.979500, "mAP@50-95": 0.773127},
    "Far":    {"Precision": 0.921771, "Recall": 0.815547, "mAP@50": 0.878431, "mAP@50-95": 0.522868},
    "Mixed":  {"Precision": 0.960075, "Recall": 0.869613, "mAP@50": 0.915167, "mAP@50-95": 0.709105},
}

dist_phase2 = {
    "Close":  {"Precision": 0.953345, "Recall": 0.885150, "mAP@50": 0.953423, "mAP@50-95": 0.832208},
    "Medium": {"Precision": 0.984306, "Recall": 0.968981, "mAP@50": 0.988874, "mAP@50-95": 0.800337},
    "Far":    {"Precision": 0.949582, "Recall": 0.847886, "mAP@50": 0.906586, "mAP@50-95": 0.550181},
    "Mixed":  {"Precision": 0.957935, "Recall": 0.880967, "mAP@50": 0.927276, "mAP@50-95": 0.724901},
}

# ─── Environment-wise validation results ───

env_phase1 = {
    "Bright/Clear":        {"Precision": 0.954229, "Recall": 0.889004, "mAP@50": 0.930995, "mAP@50-95": 0.714148},
    "Night/Low-light":     {"Precision": 0.951911, "Recall": 0.870985, "mAP@50": 0.946090, "mAP@50-95": 0.624622},
    "Cluttered/Ambiguous": {"Precision": 0.936659, "Recall": 0.848682, "mAP@50": 0.905019, "mAP@50-95": 0.595163},
}

env_phase2 = {
    "Bright/Clear":        {"Precision": 0.966404, "Recall": 0.894252, "mAP@50": 0.948552, "mAP@50-95": 0.733083},
    "Night/Low-light":     {"Precision": 0.969499, "Recall": 0.883978, "mAP@50": 0.959998, "mAP@50-95": 0.658340},
    "Cluttered/Ambiguous": {"Precision": 0.952687, "Recall": 0.842483, "mAP@50": 0.913407, "mAP@50-95": 0.608711},
}


def plot_grouped_bar(categories, phase1_data, phase2_data, metric, title, filename, output_dir):
    """Grouped bar chart for a single metric comparing Phase 1 vs Phase 2."""
    p1_vals = [phase1_data[c][metric] for c in categories]
    p2_vals = [phase2_data[c][metric] for c in categories]

    x = np.arange(len(categories))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, p1_vals, width, label="Phase 1 (Epoch 200)", color="#42A5F5", edgecolor="white")
    bars2 = ax.bar(x + width/2, p2_vals, width, label="Phase 2 (Epoch 500)", color="#EF5350", edgecolor="white")

    ax.set_ylabel(metric, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=10)
    ax.legend(fontsize=10)
    ax.set_ylim(0.4, 1.05)
    ax.grid(axis="y", alpha=0.3)

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.008,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.008,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{filename}.png"), dpi=300, bbox_inches="tight")
    plt.savefig(os.path.join(output_dir, f"{filename}.pdf"), bbox_inches="tight")
    plt.close()
    print(f"  Saved: {filename}.png / .pdf")


def plot_multi_metric_heatmap(categories, phase_data, phase_label, filename, output_dir):
    """Heatmap of all metrics across categories for one phase."""
    metrics = ["Precision", "Recall", "mAP@50", "mAP@50-95"]
    data = np.array([[phase_data[c][m] for m in metrics] for c in categories])

    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(data, cmap="YlOrRd", aspect="auto", vmin=0.5, vmax=1.0)

    ax.set_xticks(np.arange(len(metrics)))
    ax.set_yticks(np.arange(len(categories)))
    ax.set_xticklabels(metrics, fontsize=10)
    ax.set_yticklabels(categories, fontsize=10)

    for i in range(len(categories)):
        for j in range(len(metrics)):
            text_color = "white" if data[i, j] > 0.85 else "black"
            ax.text(j, i, f"{data[i, j]:.3f}", ha="center", va="center",
                    fontsize=10, color=text_color, fontweight="bold")

    ax.set_title(f"{phase_label} — Validation Metrics", fontsize=13, fontweight="bold")
    fig.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{filename}.png"), dpi=300, bbox_inches="tight")
    plt.savefig(os.path.join(output_dir, f"{filename}.pdf"), bbox_inches="tight")
    plt.close()
    print(f"  Saved: {filename}.png / .pdf")


def plot_radar_chart(categories, phase1_data, phase2_data, metric, title, filename, output_dir):
    """Radar/spider chart comparing Phase 1 vs Phase 2."""
    p1_vals = [phase1_data[c][metric] for c in categories]
    p2_vals = [phase2_data[c][metric] for c in categories]
    p1_vals.append(p1_vals[0])
    p2_vals.append(p2_vals[0])

    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles.append(angles[0])

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    ax.plot(angles, p1_vals, 'o-', linewidth=2, label="Phase 1 (Epoch 200)", color="#42A5F5")
    ax.fill(angles, p1_vals, alpha=0.15, color="#42A5F5")
    ax.plot(angles, p2_vals, 'o-', linewidth=2, label="Phase 2 (Epoch 500)", color="#EF5350")
    ax.fill(angles, p2_vals, alpha=0.15, color="#EF5350")

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=10)
    ax.set_ylim(0.4, 1.0)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{filename}.png"), dpi=300, bbox_inches="tight")
    plt.savefig(os.path.join(output_dir, f"{filename}.pdf"), bbox_inches="tight")
    plt.close()
    print(f"  Saved: {filename}.png / .pdf")


if __name__ == "__main__":
    dist_cats = ["Close", "Medium", "Far", "Mixed"]
    env_cats = ["Bright/Clear", "Night/Low-light", "Cluttered/Ambiguous"]

    # ─── Distance-wise graphs ───
    print("\n=== Distance-Wise Validation Graphs ===")
    for metric in ["Precision", "Recall", "mAP@50", "mAP@50-95"]:
        safe_metric = metric.replace("@", "").replace("-", "")
        plot_grouped_bar(
            dist_cats, dist_phase1, dist_phase2, metric,
            f"Distance-Wise {metric}: Phase 1 vs Phase 2",
            f"distance_{safe_metric}", OUTPUT_DIR,
        )

    plot_multi_metric_heatmap(dist_cats, dist_phase2, "Phase 2 — Distance-Wise", "distance_heatmap_phase2", OUTPUT_DIR)
    plot_radar_chart(dist_cats, dist_phase1, dist_phase2, "mAP@50",
                     "Distance-Wise mAP@50 Comparison", "distance_radar_map50", OUTPUT_DIR)

    # ─── Environment-wise graphs ───
    print("\n=== Environment-Wise Validation Graphs ===")
    for metric in ["Precision", "Recall", "mAP@50", "mAP@50-95"]:
        safe_metric = metric.replace("@", "").replace("-", "")
        plot_grouped_bar(
            env_cats, env_phase1, env_phase2, metric,
            f"Environment-Wise {metric}: Phase 1 vs Phase 2",
            f"environment_{safe_metric}", OUTPUT_DIR,
        )

    plot_multi_metric_heatmap(env_cats, env_phase2, "Phase 2 — Environment-Wise", "environment_heatmap_phase2", OUTPUT_DIR)
    plot_radar_chart(env_cats, env_phase1, env_phase2, "mAP@50",
                     "Environment-Wise mAP@50 Comparison", "environment_radar_map50", OUTPUT_DIR)

    print(f"\nAll validation graphs saved to: {OUTPUT_DIR}")
