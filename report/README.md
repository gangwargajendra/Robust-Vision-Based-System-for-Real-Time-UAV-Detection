# Report Generation Guide

## Folder Structure
```
report/
├── main.tex                         # LaTeX report (upload to Overleaf)
├── generate_training_graphs.py      # Script to create training graphs
├── generate_validation_graphs.py    # Script to create validation graphs
├── graphs/                          # Output folder for graphs (created by scripts)
│   ├── training_losses.png/pdf
│   ├── training_losses_combined.png/pdf
│   ├── map_curves.png/pdf
│   ├── precision_recall_curves.png/pdf
│   ├── phase_comparison.png/pdf
│   ├── distance_*.png/pdf
│   ├── environment_*.png/pdf
│   └── ... (heatmaps, radar charts)
└── README.md                        # This file
```

## Step-by-Step Instructions

### Step 1: Generate Graphs (on your server)

Upload both Python scripts to your server, then run:

```bash
# Install matplotlib if not already installed
pip install matplotlib numpy

# Generate training graphs
python generate_training_graphs.py

# Generate validation comparison graphs
python generate_validation_graphs.py
```

This will create a `graphs/` folder with all PNG and PDF graph files.

### Step 2: Set Up Overleaf

1. Go to [Overleaf](https://www.overleaf.com) and create a **New Project** → **Blank Project**
2. Replace the default `main.tex` with the content from `report/main.tex`
3. Create a folder named `graphs/` in Overleaf
4. Upload all generated `.png` files from `report/graphs/` into the `graphs/` folder

### Step 3: Enable Graph Figures

In `main.tex`, find the commented-out `\includegraphics` lines and **uncomment** them:
- Search for `% \begin{figure}` and `% \includegraphics`
- Remove the `%` comment markers from those blocks

### Step 4: Customize Your Details

Search for `TODO` in the LaTeX file and fill in:
- Your full name
- Roll number
- Supervisor name
- University logo (optional — upload as `university_logo.png`)

### Step 5: Compile

Click the **Recompile** button in Overleaf. The PDF report will be generated automatically.

## Graphs Generated

### Training Graphs (generate_training_graphs.py)
| File | Description |
|------|-------------|
| `training_losses.png` | Box, Cls, DFL losses (3 subplots) |
| `training_losses_combined.png` | All losses on one chart |
| `map_curves.png` | mAP@50 and mAP@50-95 over epochs |
| `precision_recall_curves.png` | Precision and Recall curves |
| `phase_comparison.png` | Phase 1 vs Phase 2 bar chart |

### Validation Graphs (generate_validation_graphs.py)
| File | Description |
|------|-------------|
| `distance_Precision.png` | Distance-wise Precision comparison |
| `distance_Recall.png` | Distance-wise Recall comparison |
| `distance_mAP50.png` | Distance-wise mAP@50 comparison |
| `distance_mAP5095.png` | Distance-wise mAP@50-95 comparison |
| `distance_heatmap_phase2.png` | Heatmap of all distance metrics |
| `distance_radar_map50.png` | Radar chart for distance mAP@50 |
| `environment_Precision.png` | Environment-wise Precision comparison |
| `environment_Recall.png` | Environment-wise Recall comparison |
| `environment_mAP50.png` | Environment-wise mAP@50 comparison |
| `environment_mAP5095.png` | Environment-wise mAP@50-95 comparison |
| `environment_heatmap_phase2.png` | Heatmap of all environment metrics |
| `environment_radar_map50.png` | Radar chart for environment mAP@50 |
