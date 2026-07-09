# Robust Vision-Based System for Real-Time UAV Detection

This repository contains the implementation, split-analysis scripts, training-resume workflow, and report assets for a B.Tech Semester 6 project on drone detection using YOLO11-nano. The project trains a lightweight detector on an in-house drone/non-drone dataset and evaluates it beyond a single overall validation score by breaking the validation set into distance-wise and environment-wise subsets.

The main goal is to understand how well the model detects small UAVs in realistic conditions such as long-range views, cluttered backgrounds, and low-light scenes.

## What This Repo Contains

- `SECOND/files/` - resume-training workflow for Phase 2, continuing YOLO11 training from epoch 201 to 500.
- `distance_wise/files/` - scripts that analyse the validation set by bounding-box size, split it into close/medium/far/mixed/empty subsets, and run validation on each subset.
- `environment-wise/files/` - scripts that classify validation images into bright-clear, night-lowlight, and cluttered-ambiguous subsets, split the data, and run validation.
- `report/` - LaTeX report source, graph-generation scripts, presentation tables, and exported figures.
- `distance_wise/results/` and `environment-wise/results/` - saved validation summaries and per-split metric files.
- `SECOND/results/` - Phase 2 training logs and resumed checkpoint artifacts.

## Project Summary

The workflow used in this repository is:

1. Train YOLO11-nano in Phase 1 for epochs 1-200.
2. Resume training in Phase 2 for epochs 201-500.
3. Analyse the validation set into distance-based and environment-based subsets.
4. Create split-specific YAML files for each subset.
5. Run Ultralytics validation on every split and save metrics to text files.
6. Generate plots and tables for the report and presentation.

The final reported YOLO11-nano metrics on the full validation set are:

- Precision: 0.9622
- Recall: 0.8662
- mAP@50: 0.9305
- mAP@50-95: 0.6727

Compared with the earlier YOLOv9 + EfficientNet-B3 pipeline on the same dataset, this work improves precision and localisation quality, especially in the harder distance-wise and environment-wise subsets.

## Dataset And Splits

The code expects a YOLO-format dataset with a `data.yaml` file that points to the dataset root and defines the train and validation folders.

Two custom validation taxonomies are used:

- Distance-wise: close-range, mid-range, long-range, multi-range, and empty.
- Environment-wise: bright-clear, night-lowlight, cluttered-ambiguous.

The analysis scripts write image-stem lists into the corresponding `intermediate/` folders, then the split scripts copy or link the matching images and labels into new split folders under the dataset root.

## Main Scripts

### Phase 2 resume training

`SECOND/files/train_model_resume_201_to_500.py`

- Resumes YOLO11 training from an existing checkpoint.
- Logs epoch-wise metrics to `SECOND/results/epoch_progress_201_to_500.txt`.
- Writes the training log to `SECOND/results/training_resume_201_to_500.txt`.
- Can patch a finished checkpoint if Ultralytics reports that there is nothing to resume.

`SECOND/files/run_resume_training.sh`

- Shell launcher that finds a GPU, sets `CUDA_VISIBLE_DEVICES`, and starts the Python resume script.

### Distance-wise validation

`distance_wise/files/analyze_val_distance.py`

- Reads validation labels and classifies each image by bounding-box size.
- Produces `val_images_pure_close.txt`, `val_images_pure_medium.txt`, `val_images_pure_far.txt`, `val_images_mixed.txt`, and `val_images_empty.txt`.

`distance_wise/files/split_val_by_distance_sets.py`

- Copies or symlinks the validation images and labels into `distance-wise/images/` and `distance-wise/labels/` subsets.
- Writes `data_val_close.yaml`, `data_val_medium.yaml`, `data_val_far.yaml`, `data_val_mixed.yaml`, and `data_val_empty.yaml`.

`distance_wise/files/run_split_distance_validation.py`

- Runs Ultralytics validation on each distance split and saves per-split metrics plus a combined summary.

### Environment-wise validation

`environment-wise/files/analyze_val_environment.py`

- Classifies validation images into bright-clear, night-lowlight, and cluttered-ambiguous buckets using GPU-accelerated image statistics.
- Writes the corresponding image-stem lists and an analysis summary into `environment-wise/intermediate/`.

`environment-wise/files/split_val_by_environment_sets.py`

- Builds the split dataset folders under `environment-wise/` and writes split-specific YAML files.

`environment-wise/files/run_split_environment_validation.py`

- Runs Ultralytics validation on each environment split and saves per-split metrics plus a combined summary.

### Report generation

`report/generate_training_graphs.py`

- Merges Phase 1 and Phase 2 training logs and creates training-loss, mAP, precision/recall, and phase-comparison plots.

`report/generate_validation_graphs.py`

- Uses the recorded split metrics to generate distance-wise and environment-wise comparison graphs, heatmaps, and radar charts.

`report/main.tex`

- Full LaTeX report source for the project write-up.

## Report Assets

The `report/graphs/` folder contains the generated figures used by the report and presentation, including:

- training loss curves
- mAP curves
- precision/recall curves
- phase comparison charts
- distance-wise and environment-wise heatmaps
- sample validation images for each split

The `report/presentation/` folder contains the slide content notes and compiled tables used for the end-sem presentation.

## Reproducing The Workflow

The scripts are designed to be run from their own folders, but they all resolve paths relative to the repository layout.

### 1. Prepare the dataset config

Make sure `data.yaml` exists at the repo root or update the path constants in the scripts so they point to your dataset config. The YAML should define at least:

- `path`
- `train`
- `val`
- `nc`
- `names`

### 2. Resume Phase 2 training

Run the resume script from `SECOND/files/` or use the shell launcher if you are on a Unix-like environment:

```bash
python SECOND/files/train_model_resume_201_to_500.py
```

or

```bash
bash SECOND/files/run_resume_training.sh
```

### 3. Build the distance-wise split

```bash
python distance_wise/files/analyze_val_distance.py
python distance_wise/files/split_val_by_distance_sets.py
python distance_wise/files/run_split_distance_validation.py --model path/to/best.pt
```

### 4. Build the environment-wise split

```bash
python environment-wise/files/analyze_val_environment.py
python environment-wise/files/split_val_by_environment_sets.py
python environment-wise/files/run_split_environment_validation.py --model path/to/best.pt
```

### 5. Generate report figures

```bash
python report/generate_training_graphs.py
python report/generate_validation_graphs.py
```

## Generated Outputs

Key output locations are:

- `distance_wise/intermediate/` and `environment-wise/intermediate/` for image-stem lists and split YAMLs.
- `distance_wise/results/` and `environment-wise/results/` for validation metric text files and summaries.
- `SECOND/results/` for resume-training logs and checkpoint artifacts.
- `report/graphs/` for PNG/PDF figures used in the report.

## Notes

- The training scripts assume Ultralytics YOLO checkpoints and a valid `data.yaml` path.
- Some paths in the scripts are written for the original machine layout and may need adjustment if your dataset or checkpoint locations differ.
- The environment-wise analysis script uses PyTorch and torchvision image loading, and will run faster on a CUDA-capable GPU.

## Report And Presentation

If you want the written report or slides, start with:

- `report/main.tex`
- `report/README.md`
- `report/presentation/endsem_content.md`
- `report/presentation/endsem_content_3slides.md`

These files describe the final narrative, model comparison, split metrics, and the presentation tables that match the generated figures.