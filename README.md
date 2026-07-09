# Robust Vision-Based System for Real-Time UAV Detection 🚁

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![YOLO11](https://img.shields.io/badge/YOLO-11-yellow.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-Framework-ee4c2c.svg)

> **Objective:** To understand and evaluate how well lightweight models detect small Unmanned Aerial Vehicles (UAVs) in realistic, challenging conditions—such as long-range views, cluttered backgrounds, and low-light scenes.

This repository contains the implementation, split-analysis scripts, training-resume workflow, and report assets for drone detection using **YOLO11-nano**. The project trains a lightweight detector on an in-house dataset and evaluates it by dynamically breaking the validation set into highly specific distance-wise and environment-wise subsets.

---

## 📊 Performance Metrics

The workflow trains YOLO11-nano in Phase 1 (Epochs 1-200) and resumes in Phase 2 (Epochs 201-500). Compared to earlier YOLOv9 + EfficientNet-B3 pipelines on the same dataset, this implementation significantly improves precision and localization quality—especially in difficult subsets.

**Final YOLO11-nano metrics on the full validation set:**

| Metric | Score |
| :--- | :--- |
| **Precision** | 0.9622 |
| **Recall** | 0.8662 |
| **mAP@50** | 0.9305 |
| **mAP@50-95** | 0.6727 |

---

## 📂 Repository Structure

* **`SECOND/files/`**: Resume-training workflow for Phase 2 (epochs 201 to 500).
* **`SECOND/results/`**: Phase 2 training logs and resumed checkpoint artifacts.
* **`distance_wise/files/`**: Scripts to analyze the validation set by bounding-box size, splitting it into subsets (close, medium, far, mixed, empty), and running subset validation.
* **`distance_wise/results/`**: Saved validation summaries and per-split metric files for distance analysis.
* **`environment-wise/files/`**: Scripts to classify validation images (bright-clear, night-lowlight, cluttered-ambiguous), split the data, and run validation.
* **`environment-wise/results/`**: Saved validation summaries and per-split metric files for environmental analysis.
* **`report/`**: LaTeX report source, graph-generation scripts, presentation tables, and exported figures.

---

## 🗂️ Dataset & Splits

The codebase expects a standard YOLO-format dataset containing a `data.yaml` file pointing to the dataset root and defining the `train` and `val` folders. 

We utilize two custom validation taxonomies to stress-test the model:

1. **Distance-wise:** Close-range, mid-range, long-range, multi-range, and empty.
2. **Environment-wise:** Bright-clear, night-lowlight, cluttered-ambiguous.

*Note: Analysis scripts automatically write image-stem lists into intermediate folders. Split scripts then copy/link matching images and labels into new split folders under the dataset root.*

---

## 🚀 Getting Started (Reproducing the Workflow)

The scripts are designed to be run from their own directories but resolve paths relative to the repository root. 

### 1. Prepare the Dataset Config
Ensure `data.yaml` exists at the repository root or update the path constants in the scripts. The YAML must define: `path`, `train`, `val`, `nc`, and `names`.

### 2. Resume Phase 2 Training
Run the resume script to continue training from an existing checkpoint. 
```bash
# Via Python directly
python SECOND/files/train_model_resume_201_to_500.py

# OR via the shell launcher (Unix-like environments)
bash SECOND/files/run_resume_training.sh