# End-Sem Presentation — 3-Slide Content

**Project:** Vision-based Approach for Robust UAV Detection
**Name:** Gajendra Gangwar  **Roll:** 2301081  **Supervisor:** Dr. Rakesh Matam

> Use `BTechPresentation-EndsemTemplate.pptx` (3 slides).
> Tables to paste from `tables/table pdf/` folder (already compiled).

---

## SLIDE 1 — Title: *Vision-based Approach for Robust UAV Detection*

### • Introduction:
- UAVs (drones) are widely used in surveillance, delivery, mapping and disaster monitoring, but their misuse near airports, defense areas and public events is a growing security threat.
- This project aims to build a reliable visual UAV detector that works on **small, distant drones** across **different backgrounds** (open sky, urban, forest).
- This project evaluates modern deep learning-based detection models and trains **YOLO11-nano** — a recent, lightweight single-stage detector — to improve detection accuracy across varied distances and environments.

### • Motivation:
- Security agencies need a system that can **quickly and accurately identify UAVs** to enable timely action against potential threats.
- Drones are hard to detect because they often appear as **just a few pixels** in the image and can look similar to birds or kites.
- A **lighter, single-stage detector** is preferred over a heavier two-stage pipeline for faster inference and future edge deployment.

---

## SLIDE 2 — Existing Work / Literature Review

### • Existing work:
- The existing in-house pipeline used **YOLOv9** to detect drone regions and **EfficientNet-B3** to re-classify the detections, reducing false positives.
- The model was trained for **20 epochs** on the same drone/non-drone dataset used in this project.

### • Performance of the existing model:

**Distance-wise confidence scores:**
→ *Paste table from* `table5_distance_comparison` *(table pdf)*

**Environment-wise confidence scores:**
→ *Paste table from* `table6_environment_comparison` *(table pdf)*

### • Gaps in the existing work:
- Accuracy **drops sharply on very-far / tiny drones** (confidence as low as 0.70).
- Performance **degrades in cluttered backgrounds** (forests, urban) with confidence only 0.70–0.80.
- The **two-stage pipeline** (detector + classifier) is heavy (~25M+ params) and not ideal for real-time or edge deployment.

### • Model selection:
- Six modern detectors were compared: YOLOv9, YOLO11, RT-DETR, YOLO26, NSSRD, ED-Net.
→ *Paste table from* `table1_model_compare` *(table pdf)*
- **YOLO11-nano** was selected — smallest (~2.6M params), fastest, C2PSA attention for small objects, and the most mature tooling.

---

## SLIDE 3 — Work Done Till Date

### • Proposed approach:
- **YOLO11-nano** was selected because it has a dedicated **C2PSA attention block** for small targets, uses an **anchor-free head with DFL loss** for tighter box localisation, and is the lightest model (~2.6M params) with the most stable tooling among all six candidates studied.
- Built **two custom validation splits** for detailed evaluation:
  - **Distance-wise:** Close / Mid / Long / Multi-range.
  - **Environment-wise:** Bright clear / Cluttered / Low-light.
- Key settings: image 320×320, batch 2, AdamW optimiser, patience 30 epochs.

### • Results — overall comparison with existing model:
→ *Paste table from* `table2_overall_comparison` *(table pdf)*
- Same mAP@50, but **+2.4 pts precision** and **+5.1 pts mAP@50–95** — tighter, better-localised boxes.
- All three training losses (box, cls, DFL) are significantly lower.

### • Results — per-scenario (YOLO11-nano, Phase 2):

**Distance-wise results:**
→ *Paste table from* `table3_distance_results` *(table pdf)*

**Environment-wise results:**
→ *Paste table from* `table4_environment_results` *(table pdf)*

### • Key takeaways:
- YOLO11-nano is **~10× lighter** (2.6M vs 25M+ params) and achieves better precision and localisation.
- Biggest improvement is exactly where the existing model was weakest:
  - **Long-range drones:** confidence 0.70–0.78 → mAP@50 **0.9066**.
  - **Cluttered backgrounds:** confidence 0.70–0.80 → mAP@50 **0.9134**.

---

# Tables to paste (from `tables/table pdf/`)

| Table file | Goes on | Purpose |
|---|---|---|
| `table5_distance_comparison` | **Slide 2** | Existing model's distance-wise confidence |
| `table6_environment_comparison` | **Slide 2** | Existing model's environment-wise confidence |
| `table1_model_compare` | **Slide 2** | 6-detector comparison |
| `table2_overall_comparison` | **Slide 3** | Overall: existing vs YOLO11-nano (P, R, mAP, losses) |
| `table3_distance_results` | **Slide 3** | YOLO11-nano distance-wise (P, R, mAP@50) |
| `table4_environment_results` | **Slide 3** | YOLO11-nano environment-wise (P, R, mAP@50) |

# Optional images (from `report/graphs/`)

- `training_losses_combined.png` — can paste on Slide 3 if space.
- `distance_heatmap_phase2.png`, `environment_heatmap_phase2.png` — visual support for Slide 3.
