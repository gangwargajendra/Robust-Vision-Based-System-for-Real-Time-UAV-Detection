# End-Sem Presentation — Content

**Project:** Vision-based Approach for Robust UAV Detection (Drone Detection Using YOLO11)
**Name:** Gajendra Gangwar  **Roll:** 2301081  **Supervisor:** Dr. Rakesh Matam

> Use the IIITG end-sem template (`BTechPresentation-EndsemTemplate.pptx`).
> Follow the same colour palette / layout as the mid-sem deck. Keep the
> same Name / Roll / Supervisor footer that appears in the template.

---

## Slide 1 — Title slide

**Title:** Vision-based Approach for Robust UAV Detection
**Subtitle (smaller line):** Drone detection using YOLO11 — final results
**Bottom block:**
- Gajendra Gangwar (Roll No.: 2301081)
- Supervisor: Dr. Rakesh Matam
- Department of CSE, IIIT Guwahati
- April 2026

---

## Slide 2 — Introduction & Motivation

**Project Title:** Vision-based Approach for Robust UAV Detection

- **Introduction:**
  - UAVs (drones) are now widely used for surveillance, delivery, mapping and disaster monitoring.
  - At the same time, drone misuse near airports, defense areas and public events is a growing security threat.
  - This project studies modern deep-learning detectors and builds a UAV detector that works at long ranges and in real-world backgrounds.

- **Motivation:**
  - Security agencies need a system that can detect drones quickly and accurately to enable timely action.
  - Drones are hard to detect because they are very small in the image and often appear against complex backgrounds (sky, trees, buildings).
  - The existing in-house pipeline (YOLOv9 + EfficientNet-B3) loses accuracy on far-away drones and in cluttered scenes — this needs improvement.

---

## Slide 3 — Existing Work / Literature Review

- This project continues the work of a senior student (Madhubani Basu) who built the dataset and trained a UAV detector with **YOLOv9 + EfficientNet-B3 classifier**.
- YOLOv9 was used to detect the drone region; EfficientNet-B3 then re-classified the crop to reduce false detections.
- A short comparative study was done on **six** modern detectors:
  - YOLOv9, YOLO11, RT-DETR, YOLO26, NSSRD, ED-Net.
- The goal of the study was to pick a model that handles **small / distant drones** better than the existing pipeline, while still being light enough for the lab GPU.

**Gaps in the existing solution:**
- Performs well on close drones but degrades sharply on **very-far / tiny** UAVs.
- Confidence drops in **cluttered backgrounds** (forests, urban scenes).
- The two-stage (detector + classifier) pipeline is heavier and slower than a single-stage detector.

---

## Slide 4 — Comparison of Candidate Detectors

Use this table on the slide (clean, 6 rows). LaTeX source is in
`tables/model_compare.tex` if you want to render and screenshot it.

| Model | Params (M) | Speed | Small-object detection | Tooling maturity | Key feature |
|---|---|---|---|---|---|
| YOLOv9 | ~25 | High | Medium | High | PGI + GELAN backbone |
| **YOLO11-nano** | **~2.6** | **Very high** | **High** | **Very high** | **C2PSA attention block** |
| RT-DETR | ~32 | Medium | High | Medium | Transformer, no NMS |
| YOLO26 | ~3 | Very high | Very high | Low (new) | End-to-end CPU inference |
| NSSRD | Hardware-tuned | Medium | Very high | Low | NAS backbone, coarse-to-fine |
| ED-Net | ~5 | High | Very high | Low | Dedicated XSmall head |

**One-line takeaway (put as a callout below the table):**
> YOLO11-nano gives the best balance of accuracy on small targets, speed,
> low compute and stable open-source tooling — the other strong models are
> still too immature to rely on.

---

## Slide 5 — Why YOLO11-Nano was Chosen

Pick **3 strongest** points only (the rest are in the report):

- **C2PSA attention** is built specifically to focus on small targets against an empty background — exactly the drone-on-sky case.
- **Anchor-free head + DFL loss** → tighter, better-localised boxes on tiny drones (improves mAP@50–95).
- **Very low compute** (`yolo11n` ~2.6M params) and very mature **Ultralytics tooling** (clean train / resume / validate workflow).

---

## Slide 6 — Dataset

- In-house drone / non-drone dataset originally prepared by Madhubani Basu (re-used for fair comparison).
- Standard YOLO format: each label line = `class x_center y_center w h` (normalised to `[0,1]`).
- 2 classes: `0 = non-drone`, `1 = drone`.

**Two custom validation splits** were built to evaluate the model honestly:

| Split | Categories | # Images |
|---|---|---|
| Distance-wise | Close / Mid / Long / Multi-range | 460 / 521 / 1471 / 86 |
| Environment-wise | Bright clear / Cluttered / Low-light | 682 / 1273 / 584 |

> Tip: under the table, paste the 4 distance sample images in one row, and
> below them the 3 environment sample images in one row, taken from
> `report/graphs/`.

---

## Slide 7 — Distance & Environment Split (visual)

Layout:
- **Top half:** 4 thumbnails — `sample_pure_close.jpg`, `sample_pure_medium.jpg`, `sample_pure_far.jpg`, `sample_mixed.jpg` with labels Close / Mid / Long / Multi.
- **Bottom half:** 3 thumbnails — `sample_bright_clear.jpg`, `sample_cluttered.jpg`, `sample_lowlight.jpg` with labels Bright / Cluttered / Low-light.

One-line caption at the bottom:
> The same trained model is evaluated separately on each subset, so we can
> see *where* it is strong and *where* it is weak.

---

## Slide 8 — Methodology (1) — Pipeline

Insert the YOLO11 pipeline diagram. The TikZ source is already in
`report/main.tex` — you can compile it standalone from
`figures/yolo_pipeline.tex` (file added in this folder).

Bullet points beside / under the diagram:
- **Input → Preprocess (320×320)** → image is resized & normalised.
- **Backbone (C3k2)** → fast multi-scale feature extraction.
- **C2PSA attention** → focuses on small drone-like regions.
- **Neck (PAN-FPN)** → fuses small + large object features.
- **Anchor-free head + DFL** → predicts box, class, refined edges.
- **NMS** → removes duplicate boxes → final detections.

---

## Slide 9 — Methodology (2) — Two-Phase Training

| Setting | Value |
|---|---|
| Model | YOLO11-nano (`yolo11n`) |
| Image size | 320 × 320 |
| Batch size | 2 |
| Optimiser | AdamW (Ultralytics default) |
| Patience | 30 epochs |
| Total epochs | **500** (Phase 1: 1–200, Phase 2: 201–500) |

**Why two phases?**
- Phase 1 trained for 200 epochs from `yolo11n` pretrained weights.
- Around epoch 180 the loss was still going down → the model had not converged.
- Resumed from `last.pt` and trained for 300 more epochs (Phase 2).

---

## Slide 10 — Training Curves

Insert two graphs side by side from `report/graphs/`:
- `training_losses_combined.png` (Box / Cls / DFL loss over 500 epochs)
- `map_curves.png` (mAP@50 and mAP@50–95 over 500 epochs)

One-line caption under each:
- Loss → smooth decrease, small expected jump at epoch 201 (Phase 2 reset), then resumes downward.
- mAP → fast rise in first 50 epochs, gradual climb after; Phase 2 still adds ~1% mAP@50 and ~1.5% mAP@50–95.

---

## Slide 11 — Final Results (overall)

**YOLO11-nano, full validation set (epoch 500):**

| Metric | Value |
|---|---|
| Precision | **0.9622** |
| Recall | **0.8662** |
| mAP@50 | **0.9305** |
| mAP@50–95 | **0.6727** |

**Phase 1 vs Phase 2 (final epoch of each):**

| Metric | Phase 1 | Phase 2 | Change |
|---|---|---|---|
| Precision | 0.9454 | 0.9622 | +1.78% |
| Recall | 0.8611 | 0.8662 | +0.59% |
| mAP@50 | 0.9220 | 0.9305 | +0.92% |
| mAP@50–95 | 0.6578 | 0.6727 | +2.27% |

> Phase 2 mainly tightens the predicted boxes (precision and mAP@50–95 gain the most).

---

## Slide 12 — Distance-Wise Results

**Phase 2 (epoch 500):**

| Subset | P | R | mAP@50 | mAP@50–95 |
|---|---|---|---|---|
| Close-range | 0.9533 | 0.8852 | 0.9534 | 0.8322 |
| **Mid-range** | **0.9843** | **0.9690** | **0.9889** | **0.8003** |
| Long-range | 0.9496 | 0.8479 | 0.9066 | 0.5502 |
| Multi-range | 0.9579 | 0.8810 | 0.9273 | 0.7249 |

**Key observations:**
- Mid-range is the strongest subset — drones are large enough to detect easily, small enough for a 320×320 input.
- Long-range gains the most from Phase 2 training (+3.21% mAP@50).

> Optional: also paste `distance_heatmap_phase2.png` from `report/graphs/`.

---

## Slide 13 — Environment-Wise Results

**Phase 2 (epoch 500):**

| Environment | P | R | mAP@50 | mAP@50–95 |
|---|---|---|---|---|
| Bright / clear sky | 0.9664 | 0.8943 | 0.9486 | 0.7331 |
| **Night / low-light** | **0.9695** | 0.8840 | **0.9600** | 0.6583 |
| Cluttered | 0.9527 | 0.8425 | 0.9134 | 0.6087 |

**Key observations:**
- Low-light is surprisingly strong — drones often carry small lights or contrast strongly with dark backgrounds.
- Cluttered remains the hardest case (busy backgrounds → more false positives).

> Optional: also paste `environment_heatmap_phase2.png` from `report/graphs/`.

---

## Slide 14 — Comparison with Existing Pipeline (Overall)

Existing model = YOLOv9 + EfficientNet-B3 (senior's pipeline, same dataset).

| Metric | Existing (epoch 20) | **This work (epoch 500)** |
|---|---|---|
| Precision | 0.9383 | **0.9622** |
| Recall | **0.8953** | 0.8662 |
| mAP@50 | 0.9307 | 0.9305 |
| mAP@50–95 | 0.6214 | **0.6727** |
| Box loss (train) | 0.9760 | **0.7614** |
| Cls loss | 0.6431 | **0.3774** |
| DFL loss | 1.2123 | **0.8826** |

> Same mAP@50, but much better precision (+2.4 pts) and mAP@50–95 (+5.1 pts) — boxes are tighter and better localised.

---

## Slide 15 — Comparison: Distance & Environment

The existing model only reports a **confidence-score range** (model's own
probability), not P / R / mAP. So the comparison is qualitative.

**Distance-wise:**

| Subset | Existing (confidence) | This work (mAP@50) |
|---|---|---|
| Close-range | 0.93 – 1.00 | 0.9534 |
| Mid-range | 0.90 – 0.96 | **0.9889** |
| Long-range (very far) | 0.70 – 0.78 | **0.9066** |

**Environment-wise:**

| Environment | Existing (confidence) | This work (mAP@50) |
|---|---|---|
| Bright / clear | ~1.00 | 0.9486 |
| Night / low-light | 0.85 – 0.98 | **0.9600** |
| Cluttered | 0.70 – 0.80 | **0.9134** |

> Biggest wins: **long-range** drones and **cluttered** backgrounds — exactly the cases where the existing model was weak.

---

## Slide 16 — Why YOLO11-nano Beats the Existing Model

Short, 4 bullets only:

- **Architecture:** C2PSA attention is built for small targets → big gain on long-range drones.
- **Training:** 500 epochs in two phases vs 20 epochs → lower box / cls / DFL loss on every component.
- **Evaluation:** distance + environment splits expose where the model is weak (the existing pipeline never measured this).
- **Lighter model:** `yolo11n` (~2.6M params) is much smaller than YOLOv9 + EfficientNet-B3 → easier to deploy on edge devices later.

---

## Slide 17 — Conclusion

- Out of six candidate detectors, **YOLO11-nano** was selected and trained on the same in-house dataset.
- Two-phase training (200 + 300 epochs) reached:
  - **P = 0.962, R = 0.866, mAP@50 = 0.931, mAP@50–95 = 0.673** on the full validation set.
- Performance stays strong across distance subsets (close → mid → long → multi) and environment subsets (clear → low-light → cluttered).
- Clear improvement over the existing YOLOv9 + EfficientNet-B3 pipeline, especially on the hardest cases (long-range, cluttered).

---

## Slide 18 — Future Work

- **Higher input resolution** (320 → 640) → should help the smallest, most distant drones.
- **Stronger augmentation** (mosaic, mixup) targeted at the cluttered subset → fewer false positives in busy backgrounds.
- **Bigger, more diverse dataset** (rain, fog, snow; more drone shapes) → real-world robustness.
- **Try a larger YOLO11 variant** (`yolo11s` / `yolo11m`) → see how much accuracy is left on the table.
- **Hard negatives** (birds, kites, balloons) during training → directly attacks the most common false-positive class.

---

## Slide 19 — References (use the same list as the report)

- J. Redmon et al., *You Only Look Once*, CVPR 2016.
- C.-Y. Wang et al., *YOLOv9 — Programmable Gradient Information*, arXiv:2402.13616, 2024.
- Ultralytics, *YOLO11 documentation*, https://docs.ultralytics.com/models/yolo11/, 2024.
- Y. Zhao et al., *DETRs Beat YOLOs on Real-Time Object Detection (RT-DETR)*, arXiv:2304.08069, 2023.
- M. Tan & Q. Le, *EfficientNet*, ICML 2019.
- T.-Y. Lin et al., *Microsoft COCO*, ECCV 2014.
- M. Basu, *Drone Detection Using YOLOv9 with EfficientNet-B3 Backbone*, B.Tech Project Report, IIIT Guwahati, 2025.

---

## Slide 20 — Thank You / Q&A

Big centred text:
> **Thank You**
> Questions?

Footer remains: Name | Roll No | Supervisor.

---

# Notes for assembly

- Stick to the template's **header colour bar** and **bottom Name/Roll/Supervisor footer** on every slide.
- Use **one big idea per slide** — do not pile content. If a slide gets crowded, split it.
- For images, paste from `report/graphs/`:
  - `training_losses_combined.png`, `map_curves.png`, `precision_recall_curves.png`
  - `phase_comparison.png`
  - `distance_heatmap_phase2.png`, `environment_heatmap_phase2.png`
  - sample images: `sample_pure_close/medium/far.jpg`, `sample_mixed.jpg`,
    `sample_bright_clear.jpg`, `sample_cluttered.jpg`, `sample_lowlight.jpg`
- For the YOLO11 pipeline diagram, see `figures/yolo_pipeline.tex` (compile with `pdflatex` and screenshot the PDF, or render on Overleaf and export PNG).
- For the model-comparison table screenshot, see `tables/model_compare.tex`.
