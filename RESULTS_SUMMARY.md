# AI-Powered Disaster Classification from Aerial Imagery

## Overview
A deep-learning system that classifies aerial/drone images into four categories —
**Earthquake, Fire, Flood, and Normal** — to support emergency response and
disaster monitoring.

## Dataset
- **Source:** AIDERv2 (Aerial Image Dataset for Emergency Response Applications)
- **Size:** 16,723 images across 4 classes, pre-split into Train / Val / Test
- **Split:** 13,399 train | 1,670 validation | 1,654 test
- **Class imbalance:** Flood (most common) vs. Earthquake (rarest) — handled with
  weighted cross-entropy loss so rare classes were not ignored.

## Method
- **Approach:** Transfer learning with a **ResNet50** backbone pretrained on ImageNet.
- **Phase 1 — Feature extraction:** froze the backbone, trained only a new 4-class
  head. Reached **94.6%** validation accuracy.
- **Phase 2 — Fine-tuning:** unfroze the deeper residual blocks (layer3, layer4) and
  retrained at a low learning rate (1e-4) to adapt ImageNet features to aerial
  disaster imagery. Improved to **97.8%** validation accuracy.
- **Regularization:** data augmentation (flips, rotation, color jitter) + best-model
  checkpointing on validation accuracy to prevent overfitting.
- **Hardware:** trained locally on an NVIDIA RTX 4060 GPU.

## Results (held-out test set, 1,654 unseen images)

**Overall test accuracy: 98.31%**

| Class      | Precision | Recall | F1-score | Support |
|------------|-----------|--------|----------|---------|
| Earthquake | 0.979     | 0.962  | 0.970    | 239     |
| Fire       | 0.993     | 0.991  | 0.992    | 436     |
| Flood      | 0.988     | 0.978  | 0.983    | 502     |
| Normal     | 0.971     | 0.992  | 0.981    | 477     |
| **Macro avg** | **0.983** | **0.981** | **0.982** | 1654 |

See `confusion_matrix.png` for the full breakdown.

## Key Findings
- **Strongest class:** Fire (F1 = 0.992) — smoke/flames are visually distinctive.
- **Weakest class:** Earthquake (recall = 0.962). Its main error mode is being
  misclassified as **Normal** (5 of 9 errors) — wide-area rubble in aerial shots can
  resemble ordinary terrain. This aligns with Earthquake being the rarest class.
- **Explainability (Grad-CAM):** heatmaps confirm the model attends to the correct
  regions — debris/rubble for earthquakes, water for floods, smoke for fire — rather
  than exploiting background shortcuts. See `gradcam.png`.

## Explainability
Grad-CAM was used to visualize which pixels drove each prediction. On earthquake
images the model correctly focused on collapsed structures and debris fields,
providing visual evidence that predictions are based on genuine disaster features.

## Limitations & Future Work
- Earthquake recall could improve with more training data for that class or targeted
  augmentation.
- Dataset images come from mixed sources (aerial + drone + web); performance on a
  specific sensor/platform should be validated separately.
- Next steps: extend to pixel-level segmentation (flood extent, burn area), add more
  disaster types, and test on imagery from a single operational drone platform.

## Reproducibility
- `evaluate.py` — loads the trained model, produces the metrics + both figures.
- `predict.py` — runs a live prediction on any image with confidence scores.
- `best_model.pth` — the trained, fine-tuned model weights.
