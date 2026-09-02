# Architecture — Oral Disease Detection Model (v2)

## Overview

This document defines the end-to-end architecture for the v2 rebuild: dataset pipeline,
model selection, training strategy, evaluation, and deployment plan. All decisions made
here should be reflected in code as it is written.

---

## 1. Model — YOLOv11 (Ultralytics)

**Choice:** YOLOv11n / YOLOv11s (start small, scale up if needed)

**Why YOLOv11 over Faster R-CNN (v1):**
- Pretrained on COCO — strong general visual features out of the box
- Native YOLO-format support matches our Roboflow data sources directly
- Built-in augmentations: Mosaic, Mixup, copy-paste, HSV jitter, flips — no custom code needed
- Single-file ONNX export for web deployment
- Significantly faster inference (~10ms vs ~100ms per image)

**Pretrained weights:** `yolo11s.pt` (start), upgrade to `yolo11m.pt` if dataset > 10K images

---

## 2. Disease Classes

6 canonical classes. Every dataset source must be explicitly remapped to these IDs.
No implicit trust of source label integers.

| ID | Class Name          | Notes                                                    |
|----|---------------------|----------------------------------------------------------|
| 0  | caries              | Tooth decay / cavities                                   |
| 1  | gingivitis          | Gum inflammation                                         |
| 2  | tooth_discoloration | Staining / discoloration                                 |
| 3  | ulcer               | Includes oral lesions and xerostomia                     |
| 4  | calculus            | Tartar buildup; also covers plaque (soft bacterial film) |
| 5  | hypodontia          | Missing teeth                                            |

**Why plaque was merged into calculus:** Plaque is nearly invisible in standard clinical
photographs without disclosing agents, making it unreliably annotatable from images alone.
Calculus is mineralized plaque — visually distinct and well-represented in available data.
The meaningful signal for a photo-based model is "bacterial deposit present." The
plaque vs. calculus distinction requires tactile examination and is out of scope here.

These IDs are defined in `configs/dataset.yaml` and are the single source of truth.

---

## 3. Dataset Pipeline

Each script in `scripts/` corresponds to one stage. Run them in numbered order.
All intermediate and final data lives under `data/` (gitignored).

```
data/
├── sources/          # Raw zips / extracted files, one subfolder per source
│   ├── roboflow_oral_detector/
│   ├── zenodo_caries/
│   ├── roboflow_healthy/
│   └── <any new source>/
└── processed/        # Final merged dataset ready for training
    ├── images/
    │   ├── train/
    │   ├── val/
    │   └── test/
    └── labels/
        ├── train/
        ├── val/
        └── test/
```

### Stage 1 — `scripts/1_remap_classes.py`
For each source, applies a hardcoded source-specific mapping from that source's label
integers (or names) to our canonical IDs above. Writes cleaned labels in-place.
Any annotation that cannot be mapped is logged and discarded.

Each source gets its own mapping block in the script — no guessing, no implicit trust.

### Stage 2 — `scripts/2_merge_datasets.py`
Copies all remapped images and labels from `data/sources/*/` into a single flat
`data/merged/` directory. Detects and skips duplicate filenames (hashed comparison).

### Stage 3 — `scripts/3_split_dataset.py`
Performs a **stratified** train/val/test split (70/15/15) at the image level.
Stratification is approximate — groups images by their dominant class and splits
each group independently to maintain class balance across splits.
Writes final images and labels into `data/processed/`.

### Stage 4 — `scripts/4_verify_dataset.py`
Sanity checks before training:
- Confirms every image has a corresponding label file (and vice versa)
- Reports per-class counts in each split
- Flags extreme class imbalance (> 10:1 ratio between any two classes)
- Confirms all label IDs are within [0, 6]

---

## 4. Configuration Files

### `configs/dataset.yaml`
Standard Ultralytics dataset config. Defines paths and class names.
```yaml
path: ../data/processed
train: images/train
val: images/val
test: images/test

nc: 6
names:
  0: caries
  1: gingivitis
  2: tooth_discoloration
  3: ulcer
  4: calculus
  5: hypodontia
```

### `configs/train.yaml`
Training hyperparameters. Key settings:
```yaml
model: yolo11s.pt
epochs: 100
patience: 15          # early stopping
batch: 16
imgsz: 640
optimizer: AdamW
lr0: 0.001
lrf: 0.01             # final LR = lr0 * lrf
momentum: 0.937
weight_decay: 0.0005
mixup: 0.15           # MDA-style mixing
mosaic: 1.0
copy_paste: 0.1
degrees: 10.0
hsv_h: 0.015
hsv_s: 0.7
hsv_v: 0.4
flipud: 0.1
fliplr: 0.5
```

---

## 5. Training

**Script:** `src/train.py`

Key decisions:
- Freeze backbone for first 10 epochs, then unfreeze (fine-tuning via transfer learning)
- Validate every epoch; save best checkpoint by `mAP@50:95`
- Use AdamW (not SGD) — better convergence for fine-tuning transformers and modern CNNs
- Early stopping with patience=15

**HPC:** `hpc/train_job.sh` — configured for Oregon State SLURM with GPU partition

---

## 6. Evaluation

**Script:** `src/evaluate.py`

Metrics (via `torchmetrics.detection.MeanAveragePrecision`):
- `mAP@50` — primary metric for detection quality
- `mAP@50:95` — COCO standard, stricter
- Per-class AP for all 7 classes
- Precision / Recall per class

Results saved to `logs/evaluation_results.json`.

---

## 7. Inference

**Script:** `src/infer.py`

Takes a single image path, runs the trained model, and returns/displays annotated
bounding boxes with class names and confidence scores. Used for manual testing before
deployment.

---

## 8. Export for Deployment

**Script:** `src/export.py`

Exports the best `.pt` checkpoint to ONNX format:
```
models/oral_disease_v2_best.onnx
```

ONNX enables serving without a PyTorch dependency on the server. Target runtime:
ONNX Runtime (Python) in a FastAPI backend.

---

## 9. Deployment Plan (Future)

Architecture once training is complete:

```
Browser (React frontend)
    │
    ▼
FastAPI backend
    ├── POST /predict   — accepts image, returns boxes + labels + scores
    └── GET  /health    — liveness check

ONNX Runtime (loads oral_disease_v2_best.onnx at startup)
```

Hosting options (in order of preference):
1. **Hugging Face Spaces** — free GPU, Gradio or FastAPI, one-command deploy
2. **Render / Railway** — simple Docker deploy, free tier available
3. **AWS EC2 / GCP** — full control, more ops overhead

---

## 10. Known Issues from v1 (Do Not Repeat)

- No data augmentation → add via `configs/train.yaml`
- `NUM_CLASSES = 20` with 7 actual classes → set `nc: 7` exactly
- Random (non-stratified) splits → fixed in Stage 3
- Implicit class ID trust from source files → fixed in Stage 1
- Custom mAP implementation with bugs → replaced with `torchmetrics`
- `venv/` tracked in git → covered by `.gitignore`
- `requirements.txt` encoding corruption → rewritten
