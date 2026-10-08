# ODAM Results

All runs evaluated on the same locked test split (1,614 images, grouped by original photo, seed 42).

## Baseline (2026-09-29)
- Run: `runs/detect/train` (HPC), eval job 21471503
- Commit: `d9e1b97`
- Config: yolo11m, 300 epochs, batch 32, lr0 0.001, AdamW, no freeze
- Data: 8 original sources, 6,141 photos (4,296 / 917 / 928)

| Class | Val AP50 | Test AP50 | Test AP50-95 |
|---|---|---|---|
| caries | 0.855 | 0.851 | 0.630 |
| gingivitis | 0.778 | 0.752 | 0.440 |
| tooth_discoloration | 0.682 | 0.631 | 0.413 |
| ulcer | 0.824 | 0.832 | 0.535 |
| calculus | 0.214 | 0.419 | 0.134 |
| hypodontia | 0.636 | 0.562 | 0.371 |
| **all** | **0.665** | **0.674** | **0.420** |

Notes: old train-6 (0.814) was inflated by Roboflow-copy leakage and is not comparable.
Calculus AP swings by about 20 points between val and test, which suggests inconsistent labels across sources.

## Experiment (a): plaque -> DISCARD (2026-10-07)
- Change: `oral_detector` plaque (class 4) mapped to DISCARD instead of calculus in `1_remap_classes.py`
- Run: `runs/detect/train-3` (HPC), train job 21646869, eval job 21668343
- Commit: `880aab6` (data change in `7e2c56d`)
- Config: same as baseline, all 300 epochs ran (no early stop); job used 8 CPUs / 32G after an OOM kill
- Data: same 6,141 photos and same test set; 69 train + 18 val calculus boxes removed

| Class | Val AP50 | Test AP50 | Test AP50-95 | Test AP50 vs baseline |
|---|---|---|---|---|
| caries | 0.864 | 0.857 | 0.624 | +0.006 |
| gingivitis | 0.785 | 0.748 | 0.430 | -0.004 |
| tooth_discoloration | 0.717 | 0.635 | 0.412 | +0.004 |
| ulcer | 0.822 | 0.832 | 0.551 | 0.000 |
| calculus | 0.240 | 0.395 | 0.124 | -0.024 |
| hypodontia | 0.617 | 0.532 | 0.355 | -0.030 |
| **all** | **0.674** | **0.666** | **0.416** | **-0.008** |

Notes: no measurable effect. Calculus went +0.026 on val but -0.024 on test (opposite directions), and hypodontia moved -0.030 with no data change, so one-run noise is about +/-0.03 AP50 for the small classes.
Kept plaque as DISCARD because those labels were visibly wrong (boxes on plaque, real tartar unlabeled). This run is the new reference for future experiments.