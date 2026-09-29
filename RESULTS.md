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