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

## Known limitation: calculus labels (2026-10-22)
- Finding: calculus is weak because of its data, not a pipeline bug (checked with `scripts/analyze_class.py`)
- Data: only ~233 unique train photos with calculus (Roboflow copies inflate the box count); test has 56
- Labels: one box per affected tooth, but many teeth with visible tartar have no box (e.g. test photo `calculus_-1000-`: tartar on most teeth, 3 boxes)
- Test labels have the same gaps, so calculus test AP is unreliable: correct predictions on unlabeled tartar count as false positives, which lowers precision

| Model | Test mAP50 (6 classes) | Test mAP50 (5, no calculus) | Test mAP50-95 (6 classes) | Test mAP50-95 (5, no calculus) |
|---|---|---|---|---|
| Baseline (`train`) | 0.674 | 0.726 | 0.420 | 0.478 |
| Experiment (a) (`train-3`) | 0.666 | 0.721 | 0.416 | 0.474 |

Notes: the 5-class numbers are the mean of the other 5 per-class test APs in the tables above (Ultralytics mAP is a plain mean over classes; averaging all 6 rounded values reproduces the reported 6-class numbers).
Relabeling the 56 test calculus photos by hand was planned (experiment (b), `scripts/export_for_relabel.py`) but cancelled: without dental expertise, new labels could add their own errors. Fixing calculus needs expert-annotated data.
Decision: `train-3` stays the reference, but accuracy is not good enough to deploy yet. Next: re-add the removed sources one at a time (audited, new photos to train only so val and test stay locked), compared on the 5-class mAP since calculus test AP is unreliable.

## Noise check: seed 1 (2026-10-23)
- Change: only `seed: 1` in `train.yaml` (every earlier run used Ultralytics' default seed 0)
- Run: `runs/detect/train-4` (HPC), train job 21905788; a measurement only, not promoted and not evaluated on test
- Data and config: same as experiment (a)

| Class | Val AP50 train-3 (seed 0) | Val AP50 train-4 (seed 1) | Difference |
|---|---|---|---|
| caries | 0.864 | 0.855 | -0.009 |
| gingivitis | 0.785 | 0.782 | -0.003 |
| tooth_discoloration | 0.717 | 0.680 | -0.037 |
| ulcer | 0.822 | 0.846 | +0.024 |
| calculus | 0.240 | 0.226 | -0.014 |
| hypodontia | 0.617 | 0.658 | +0.041 |
| **all (6 classes)** | **0.674** | **0.675** | **+0.001** |
| all except calculus | 0.761 | 0.764 | +0.003 |

Notes: with no data change, single classes moved by up to 0.04 AP50, but the 5-class mean moved only 0.003. So per-class differences under ~0.04 are noise, and the 5-class mean is the number to compare experiments on.
Val turned out to share photos with train (next section), so these values are inflated, but the seed-to-seed gap should be similar.

## Cross-source duplicates found (2026-10-23)
- Check: `scripts/find_duplicates.py` (perceptual hash of each image and its mirror; same photo = at most 4 of 64 bits different, cutoff checked by eye)
- Finding: 319 of 928 test photos (34%) have a near-identical copy in train under another source or file name. Main cause: `calculus_not` and `oral_detector` share most of their photos, and `calculus` overlaps with both. Stage 4's leak check compares photo keys, which include the source name, so it could not see this.
- Impact: every test number above (baseline, experiment (a), and the 5-class means) is inflated.
- Fix: the 788 train photos that duplicate a val or test photo are listed in `configs/duplicate_train_photos.json`, and Stage 3 leaves them out of train. Val and test images are unchanged. A clean baseline will be retrained and becomes the new reference.

## Clean baseline: duplicates removed (2026-10-24)
- Change: the 788 duplicate photos listed in `configs/duplicate_train_photos.json` left out of train (Stage 3); val and test images byte-identical to before (md5-checked)
- Run: `runs/detect/train-5` (HPC), train job 21916990, eval job 21923802; `models/best.pt` md5 `43023753c22cbeeb5d45cb17b84b814c`
- Config: same as experiment (a), seed 0. Early-stopped at epoch 136 (patience 50), best epoch 86, 2.6 h
- Data: 3,508 unique train photos (5,661 files), was 4,296 photos (7,656 files); val 917 and test 928 photos unchanged

| Class | Val AP50 | Test AP50 | Test AP50-95 | Test AP50 train-3 (inflated) | Difference |
|---|---|---|---|---|---|
| caries | 0.633 | 0.649 | 0.373 | 0.857 | -0.208 |
| gingivitis | 0.161 | 0.124 | 0.035 | 0.748 | -0.624 |
| tooth_discoloration | 0.249 | 0.350 | 0.158 | 0.635 | -0.285 |
| ulcer | 0.556 | 0.568 | 0.246 | 0.832 | -0.264 |
| calculus | 0.244 | 0.350 | 0.106 | 0.395 | -0.045 |
| hypodontia | 0.653 | 0.538 | 0.353 | 0.532 | +0.006 |
| **all (6 classes)** | **0.416** | **0.430** | **0.212** | **0.666** | **-0.236** |
| all except calculus | 0.450 | 0.446 | 0.233 | 0.721 | -0.275 |

Notes: the drop follows the duplicates. Gingivitis, tooth discoloration, caries and ulcer come mostly from `calculus_not` and `oral_detector`, which held most of the removed photos, and they fell 0.2-0.6. Hypodontia and calculus come mostly from other sources and barely moved. So most of the old scores came from recognizing near-copies of training photos, not from detecting disease in new photos.
The early stop did not cause the drop: train-3's val mAP50 at epoch 86 was already 0.671 (final 0.674). Val loss stayed far above train loss (box 2.19 vs 0.96, cls 2.04 vs 0.70 at the last epoch), so the model overfits the remaining train photos.
Gingivitis has only 399 train boxes left (val has 385), which explains its near-zero AP.
The duplicate cutoff (4 of 64 bits) is strict on purpose; some real copies at distances 5-8 (non-zenodo) are still in train, so these numbers may still be slightly optimistic.
The seed noise estimate (previous sections) was measured on the leaky data and should be rechecked on this setup.
Decision: `train-5` is the new reference (test 0.430 / 0.446 without calculus). Not deployable. Next: re-add the removed sources one at a time, with a duplicate check against val and test for each, prioritizing gingivitis and tooth discoloration data.
