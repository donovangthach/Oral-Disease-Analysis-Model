# Oral Disease Detection Model (v2)

Object detection model for identifying 6 oral disease classes from clinical photographs.
Built on YOLO11 (Ultralytics) with a multi-source dataset pipeline and a locked, leak-free test split.
See [ARCHITECTURE.md](ARCHITECTURE.md) for design decisions and [RESULTS.md](RESULTS.md) for every
experiment's results.

## Disease Classes

| ID | Class                |
|----|----------------------|
| 0  | Caries               |
| 1  | Gingivitis           |
| 2  | Tooth Discoloration  |
| 3  | Ulcer                |
| 4  | Calculus             |
| 5  | Hypodontia           |

Plaque is not a class. Plaque boxes from the Oral Detector source are discarded (see RESULTS.md, experiment (a)).

## Current Results

Model: YOLO11m, 300 epochs, evaluated on the locked test split (1,614 images).

| Class | Test AP50 |
|---|---|
| Caries | 0.857 |
| Gingivitis | 0.748 |
| Tooth Discoloration | 0.635 |
| Ulcer | 0.832 |
| Calculus | 0.395 |
| Hypodontia | 0.532 |
| **All (mAP50)** | **0.666** |
| All except Calculus (mAP50) | 0.721 |

Calculus is the weak class. Its labels are incomplete (one box per affected tooth, with many affected
teeth left unlabeled), and it has only about 233 unique training photos. The test set's calculus labels
have the same gaps, so its score is unreliable. Fixing it needs expert-annotated data (see RESULTS.md).

## Project Structure

```
├── ARCHITECTURE.md              # Design decisions and rationale
├── RESULTS.md                   # Results log, one section per experiment
├── configs/
│   ├── dataset.yaml             # Class names and data paths
│   ├── train.yaml               # Model and training hyperparameters
│   └── split_manifest.json      # Photo -> split assignment that locks the test set
├── scripts/                     # Data pipeline (run 1-4 in order) and analysis tools
│   ├── 1_remap_classes.py       # Remap each source's class IDs to the 6 canonical classes
│   ├── 2_merge_datasets.py      # Merge all sources into one flat folder
│   ├── 3_split_dataset.py       # Grouped, stratified 70/15/15 split that follows the manifest
│   ├── 4_verify_dataset.py      # Format check, class counts, backgrounds, leak check
│   ├── save_split_manifest.py   # One-time: save the current split to the manifest
│   ├── analyze_class.py         # Calculus label analysis (per-source counts, box sizes, samples)
│   └── export_for_relabel.py    # Export test calculus photos for relabeling in CVAT
├── src/
│   ├── train.py
│   ├── evaluate.py
│   ├── infer.py
│   └── export.py
└── hpc/                         # Oregon State SLURM job scripts
    ├── train_job.sh
    └── eval_job.sh
```

`data/`, `models/`, `runs/`, and `logs/` are not tracked in git.

## Setup

Requires Python 3.10+. Run every command from the repository root.

```bash
python -m venv venv
venv\Scripts\activate          # Windows
source venv/bin/activate       # Linux / macOS
pip install -r requirements.txt
```

## Usage

### 1. Prepare Data
Download each source into `data/sources/<source_name>/` (see [Dataset Sources](#dataset-sources)), then
run the pipeline:
```bash
python scripts/1_remap_classes.py
python scripts/2_merge_datasets.py
python scripts/3_split_dataset.py
python scripts/4_verify_dataset.py
```
The output is `data/processed/{train,val,test}/{images,labels}`.

**How the split works:**
- **Grouped by photo.** Roboflow exports several augmented copies of each photo (`<name>.rf.<hash>`).
  All copies of a photo go to the same split, so copies can't leak from train into test.
- **Locked.** Stage 3 puts every photo listed in `configs/split_manifest.json` back in its saved split,
  so the test set stays the same across experiments. Only new photos are split (and added to the manifest).
- **Background images are kept.** Images with no boxes (e.g. healthy teeth) stay in the dataset as negatives.

### 2. Train
```bash
python src/train.py
# or on the Oregon State HPC:
sbatch hpc/train_job.sh
```
`src/train.py` reads the model name and hyperparameters from `configs/train.yaml`. Each run is saved
to `runs/detect/train`, `runs/detect/train-2`, and so on.

### 3. Evaluate
Copy the run's weights to `models/best.pt`, then evaluate on the test split:
```bash
cp runs/detect/<run>/weights/best.pt models/best.pt
python src/evaluate.py
# or on the HPC:
sbatch hpc/eval_job.sh
```
The results (overall and per-class AP50 / AP50-95) are written to `logs/evaluation_results.json`.

### 4. Inference
```bash
python src/infer.py path/to/image.jpg
```
Runs `models/best.pt` on one image and shows the detections.

### 5. Export for Deployment
```bash
python src/export.py
```
Exports `models/best.pt` to ONNX (`models/best.onnx`). OpenVINO export for Intel CPUs is planned.

## Dataset Sources

| Folder (`data/sources/`) | Source | Classes used |
|---|---|---|
| `oral_detector` | Roboflow `clients-mpvn2/oral-detector` v3 | Calculus, Caries, Gingivitis, Ulcer (incl. lesions and xerostomia), Tooth Discoloration. Plaque discarded |
| `oral_diseases` | Roboflow `gilang-pappa-tanto-pambua-zplqr/oral-diseases-rmzuw` v2 | All 6 classes |
| `zenodo_caries` | Zenodo, DOI: 10.5281/zenodo.14827784 | Caries, Hypodontia (missing tooth) |
| `calculus` | Roboflow `dental-iex5i/calculus-tzg18` v1 | Calculus |
| `calculus_not` | Roboflow `ashvikha/calculus-7eza2` v1 (no calculus despite its name) | Caries, Gingivitis, Tooth Discoloration, Ulcer |
| `ulcer` | Roboflow `seer-jtb9k/ulcer-4arpj` v16 | Caries, Gingivitis, Tooth Discoloration, Ulcer |
| `hypodontia` | Roboflow `hypodontia/hypodontia-wb6lf` v1 | Hypodontia |
| `healthy_teeth` | Roboflow `sultan-qyobm/healthy-teeth-hgddf` v1 | None (background images) |

Every source's class IDs are explicitly remapped in `scripts/1_remap_classes.py` before merging.
More sources will be added one at a time after their labels are audited.
