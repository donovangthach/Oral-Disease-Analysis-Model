# Oral Disease Detection Model (v2)

Object detection model for identifying 7 oral disease classes from clinical photographs.
Built on YOLOv11 with a multi-source dataset pipeline. See [ARCHITECTURE.md](ARCHITECTURE.md)
for full design decisions and rationale.

## Disease Classes

| ID | Class                |
|----|----------------------|
| 0  | Caries               |
| 1  | Gingivitis           |
| 2  | Tooth Discoloration  |
| 3  | Ulcer                |
| 4  | Calculus             |
| 5  | Plaque               |
| 6  | Hypodontia           |

## Project Structure

```
├── ARCHITECTURE.md       # Design decisions and rationale
├── configs/
│   ├── dataset.yaml      # Class definitions and data paths
│   └── train.yaml        # Training hyperparameters
├── scripts/              # Data pipeline — run in order
│   ├── 1_remap_classes.py
│   ├── 2_merge_datasets.py
│   ├── 3_split_dataset.py
│   └── 4_verify_dataset.py
├── src/                  # Core ML code
│   ├── train.py
│   ├── evaluate.py
│   ├── infer.py
│   └── export.py
└── hpc/                  # Oregon State SLURM scripts
    ├── train_job.sh
    └── eval_job.sh
```

## Setup

```bash
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

## Usage

### 1. Prepare Data
Download dataset sources into `data/sources/<source_name>/`, then run the pipeline:
```bash
python scripts/1_remap_classes.py
python scripts/2_merge_datasets.py
python scripts/3_split_dataset.py
python scripts/4_verify_dataset.py
```

### 2. Train
```bash
python src/train.py
# or on Oregon State HPC:
sbatch hpc/train_job.sh
```

### 3. Evaluate
```bash
python src/evaluate.py
```

### 4. Inference
```bash
python src/infer.py --image path/to/image.jpg
```

### 5. Export for Deployment
```bash
python src/export.py
```

## Dataset Sources

| Source | Classes | Format |
|--------|---------|--------|
| Roboflow — Oral Detector | Gingivitis, Calculus, Ulcer, Plaque, Discoloration | YOLO |
| Zenodo — Caries Dataset (DOI: 10.5281/zenodo.14827784) | Caries | YOLO |
| Roboflow — Healthy Teeth | Healthy (background) | YOLO |

Additional sources to be added. All sources are explicitly remapped in
`scripts/1_remap_classes.py` before merging.
