"""
Trains the YOLOv11 model on the annotated data in data/processed/.
"""

from ultralytics import YOLO
from pathlib import Path
import yaml

DATASET_YAML = Path("configs") / "dataset.yaml"
TRAIN_YAML = Path("configs") / "train.yaml"


def main():
    """
    Loads the model and runs the training config with the model, saving the best current model state.
    """
    with open(TRAIN_YAML) as f:
        train_cfg = yaml.safe_load(f)                           # read train.yaml into a dict

    model = YOLO(train_cfg["model"])                            # load the model named in train.yaml
    model.train(data=DATASET_YAML, cfg=TRAIN_YAML)              # train the model on the dataset with the configurations set


if __name__ == "__main__":
    main()
