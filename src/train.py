"""
Trains the YOLOv11 model on the annotated data in data/processed/.
"""

from ultralytics import YOLO
from pathlib import Path

DATASET_YAML = Path("configs") / "dataset.yaml"
TRAIN_YAML = Path("configs") / "train.yaml"


def main():
    """
    Loads the model and runs the training config with the model, saving the best current model state.
    """
    model = YOLO("yolo11s.pt")                                  # load YOLOv11s model

    for param in model.model[:10].parameters():                 # freeze the backbone to prevent weights from changing too fast
        param.requires_grad = False

    model.train(data=DATASET_YAML, cfg=TRAIN_YAML)              # train the model on the dataset with the configurations set

if __name__ == "__main__":
    main()
