"""
Evaluates the model on data/processed/test/ data.
"""

from ultralytics import YOLO
from pathlib import Path
import json
import logging

logging.basicConfig(level=logging.INFO, format="%(message)s")   # logging formatting

DATASET_YAML = Path("configs") / "dataset.yaml"

BEST_MODEL = Path("models") / "best.pt"
RESULTS_PATH = Path("logs") / "evaluation_results.json"


def main():
    """
    Loads the best trained model, runs evaluation on the test set, and saves metrics to a JSON file.
    """
    if not BEST_MODEL.exists():                                 # check if the model exists
        logging.warning("model not found")
        return

    model = YOLO(BEST_MODEL)
    results = model.val(data=DATASET_YAML, split="test")        # evaluate the model on the test split

    metrics = {                                                 # store metric data in a dict
        "mAP50":     results.box.map50,
        "mAP50_95":  results.box.map,
        "precision": results.box.mp,
        "recall":    results.box.mr,
    }

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)      # create the directory
    RESULTS_PATH.write_text(json.dumps(metrics, indent=2))      # write the metric data into a file in the directory

    logging.info(f"Results saved to [{RESULTS_PATH}]")

if __name__ == "__main__":
    main()
