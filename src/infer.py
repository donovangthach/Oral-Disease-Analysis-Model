"""
Takes an image and runs the model on it.
"""

from ultralytics import YOLO
from pathlib import Path
import sys
import logging

logging.basicConfig(level=logging.INFO, format="%(message)s")   # logging formatting

BEST_MODEL = Path("models") / "best.pt"

def main():
    """
    Loads the best trained model, runs inference on a single image provided via command line, and 
    displays the annotated result.
    """
    if len(sys.argv) < 2:
        logging.error("Usage: python src/infer.py <image_path>")
        return

    image_path = Path(sys.argv[1])
    if not image_path.exists():                                 # check if the image path exists
        logging.warning("image path does not exist")
        return

    if not BEST_MODEL.exists():                                 # check if the model exists
        logging.warning("model not found")
        return

    model = YOLO(BEST_MODEL)
    results = model.predict(source=image_path, show=True)       # runs the model on the image and returns a list of Results objects containing the detected bounding boxes

if __name__ == "__main__":
    main()