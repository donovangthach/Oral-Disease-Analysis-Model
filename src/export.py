"""
Exports the trained model to ONNX format so it can be served without PyTorch dependency.
"""

from pathlib import Path
from ultralytics import YOLO
import logging

logging.basicConfig(level=logging.INFO, format="%(message)s")   # logging formatting

BEST_MODEL = Path("models") / "best.pt"

def main():
    """
    Exports model to ONNX format.
    """
    if not BEST_MODEL.exists():
        logging.warning("model not found")
        return
    
    model = YOLO(BEST_MODEL)
    model.export(format="onnx")                                 # export the model in onnx format

    logging.info(f"Exported to {BEST_MODEL.parent / 'best.onnx'}")

if __name__ == "__main__":
    main()
