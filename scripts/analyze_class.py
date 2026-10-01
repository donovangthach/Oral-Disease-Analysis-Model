"""
Analyzes each class from their source to pinpoint good or bad data.
"""

from pathlib import Path
import logging
from collections import Counter, defaultdict
from statistics import median

logging.basicConfig(level=logging.INFO, format="%(message)s")   # logging formatting

PROCESSED_DIR = Path("data") / "processed"

CALCULUS_ID = 4

# longest names first: "calculus_not_..." also starts with "calculus_"
SOURCES = ["calculus_not", "oral_detector", "oral_diseases", "healthy_teeth", "zenodo_caries",
           "hypodontia", "calculus", "ulcer"]


def get_source(label_path):
    """
    Returns the source name that Stage 2 added to the front of the filename.
    """
    for source in SOURCES:                                      # check list in order, so calculus_not wins over calculus
        if label_path.name.startswith(source + "_"):            # match the whole prefix, including the "_"
            return source
        
    return "unknown"


def get_boxes_by_source(split):
    """
    Returns the width and height of every CALCULUS_ID box in one split, grouped by source.
    """
    label_files = list((PROCESSED_DIR / split / "labels").glob("*.txt"))    # glob all .txt files in the split/labels/ directory

    boxes = defaultdict(list)                                   # source name -> list of (width, height)
    for label_path in label_files:
        content = label_path.read_text(encoding="utf-8", errors="ignore")   # get content from .txt file
        lines = content.splitlines()                            # separate each line
        for line in lines:
            parts = line.strip().split()
            if not parts:                                       # skip if line is empty
                continue

            class_id = int(parts[0])
            if class_id == CALCULUS_ID:
                width, height = float(parts[3]), float(parts[4])    # YOLO line: class x_center y_center width height
                boxes[get_source(label_path)].append((width, height))   # add this box to its source's list

    return boxes


def log_box_stats(split, boxes):
    """
    Logs the median width, height, and area of the boxes from each source.
    """
    for source, source_boxes in boxes.items():                  # loop through each source and its list of boxes
        widths = [w for w, h in source_boxes]                   # keep only width of each (width, height)
        heights = [h for w, h in source_boxes]                  # keep only the height
        areas = [w * h for w, h in source_boxes]                # fraction of the image each box covers
        logging.info(f"[{split}] {source}: median w {median(widths):.3f} | h {median(heights):.3f} | area {median(areas):.4f}")


def main():
    """
    Counts the calculus boxes per source in each split and logs the results.
    """
    logging.info("=== Calculus boxes per source ===")

    for split in ("train", "val", "test"):                      # loop through each split
        boxes = get_boxes_by_source(split)                      # source name -> list of (width, height)
        counts = Counter({source: len(source_boxes) for source, source_boxes in boxes.items()}) # source name -> number of calculus boxes
        total = sum(counts.values())                            # all calculus boxes in this split
        dist = " | ".join(f"{source}: {n} ({n / total:.0%})" for source, n in counts.most_common())
        logging.info(f"[{split}] {total} boxes | {dist}")
        log_box_stats(split, boxes)                             # log median box sizes for each source


if __name__ == "__main__":
    main()
