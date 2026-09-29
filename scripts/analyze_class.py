"""
Analyzes each class from their source to pinpoint good or bad data.
"""

from pathlib import Path
import logging
from collections import Counter

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


def count_by_source(split):
    """
    Counts the CALCULUS_ID boxes in one split, grouped by the source each file came from.
    """
    label_files = list((PROCESSED_DIR / split / "labels").glob("*.txt"))    # glob all .txt files in the split/labels/ directory

    counts = Counter()                                          # source name -> number of calculus boxes
    for label_path in label_files:
        content = label_path.read_text(encoding="utf-8", errors="ignore")   # get content from .txt file
        lines = content.splitlines()                            # separate each line
        for line in lines:
            parts = line.strip().split()
            if not parts:                                       # skip if line is empty
                continue

            class_id = int(parts[0])
            if class_id == CALCULUS_ID:
                counts[get_source(label_path)] += 1             # add one more box to this file's source

    return counts


def main():
    """
    Counts the calculus boxes per source in each split and logs the results.
    """
    logging.info("=== Calculus boxes per source ===")

    for split in ("train", "val", "test"):                      # loop through each split
        counts = count_by_source(split)                         # source name -> number of calculus boxes
        total = sum(counts.values())                            # all calculus boxes in this split
        dist = " | ".join(f"{source}: {n} ({n / total:.0%})" for source, n in counts.most_common())
        logging.info(f"[{split}] {total} boxes | {dist}")


if __name__ == "__main__":
    main()
