"""
Checks data/processed/ for valid YOLO format and class distribution.
"""

from pathlib import Path
import logging
from collections import Counter

logging.basicConfig(level=logging.INFO, format="%(message)s")   # logging formatting

PROCESSED_DIR = Path("data") / "processed"

CLASS_NAMES = ["caries", "gingivitis", "tooth_discoloration", "ulcer", "calculus", "hypodontia"]

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def get_group_key(label_path):
    """
    Returns an ID shared by every copy of the same original photo (matches stage 3).
    """
    key, _, _ = label_path.stem.partition(".rf.")                          # keep everything before Roboflow's copy ID

    return key


def main():
    """
    Loops through each split folder, checking each label file and validates each file's YOLO
    format, and logs the class distribution.
    """
    logging.info("=== Stage 4: Verifying Dataset ===")

    if not PROCESSED_DIR.exists():
        logging.warning(f"[{PROCESSED_DIR.name}] doesn't exist. run Stage 3 first")
        return

    total_invalid = 0                                           # invalid lines across ALL splits
    split_keys = {}                                             # split name -> set of photo keys in that split
    for split in PROCESSED_DIR.iterdir():                       # loop through each directory in data/processed/
        if not split.is_dir():                                  # skip over the .processed marker file
            continue

        label_files = list((split / "labels").glob("*.txt"))    # glob all .txt files in the split/labels/ directory

        missing_images = 0
        invalid_lines = 0
        class_counts = Counter()
        keys = set()                                            # unique photo keys in this split
        background = 0                                          # label files with no boxes (healthy / negative images)

        for label_path in label_files:
            keys.add(get_group_key(label_path))                 # record which photo this file came from
            image_path = None
            for ext in IMAGE_EXTENSIONS:
                candidate = split / "images" / (label_path.stem + ext) # try to find a candidate with the IMAGE_EXTENSIONS extension
                if candidate.exists():
                    image_path = candidate                      # set the image_path to match the label_path if candidate match is found
                    break

            if image_path is None:                              # check if image_path was found
                logging.warning(f"[{split.name}] no image found for: [{label_path.stem}], skipping")
                missing_images += 1
                continue

            content = label_path.read_text(encoding="utf-8", errors="ignore") # get content from .txt file
            if not content.strip():                             # empty or whitespace-only label file
                background += 1
            lines = content.splitlines()                        # separate each line
            for line in lines:
                parts = line.strip().split()

                if not parts:                                   # skip if line is empty
                    continue
                if len(parts) != 5:                             # YOLO format requires 5 columns
                    invalid_lines += 1
                    continue

                class_id = int(parts[0])
                coords = [float(x) for x in parts[1:]]

                if class_id not in range(len(CLASS_NAMES)) or not all(0.0 <= c <= 1.0 for c in coords): # values required to be between 0 and 1 for the non-class_id columns
                    invalid_lines += 1
                    continue

                class_counts[class_id] += 1

        logging.info(f"[{split.name}] {len(label_files)} labels | {missing_images} missing images | {invalid_lines} invalid lines")
        dist = " | ".join(f"{CLASS_NAMES[i]}: {class_counts[i]}" for i in range(len(CLASS_NAMES)))
        logging.info(f"[{split.name}] {dist}")
        logging.info(f"[{split.name}] {len(keys)} unique photos | {background} background images")
        total_invalid += invalid_lines                          # add this split's count to the running total
        split_keys[split.name] = keys                           # save this split's keys for the leak check

    train, val, test = split_keys["train"], split_keys["val"], split_keys["test"]
    leaked = (train & val) | (train & test) | (val & test)  # photo keys found in more than one split

    if leaked:
        logging.warning(f"LEAK: {len(leaked)} photos appear in more than one split, e.g. {sorted(leaked)}[:5]")
    else:
        logging.info("No leaks: every photo is in exactly one split")

    if total_invalid > 0 or leaked:
        logging.warning("=== Stage 4 complete, please fix the issues with the invalid lines ===")
    else:
        logging.info("=== Stage 4 complete, model is ready to train ===")


if __name__ == "__main__":
    main()
