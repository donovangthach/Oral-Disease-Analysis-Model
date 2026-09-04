"""
Splits the data/merged/ directory into train/val/test sets at the image level.

Groups the images by their dominant class and splits each group independently to maintain class
balance across splits.

Writes final images and labels into data/processed/ and deletes data/merged/ afterward to save
storage and space (mostly for the Oregon State HPC space quota).
"""

from pathlib import Path
import logging
import shutil
import random
from collections import Counter, defaultdict

logging.basicConfig(level=logging.INFO, format="%(message)s")   # logging formatting

MERGED_DIR = Path("data") / "merged"
PROCESSED_DIR = Path("data") / "processed"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

TRAIN = 0.70
VAL   = 0.15
TEST  = 0.15

SEED = 42


def get_dominant_class(label_path):
    """
    Parses a txt file and counts the total amount of classes in it to find the dominant class.
    """
    content = label_path.read_text(encoding="utf-8", errors="ignore")   # get the content from the .txt file
    lines = content.splitlines()                                # separate each line

    id_list = []
    for line in lines:
        parts = line.strip().split()                            # split line into strings
        if not parts:
            continue

        class_id = int(parts[0])                                # grab the class_id
        id_list.append(class_id)

    if not id_list:                                             # handle an empty label file
        return None
    
    return Counter(id_list).most_common(1)[0][0]                # return the most dominant class as a tuple [(class_id), (count)]


def copy_split(label_files, split):
    """
    Copies each label file and its matching image into PROCESSED_DIR/split/.
    """
    pairs = 0
    mismatches = 0
    for label_path in label_files:
        image_path = None
        for ext in IMAGE_EXTENSIONS:
            candidate = MERGED_DIR / "images" / (label_path.stem + ext) # try to find a candidate with the IMAGE_EXTENSIONS extension
            if candidate.exists():
                image_path = candidate                          # set the image_path to match the label_path if candidate match is found
                break

        if image_path is None:                                  # check if image_path was found
            logging.warning(f"[{split}] no image found for: {label_path.stem}, skipping")
            mismatches += 1
            continue

        shutil.copy2(image_path, PROCESSED_DIR / split / "images" / image_path.name)    # copy image to PROCESSED_DIR/split/images/
        shutil.copy2(label_path, PROCESSED_DIR / split / "labels" / label_path.name)    # copy label to PROCESSED_DIR/split/labels/
        pairs += 1

    return pairs, mismatches


def main():
    """
    Reads all label/image pairs from MERGED_DIR, groups them by dominant class, performs a
    stratified split, writes result to PROCESSED_DIR, and deletes MERGED_DIR.
    """
    logging.info("=== Stage 3: Splitting Dataset ===")

    random.seed(SEED)                                           # set random seed to predetermined seed

    if (PROCESSED_DIR / ".processed").exists():                 # check if dataset has already been split
        logging.warning(f"The dataset has already been split and is in [{PROCESSED_DIR}], skipping Stage 3")
        return

    if not MERGED_DIR.exists():                                 # check if Stage 2 was ran before Stage 3
        logging.warning("data/merged/ directory currently does not exist. Run Stage 2 before running Stage 3")
        return

    for split in ("train", "val", "test"):                      # create subdirectories in data/processed/ for each split
        (PROCESSED_DIR / split / "images").mkdir(parents=True, exist_ok=True)
        (PROCESSED_DIR / split / "labels").mkdir(parents=True, exist_ok=True)

    label_files = list((MERGED_DIR / "labels").glob("*.txt"))   # put all .txt files in data/merged/labels/ in a list
    groups = defaultdict(list)
    for label_path in label_files:
        dominant_class = get_dominant_class(label_path)         # get dominant class from label_path
        if dominant_class is None:
            continue

        groups[dominant_class].append(label_path)               # append label_path with dominant_class as it's key

    train_total = train_fails = val_total = val_fails = test_total = test_fails = 0
    for dominant_class, paths in groups.items():                # loop through each class group
        random.shuffle(paths)                                   # shuffle the labels

        n = len(paths)
        train_end = int(n * TRAIN)                              # gain index cut off for training split
        val_end = train_end + int(n * VAL)                      # gain index cut off for validation split

        train = paths[:train_end]                               # split them into different lists called train, val, and test
        val   = paths[train_end:val_end]
        test  = paths[val_end:]

        p, m = copy_split(train, "train"); train_total += p; train_fails += m   # copy each split into the PROCESS_DIR directory and track how many pairs and mismatches there are
        p, m = copy_split(val, "val"); val_total += p; val_fails += m
        p, m = copy_split(test, "test"); test_total += p; test_fails += m

    shutil.rmtree(MERGED_DIR)                                    # delete MERGED_DIR
    (PROCESSED_DIR / ".processed").write_text("")                # make .processed marker file in PROCESSED_DIR

    logging.info(f"train total: {train_total} pairs | train fails: {train_fails}")
    logging.info(f"val total: {val_total} pairs | val fails: {val_fails}")
    logging.info(f"test total: {test_total} pairs | test fails: {test_fails}")
    logging.info("=== Stage 3 complete, run Stage 4 ===")

if __name__ == "__main__":
    main()
