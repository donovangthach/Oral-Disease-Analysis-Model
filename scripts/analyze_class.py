"""
Analyzes each class from their source to pinpoint good or bad data.
"""

from pathlib import Path
import logging
import random
from collections import Counter, defaultdict
from statistics import median
from PIL import Image, ImageDraw

logging.basicConfig(level=logging.INFO, format="%(message)s")   # logging formatting

PROCESSED_DIR = Path("data") / "processed"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

CALCULUS_ID = 4

# longest names first: "calculus_not_..." also starts with "calculus_"
SOURCES = ["calculus_not", "oral_detector", "oral_diseases", "healthy_teeth", "zenodo_caries",
           "hypodontia", "calculus", "ulcer"]

SAMPLES_DIR = Path("logs") / "calculus_samples"
SAMPLES_PER_SOURCE = 5

SEED = 42


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


def yolo_to_corners(parts, img_w, img_h):
    """
    Converts a YOLO box (normalized center, width, height) into pixel corners (x0, y0, x1, y1).
    """
    x_center = float(parts[1]) * img_w                          # normalized -> pixels (multiply by image width)
    y_center = float(parts[2]) * img_h                          # heights and y values use the image height
    width =    float(parts[3]) * img_w
    height =   float(parts[4]) * img_h

    return x_center - width / 2, y_center - height / 2, x_center + width / 2, y_center + height / 2


def draw_boxes(label_path, image_path):
    """
    Draws every CALCULUS_ID box from a label file onto its image and saves a copy to SAMPLES_DIR.
    """
    image = Image.open(image_path).convert("RGB")               # RGB so a red outline works on any image
    draw = ImageDraw.Draw(image)                                # a "pen" that draws onto this image
    img_w, img_h = image.size                                   # (width, height) in pixels

    content = label_path.read_text(encoding="utf-8", errors="ignore")   # get content from .txt file
    lines = content.splitlines()                                # separate each line
    for line in lines:
        parts = line.strip().split()
        if not parts:                                           # skip if line is empty
            continue

        class_id = int(parts[0])
        if class_id == CALCULUS_ID:                             # only draw calculus boxes
            draw.rectangle(yolo_to_corners(parts, img_w, img_h), outline="red", width=3)

    image.save(SAMPLES_DIR / image_path.name)                   # save a copy, never overwrite the original


def get_calculus_photos(split):
    """
    Returns one label file per photo that has a CALCULUS_ID box, grouped by source.
    """
    label_files = sorted((PROCESSED_DIR / split / "labels").glob("*.txt"))  # sorted so the same seed always picks the same photos

    photos = defaultdict(dict)                                  # source name -> {photo key: label_path}
    for label_path in label_files:
        content = label_path.read_text(encoding="utf-8", errors="ignore")   # get content from .txt file
        class_ids = [int(line.split()[0]) for line in content.splitlines() if line.strip()] # class ID of every box in this file

        if CALCULUS_ID in class_ids:                            # only keep photos with calculus
            key, _, _ = label_path.stem.partition(".rf.")       # same photo key as Stage 3
            photos[get_source(label_path)][key] = label_path    # copies share a key, so only one per photo is kept

    return photos


def find_image(split, label_path):
    """
    Returns the image that matches a label file, or None if there is no match.
    """
    for ext in IMAGE_EXTENSIONS:
        candidate = PROCESSED_DIR / split / "images" / (label_path.stem + ext)  # try each image extension
        if candidate.exists():
            return candidate

    return None


def main():
    """
    Counts the calculus boxes per source in each split, logs their median sizes, and saves sample images with the boxes drawn.
    """
    logging.info("=== Calculus boxes per source ===")

    for split in ("train", "val", "test"):                      # loop through each split
        boxes = get_boxes_by_source(split)                      # source name -> list of (width, height)
        counts = Counter({source: len(source_boxes) for source, source_boxes in boxes.items()}) # source name -> number of calculus boxes
        total = sum(counts.values())                            # all calculus boxes in this split
        dist = " | ".join(f"{source}: {n} ({n / total:.0%})" for source, n in counts.most_common())
        logging.info(f"[{split}] {total} boxes | {dist}")
        log_box_stats(split, boxes)                             # log median box sizes for each source

    logging.info("=== Saving calculus samples ===")
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)              # create logs/calculus_samples/ if it does not exist
    random.seed(SEED)                                           # pick the same photos every run

    photos = get_calculus_photos("train")                       # train has the most calculus photos
    for source, source_photos in photos.items():                # loop through each source and its photos
        n = min(SAMPLES_PER_SOURCE, len(source_photos))         # a source may have fewer than 5 photos
        picks = random.sample(list(source_photos.values()), n)  # n random label files, no repeats
        for label_path in picks:
            image_path = find_image("train", label_path)
            if image_path is None:                              # skip if no image is found
                logging.warning(f"[{source}] no image found for: {label_path.stem}, skipping")
                continue

            draw_boxes(label_path, image_path)

        logging.info(f"[{source}] drew {n} of {len(source_photos)} photos into [{SAMPLES_DIR}]")

    logging.info("=== Done, open the images in logs/calculus_samples/ ===")


if __name__ == "__main__":
    main()
