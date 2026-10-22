"""
Copies one copy of every test photo with calculus into a folder CVAT can import, so the calculus
boxes can be relabeled by hand.
"""

from pathlib import Path
import logging
import shutil
import yaml

logging.basicConfig(level=logging.INFO, format="%(message)s")   # logging formatting

PROCESSED_DIR = Path("data") / "processed"
DATASET_CONFIG = Path("configs") / "dataset.yaml"
EXPORT_DIR = Path("data") / "relabel" / "test_calculus"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

CALCULUS_ID = 4


def get_calculus_photos():
    """
    Returns one label file for every test photo with a CALCULUS_ID box, keyed by its photo ID.
    """
    label_files = sorted((PROCESSED_DIR / "test" / "labels").glob("*.txt")) # sorted so every run keeps the same copy

    photos = {}                                                 # photo key -> label path of the copy we keep
    for label_path in label_files:
        key, _, _ = label_path.stem.partition(".rf.")           # every copy of a photo shares this key
        if key in photos:                                       # already kept a copy of this photo
            continue

        content = label_path.read_text(encoding="utf-8", errors="ignore")
        lines = content.splitlines()
        for line in lines:
            parts = line.strip().split()
            if not parts:                                       # skip if line is empty
                continue

            class_id = int(parts[0])
            if class_id == CALCULUS_ID:                         # this copy has calculus, keep it
                photos[key] = label_path
                break

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


def copy_photo(label_path, image_path):
    """
    Copies one image and its label file into the CVAT folder layout under EXPORT_DIR.
    """
    shutil.copy2(image_path, EXPORT_DIR / "images" / "train" / image_path.name)
    shutil.copy2(label_path, EXPORT_DIR / "labels" / "train" / label_path.name) 


def write_cvat_files(image_names):
    """
    Writes the data.yaml (class name) and train.txt (image list) files that CVAT needs.
    """
    config = yaml.safe_load(DATASET_CONFIG.read_text(encoding="utf-8")) # same class names as training
    data = {"path": "./", "train": "train.txt", "names": config["names"]}
    (EXPORT_DIR / "data.yaml").write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")

    lines = [f"images/train/{name}" for name in image_names]     # one image path per line, relative to EXPORT_DIR
    (EXPORT_DIR / "train.txt").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def main():
    """
    Exports one copy of every test photo with calculus to EXPORT_DIR, ready to zip and import into CVAT.
    """
    logging.info("=== Exporting test calculus photos for CVAT ===")

    if EXPORT_DIR.exists():                                     # never mix a new export with an old one
        logging.warning(f"[{EXPORT_DIR}] already exists, delete it first")
        return

    (EXPORT_DIR / "images" / "train").mkdir(parents=True)       # create the folders copy_photo() needs
    (EXPORT_DIR / "labels" / "train").mkdir(parents=True)

    photos = get_calculus_photos()                              # photo key -> label path of the copy we keep
    image_names = []
    for key, label_path in photos.items():
        image_path = find_image("test", label_path)
        if image_path is None:                                  # skip if no image is found
            logging.warning(f"[{key}] no image found, skipping")
            continue

        copy_photo(label_path, image_path)
        image_names.append(image_path.name)                     # train.txt lists only photos that were copied

    write_cvat_files(image_names)

    logging.info(f"{len(image_names)} of {len(photos)} photos exported to [{EXPORT_DIR}]")
    logging.info("=== Done, zip the folder's contents and import them into CVAT ===")


if __name__ == "__main__":
    main()
