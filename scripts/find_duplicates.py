"""
Finds train photos that are near-duplicates of a val or test photo (the same photo under another
source or file name) and saves their photo keys to configs/duplicate_train_photos.json, so
Stage 3 can leave them out of train.
"""

from pathlib import Path
import logging
import json
import imagehash
from PIL import Image, ImageOps

logging.basicConfig(level=logging.INFO, format="%(message)s")   # logging formatting

PROCESSED_DIR = Path("data") / "processed"
TRAIN_DIR = PROCESSED_DIR / "train" / "images"
VAL_DIR = PROCESSED_DIR / "val" / "images"
TEST_DIR = PROCESSED_DIR / "test" / "images"
DUPLICATES_PATH = Path("configs") / "duplicate_train_photos.json"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

# max number of different bits (out of 64) for two images to count as the same photo
# checked by eye: 0-4 were always the same photo, 8 matched different zenodo patients
MAX_DISTANCE = 4


def hash_image(image_path):
    """
    Returns the perceptual hash of an image and of its left-right mirror.
    """
    with Image.open(image_path) as image:                       # open the image and close it again when done
        image_hash = imagehash.phash(image)                     # 64-bit fingerprint of the image
        mirror_hash = imagehash.phash(ImageOps.mirror(image))   # same, for the left-right flipped image

    return image_hash, mirror_hash


def hash_folder(image_dir):
    """
    Returns the hash and mirror hash of every image in image_dir, keyed by the file name.
    """
    hashes = {}                                                 # file name -> (hash, mirror hash)
    for image_path in sorted(image_dir.iterdir()):              # sorted so every run logs in the same order
        if image_path.suffix.lower() not in IMAGE_EXTENSIONS:   # skip anything that is not an image
            continue

        hashes[image_path.name] = hash_image(image_path)        # store both hashes under the file name

    logging.info(f"[{image_dir}] hashed {len(hashes)} images")

    return hashes


def find_matches(train_hashes, held_out_hashes):
    """
    Returns a dict of train photo key -> held-out file name, for every train image within
    MAX_DISTANCE of a held-out image.
    """
    matches = {}                                                # train photo key -> held-out file name it duplicates
    for train_name, (image_hash, mirror_hash) in train_hashes.items():
        for held_out_name, (held_out_hash, _) in held_out_hashes.items():
            distance = min(image_hash - held_out_hash, mirror_hash - held_out_hash) # closest of: same way round, or flipped
            if distance <= MAX_DISTANCE:                        # few enough different bits to be the same photo
                key, _, _ = Path(train_name).stem.partition(".rf.") # same key as get_group_key() in Stage 3
                matches[key] = held_out_name                    # remember one example, to check by eye
                break                                           # one match is enough for this train image

    return matches


def main():
    """
    Finds every train photo that duplicates a val or test photo and writes their keys to
    DUPLICATES_PATH.
    """
    logging.info("=== Finding Duplicates ===")

    if DUPLICATES_PATH.exists():                                # a rerun after Stage 3 would find nothing and overwrite the list
        logging.warning(f"[{DUPLICATES_PATH}] already exists, delete it first to rebuild it")
        return

    train_hashes = hash_folder(TRAIN_DIR)                       # hashed once, compared twice
    test_matches = find_matches(train_hashes, hash_folder(TEST_DIR))
    val_matches = find_matches(train_hashes, hash_folder(VAL_DIR))
    logging.info(f"[test] {len(test_matches)} train photos duplicate a test photo")
    logging.info(f"[val] {len(val_matches)} train photos duplicate a val photo")

    duplicates = test_matches | val_matches                     # merge both dicts into one
    DUPLICATES_PATH.write_text(json.dumps(duplicates, indent=2, sort_keys=True))    # same format as split_manifest.json
    logging.info(f"{len(duplicates)} train photos written to [{DUPLICATES_PATH}], commit it to git")
    logging.info("=== Finding Duplicates complete ===")
    

if __name__ == "__main__":
    main()
