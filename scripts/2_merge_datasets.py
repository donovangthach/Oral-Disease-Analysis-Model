"""
Copies all remapped images and labels from 'data/sources/*/' into a single flat 'data/merged/'
directory.

Prefixes each filename with its source name to avoid collisions across datasets.
"""

from pathlib import Path
import logging
import shutil

logging.basicConfig(level=logging.INFO, format="%(message)s")   # logging formatting

SOURCES_DIR = Path("data") / "sources"
MERGED_DIR =  Path("data") / "merged"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def main():
    """
    Loops through each source folder to merge all the data into MERGED_DIR.
    """
    logging.info("=== Stage 2: Merging Datasets ===")

    if (MERGED_DIR / ".merged").exists():                       # check if .merged file exists
        logging.warning("Data has already been merged, move onto Stage 3")
        return

    (MERGED_DIR / "images").mkdir(parents=True, exist_ok=True)  # create 'images' directory
    (MERGED_DIR / "labels").mkdir(parents=True, exist_ok=True)  # create 'labels' directory

    copied = 0

    for source_dir in sorted(SOURCES_DIR.iterdir()):            # loop through SOURCES_DIR and grab all source directories
        if not source_dir.is_dir():
            continue

        if not (source_dir / ".remapped").exists():             # skip if .remapped file doesn't exist in source folder
            logging.warning(f"[{source_dir}] has not been remapped, skipping")
            continue

        label_files = list(source_dir.rglob("*.txt"))           # use rglob to grab all .txt files
        label_files = [f for f in label_files if "README" not in f.name]    # filter .txt files

        image_lookup = {
            f.stem: f
            for ext in IMAGE_EXTENSIONS
            for f in source_dir.rglob(f"*{ext}")
        }

        for label_path in label_files:
            image_path = image_lookup.get(label_path.stem)      # use the image_lookup dict to find connecting images

            if image_path is None:                              # skip if no image_path is found
                logging.warning(f"[{source_dir.name}] no image found for: {label_path.stem}, skipping")
                continue

            new_stem = source_dir.name + "_" + label_path.stem

            dst_image = MERGED_DIR / "images" / (new_stem + image_path.suffix)  # destination path for images
            dst_label = MERGED_DIR / "labels" / (new_stem + ".txt")             # destination path for labels

            shutil.copy2(image_path, dst_image)                 # copy image to images
            shutil.copy2(label_path, dst_label)                 # copy label to labels

            copied += 1                                         # increment the amount of copied image/label pairs

    logging.info(f"{copied} image/label pairs copied")

    (MERGED_DIR / ".merged").write_text("")                     # create a .merged file to prevent remerging

    logging.info("=== Stage 2 complete, run Stage 3")

if __name__ == "__main__":
    main()
