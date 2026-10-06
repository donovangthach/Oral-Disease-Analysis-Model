"""
Saves the current train/val/test split in data/processed/ to a manifest file.

Stage 3 reads this manifest so every photo stays in the same split across experiments, which keeps
the test set locked.
"""

from pathlib import Path
import json
import logging

logging.basicConfig(level=logging.INFO, format="%(message)s")   # logging formatting

PROCESSED_DIR = Path("data") / "processed"
MANIFEST_PATH = Path("configs") / "split_manifest.json"


def get_group_key(label_path):
    """
    Returns an ID shared by every copy of the same original photo (matches Stage 3).
    """
    key, _, _ = label_path.stem.partition(".rf.")               # keep everything before RoboFlow's copy ID

    return key


def main():
    """
    Records which split every photo in data/processed/ is in and save it to MANIFEST_PATH.
    """
    logging.info("=== Saving split manifest ===")

    if MANIFEST_PATH.exists():                                  # never overwrite a locked split
        logging.warning(f"[{MANIFEST_PATH}] already exists, skipping")
        return

    if not PROCESSED_DIR.exists():                              # check if Stage 3 was ran
        logging.warning(f"[{PROCESSED_DIR}] doesn't exist, run Stage 3 first")
        return

    manifest = {}                                               # photo key -> split name
    for split in ("train", "val", "test"):                      # loop through each split
        for label_path in (PROCESSED_DIR / split / "labels").glob("*.txt"):
            key = get_group_key(label_path)                     # every copy of a photo shares this key
            if key in manifest and manifest[key] != split:      # same photo already saved under another split
                logging.warning(f"LEAK: [{key}] is in both {manifest[key]} and {split}")
            manifest[key] = split

    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, sort_keys=True))    # sorted so git diffs stay readable

    logging.info(f"{len(manifest)} photos saved to [{MANIFEST_PATH}]")
    logging.info("=== Manifest saved, commit to git ===")


if __name__ == "__main__":
    main()
