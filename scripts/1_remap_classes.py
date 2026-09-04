"""
Remaps classes from separate datasets to match the classes the model is going to use.

REMAPPED CLASSES:

|----|---------------------|
| ID | Class Name          |
|----|---------------------|
| 0  | caries              |
| 1  | gingivitis          |
| 2  | tooth_discoloration |
| 3  | ulcer               |
| 4  | calculus            |
| 5  | hypodontia          |
|----|---------------------|

DISCARD is used to remove the annotation line from the dataset, but keeps the image as a 
background/negative example.
"""

from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format="%(message)s")   # logging formatting

DISCARD = -1

# roboflow: clients-mpvn2/oral-detector v3
# data.yaml order: calculus(0), caries(1), gingivitis(2), lessions(3),
#                  plaque(4), tooth discoloration(5), ulcer(6), xerostomia(7)
ORAL_DETECTOR = {
    0: 4,   # calculus -> calculus
    1: 0,   # caries -> caries
    2: 1,   # gingivitis -> gingivitis
    3: 3,   # lessions -> ulcer
    4: 4,   # plaque -> calculus
    5: 2,   # tooth discoloration -> tooth_discoloration
    6: 3,   # ulcer -> ulcer
    7: 3,   # xerostomia -> ulcer
}

# roboflow: sultan-qyobm/healthy-teeth-hgddf v1
# data.yaml order: Healthy teeths(0)
HEALTHY_TEETH = {
    0: DISCARD,  # Healthy teeths -> DISCARD
}

# zenodo: 10.5281/zenodo.14827784
# NOTE: there is no data.yaml file in this dataset
# caries(0), missing tooth(1)
ZENODO_CARIES = {
    0: 0,   # caries -> caries
    1: 5,   # missing tooth -> hypodontia
}

# roboflow: gilang-pappa-tanto-pambua-zplqr/oral-diseases-rmzuw v2
# data.yaml order: calculus(0), caries(1), gingivitis(2), hypodontia(3),
#                  tooth_discoloration(4), ulcer(5)
ORAL_DISEASES = {
    0: 4,   # calculus -> calculus
    1: 0,   # caries -> caries
    2: 1,   # gingivitis -> gingivitis
    3: 5,   # hypodontia -> hypodontia
    4: 2,   # tooth_discoloration -> tooth_discoloration
    5: 3,   # ulcer -> ulcer
}

# roboflow: seer-jtb9k/ulcer-4arpj v16
# data.yaml: Caries(0), Gingivitis(1), Tooth Discoloration(2), Ulcer(3)
ULCER = {
    0: 0,   # Caries -> caries
    1: 1,   # Gingivitis -> gingivitis
    2: 2,   # Tooth Discoloration -> tooth_discoloration
    3: 3,   # Ulcer -> ulcer
}

# roboflow: ashvikha/calculus-7eza2 v1
# data.yaml: caries(0), gingivitis(1), tooth discoloration(2), ulcer(3)
# NOTE: this dataset is named calculus, but has no calculus data
CALCULUS_NOT = {
    0: 0,   # caries -> caries
    1: 1,   # gingivitis -> gingivitis
    2: 2,   # tooth discoloration -> tooth_discoloration
    3: 3,   # ulcer -> ulcer
}

# roboflow: dental-iex5i/calculus-tzg18 v1
# data.yaml: CALCULUS(0)
CALCULUS = {
    0: 4,   # CALCULUS -> calculus
}

# roboflow: hypodontia/hypodontia-wb6lf v1
# data.yaml: hypodotia(0)
HYPODONTIA = {
    0: 5,   # hypodotia -> hypodontia
}

# SOURCES dict used to map each folder name in data/sources/ to its corresponding mapping dict
SOURCES = {
    "oral_detector": ORAL_DETECTOR,
    "healthy_teeth": HEALTHY_TEETH,
    "zenodo_caries": ZENODO_CARIES,
    "oral_diseases": ORAL_DISEASES,
    "ulcer":         ULCER,
    "calculus_not":  CALCULUS_NOT,
    "calculus":      CALCULUS,
    "hypodontia":    HYPODONTIA,
}

# path to source directory
SOURCES_DIR = Path("data") / "sources"


def remap_source(source_name: str, mapping: dict) -> None:
    """
    Remaps all label files in a single source folder to canonical class IDs.
    Discards annotations that don't map to any of the 6 canonical classes.
    Logs a summary of how many annotations were remapped and discarded.
    """
    source_dir = SOURCES_DIR / source_name          # grab source directory

    if not source_dir.exists():                     # check if source directory exists
        logging.warning(f"Skipping '{source_name}', folder not found ")
        return

    label_files = list(source_dir.rglob("*.txt"))   # use rglob to grab all .txt files
    label_files = [f for f in label_files if "README" not in f.name]    # filter txt files

    if (source_dir / ".remapped").exists():                    # check if source has already been remapped
        logging.warning(f"[{source_name}] has already been remapped, skipping.")
        return

    remapped_lines = 0
    discarded_lines = 0
    unknown_ids = set()

    for label_path in label_files:
        content = label_path.read_text(encoding="utf-8", errors="ignore") # ignore errors and continue
        lines = content.splitlines()                # separate each line

        new_lines = []                              # stores new lines made
        for line in lines:
            parts = line.strip().split()            # split the line into strings
            if not parts:
                continue

            src_id = int(parts[0])                  # make the class ID an int

            if src_id not in mapping:
                unknown_ids.add(src_id)             # add unknown IDs to a set
                discarded_lines += 1
                continue                            # skip this line
            elif mapping[src_id] == DISCARD:
                discarded_lines += 1
                continue                            # skip this line
            else:
                parts[0] = str(mapping[src_id])     # remap the ID in the file
                new_lines.append(" ".join(parts))   # store the new line
                remapped_lines += 1

        label_path.write_text("\n".join(new_lines), encoding="utf-8") # write to the .txt file

    if unknown_ids:
        logging.warning(f"[{source_name}] Unknown class IDs discarded: {sorted(unknown_ids)}")

    logging.info(f"[{source_name}] {len(label_files)} files | {remapped_lines} remapped | {discarded_lines} discarded")

    (source_dir / ".remapped").write_text("")       # store an empty file to prevent remapping the source again


def main():
    """
    Loops through each source from SOURCES dict and runs the remapping function on each of them.
    """
    logging.info("=== Stage 1: Remapping class IDs ===")

    for source in SOURCES:                          # remap each source in SOURCES
        remap_source(source, SOURCES[source])

    logging.info("=== Stage 1 complete, run Stage 2 ===")

if __name__ == "__main__":
    main()
