import argparse
from pathlib import Path

import pandas as pd

ORGANIZER = Path(__file__).resolve().parents[2] / "ndb_ufes_data_organizer"
CLASSES = ["Leukoplakia without dysplasia", "Leukoplakia with dysplasia", "OSCC"]  # the class numbers 0, 1 and 2 of y_pred
WITH, OSCC = CLASSES[1], CLASSES[2]


def load(predictions: Path, patch_table: Path, ndb_metadata: Path) -> pd.DataFrame:
    """The test predictions with the NDB-UFES diagnosis of the picture and, when it has dysplasia, its severity."""
    frame = pd.read_csv(predictions, dtype=str, keep_default_na=False)
    frame = frame[frame["stage"] == "test"].assign(predicted=lambda f: f["y_pred"].astype(int).map(dict(enumerate(CLASSES))))
    origin = pd.read_csv(patch_table, dtype=str, keep_default_na=False).set_index("patch_id")["ndb_origin_id"]
    metadata = pd.read_csv(ndb_metadata, dtype=str, keep_default_na=False)
    severity = metadata.assign(origin=metadata["path"].str.removesuffix(".png")).set_index("origin")["dysplasia_severity"]
    return frame.assign(severity=frame["patch_id"].map(origin).map(severity).fillna(""))


def shares(frame: pd.DataFrame, by: str, order: list[str]) -> str:
    """One line for each value of ``by``: patches, pictures, and the share of the patches predicted as each class."""
    lines = ["| " + by + " | patches | pictures | called without dysplasia | called with dysplasia | called OSCC |", "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for value in order:
        group = frame[frame[by] == value]
        if group.empty:
            continue
        called = [f"{(group['predicted'] == name).mean() * 100:.1f}%" for name in CLASSES]
        lines.append(f"| {value or 'not in NDB-UFES'} | {len(group)} | {group['histopath_id'].nunique()} | {' | '.join(called)} |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--predictions", type=Path, required=True, help="The predictions CSV of a fold run.")
    parser.add_argument("--patch-table", type=Path, default=ORGANIZER / "results/phase0/dataset_tables/patch_table_public.csv")
    parser.add_argument("--ndb-metadata", type=Path, default=ORGANIZER / "data/ndb_ufes/origin_level/csvs/ndb-ufes.csv")
    args = parser.parse_args()
    frame = load(args.predictions, args.patch_table, args.ndb_metadata)
    print(f"{len(frame)} test patches, {frame['histopath_id'].nunique()} pictures, model {', '.join(sorted(set(frame['model'])))}\n")

    with_severity = frame[frame["severity"] != ""]
    print("Patches of the pictures diagnosed with dysplasia, by the severity of the picture (any patch label):\n")
    print(shares(with_severity, "severity", ["Mild", "Moderate", "Severe"]))
    print("\nThe same, only the patches labeled with dysplasia:\n")
    print(shares(with_severity[with_severity["patch_diagnosis"] == WITH], "severity", ["Mild", "Moderate", "Severe"]))
    print("\nPatches labeled with dysplasia, by the NDB-UFES diagnosis of their picture:\n")
    print(shares(frame[frame["patch_diagnosis"] == WITH], "ndb_diagnosis", ["OSCC", WITH, "Leukoplakia without dysplasia", ""]))


if __name__ == "__main__":
    main()
