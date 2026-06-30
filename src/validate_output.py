import argparse
from collections import defaultdict
import json
import os
from pathlib import Path

import pandas as pd

from src.config import load_config


REQUIRED_COLUMNS = {
    "sequence_id",
    "dataset_id",
    "video_id",
    "group_id",
    "label",
    "split",
    "frame_paths",
    "seq_len",
    "pipeline_fingerprint",
}


def validate_output(manifest_path, sequence_dir, require_all_splits=True):
    manifest_path = Path(manifest_path)
    sequence_root = Path(sequence_dir).resolve()
    if not manifest_path.is_file():
        raise ValueError(f"Manifest not found: {manifest_path}")
    table = pd.read_csv(manifest_path)
    missing_columns = sorted(REQUIRED_COLUMNS.difference(table.columns))
    if missing_columns:
        raise ValueError(f"Manifest is missing columns: {', '.join(missing_columns)}")
    if table.empty:
        raise ValueError("Manifest contains no accepted sequences")

    errors = []
    if table["sequence_id"].duplicated().any():
        errors.append("sequence_id contains duplicates")
    if table["pipeline_fingerprint"].astype(str).nunique() != 1:
        errors.append("manifest mixes multiple pipeline fingerprints")
    if not set(table["label"].astype(int)).issubset({0, 1}):
        errors.append("label must contain only 0 and 1")
    if not set(table["split"]).issubset({"train", "val", "test"}):
        errors.append("split must contain only train, val, and test")
    if require_all_splits:
        missing_splits = {"train", "val", "test"}.difference(set(table["split"]))
        if missing_splits:
            errors.append(
                "missing required split(s): " + ", ".join(sorted(missing_splits))
            )
        train_labels = set(table.loc[table["split"] == "train", "label"].astype(int))
        if train_labels != {0, 1}:
            errors.append("train split must contain both labels 0 and 1")
    split_per_group = table.groupby(["dataset_id", "group_id"])["split"].nunique()
    if (split_per_group > 1).any():
        errors.append("one or more leakage groups occur in multiple splits")

    missing_files = 0
    invalid_rows = 0
    expected_by_directory = defaultdict(set)
    for row_index, row in table.iterrows():
        try:
            frame_paths = json.loads(row["frame_paths"])
        except (TypeError, json.JSONDecodeError):
            invalid_rows += 1
            continue
        if not isinstance(frame_paths, list) or len(frame_paths) != int(row["seq_len"]):
            invalid_rows += 1
            continue
        for relative_path in frame_paths:
            relative = Path(relative_path)
            if relative.is_absolute() or ".." in relative.parts:
                invalid_rows += 1
                break
            expected_by_directory[relative.parent].add(relative.name)
    for relative_directory, expected_names in expected_by_directory.items():
        directory = sequence_root / relative_directory
        try:
            actual_names = set(os.listdir(directory))
        except OSError:
            missing_files += len(expected_names)
            continue
        missing_files += len(expected_names.difference(actual_names))
    if invalid_rows:
        errors.append(f"{invalid_rows} row(s) have invalid frame_paths")
    if missing_files:
        errors.append(f"{missing_files} referenced frame file(s) are missing")
    if errors:
        raise ValueError("Output validation failed: " + "; ".join(errors))

    return {
        "sequences": int(len(table)),
        "videos": int(table[["dataset_id", "video_id"]].drop_duplicates().shape[0]),
        "datasets": sorted(table["dataset_id"].astype(str).unique().tolist()),
        "splits": table["split"].value_counts().sort_index().to_dict(),
    }


def main():
    parser = argparse.ArgumentParser(description="Validate processed sequences and manifest")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument(
        "--allow-partial", action="store_true", help="Allow smoke-test output without all splits"
    )
    args = parser.parse_args()
    config = load_config(args.config)
    result = validate_output(
        config.data.manifest_path,
        config.data.sequence_dir,
        require_all_splits=not args.allow_partial,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
