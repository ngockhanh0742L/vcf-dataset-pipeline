"""Build a canonical-split manifest for numeric-ID FaceForensics++ videos."""

import argparse
import json
from pathlib import Path

import pandas as pd


METHODS = {"original", "Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures", "FaceShifter"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data/ff-c23/FaceForensics++_C23", type=Path)
    parser.add_argument("--splits", default="data/ff-c23/splits", type=Path)
    parser.add_argument("--output", default="data/ff-c23/ffpp_manifest.csv", type=Path)
    args = parser.parse_args()

    split_by_id = {}
    for split in ("train", "val", "test"):
        pairs = json.loads((args.splits / f"{split}.json").read_text(encoding="utf-8"))
        for pair in pairs:
            for video_id in pair:
                previous = split_by_id.setdefault(str(video_id), split)
                if previous != split:
                    raise ValueError(f"Video ID {video_id} appears in multiple splits")
    if len(split_by_id) != 1000:
        raise ValueError(f"Expected 1000 canonical video IDs, found {len(split_by_id)}")

    metadata = pd.read_csv(args.root / "csv" / "FF++_Metadata.csv")
    rows = []
    for item in metadata.itertuples(index=False):
        relative = Path(getattr(item, "_1"))  # 'File Path' after itertuples sanitization
        method = relative.parts[0]
        if method not in METHODS:
            continue
        ids = relative.stem.split("_")
        if not all(video_id in split_by_id for video_id in ids):
            raise ValueError(f"Cannot assign canonical split to {relative}")
        splits = {split_by_id[video_id] for video_id in ids}
        if len(splits) != 1:
            raise ValueError(f"Manipulated pair crosses canonical splits: {relative}")
        label = 0 if method == "original" else 1
        rows.append(
            {
                "path": relative.as_posix(),
                "label": label,
                "group_id": ids[0],
                "split": splits.pop(),
                "video_id": relative.as_posix(),
                "class_name": "real" if label == 0 else "fake",
                "compression": "c23",
                "gen_method": method.lower(),
                "resolution": f"{item.Width}x{item.Height}",
                "media_type": "mp4",
                "background_type": "",
                "video_name": relative.stem,
            }
        )

    table = pd.DataFrame(rows).sort_values(["split", "label", "gen_method", "video_id"])
    if len(table) != 6000:
        raise ValueError(f"Expected 6000 FF++ videos, found {len(table)}")
    if table["path"].duplicated().any():
        raise ValueError("Duplicate video paths in FF++ manifest")
    missing = [path for path in table["path"] if not (args.root / path).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} videos; first: {missing[0]}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output, index=False)
    print(table.groupby(["split", "class_name"]).size().to_string())
    print(f"Wrote {len(table)} rows to {args.output}")


if __name__ == "__main__":
    main()
