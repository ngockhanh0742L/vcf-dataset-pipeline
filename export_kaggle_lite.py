"""Export JPEG sequences without modifying the existing output."""

import argparse
import csv
import io
import json
import os
import zipfile
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path

import cv2
import pandas as pd


def select_windows(table, limit):
    selected = []
    for _, group in table.groupby(["dataset_id", "video_id"], sort=True):
        rows = group.sort_values("start_time", kind="stable")
        if len(rows) <= limit:
            selected.extend(rows.to_dict("records"))
            continue
        size = len(rows) / limit
        for index in range(limit):
            chunk = rows.iloc[int(index * size) : int((index + 1) * size)]
            selected.append(chunk.loc[chunk["window_score"].idxmax()].to_dict())
    return selected


def encode_sequence(task):
    root, paths, quality = task
    result = []
    for relative in paths:
        source = Path(root) / relative
        image = cv2.imread(str(source), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(source)
        ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if not ok:
            raise RuntimeError(f"Cannot encode {source}")
        name = Path(relative).with_suffix(".jpg").as_posix()
        result.append((f"sequences/{name}", encoded.tobytes()))
    return result


def csv_bytes(rows, fields):
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="output", type=Path)
    parser.add_argument("--archive", default="vcf-processed-jpeg95-3seq.zip", type=Path)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--sequences-per-video", type=int, default=3)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--max-videos", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.workers < 1 or args.sequences_per_video < 1:
        parser.error("workers and sequences-per-video must be positive")
    if not 1 <= args.jpeg_quality <= 100:
        parser.error("jpeg-quality must be between 1 and 100")
    if args.archive.exists():
        raise FileExistsError(f"Refusing to overwrite {args.archive}")

    manifest = args.source / "manifests" / "sequence_manifest.csv"
    table = pd.read_csv(manifest)
    fields = list(table.columns)
    if args.max_videos is not None:
        keys = table[["dataset_id", "video_id"]].drop_duplicates().head(args.max_videos)
        table = table.merge(keys, on=["dataset_id", "video_id"], how="inner")
    selected = select_windows(table, args.sequences_per_video)
    tasks, exported_rows = [], []
    for row in selected:
        paths = json.loads(row["frame_paths"])
        tasks.append((str(args.source / "sequences"), paths, args.jpeg_quality))
        exported = dict(row)
        exported["frame_paths"] = json.dumps(
            [Path(path).with_suffix(".jpg").as_posix() for path in paths]
        )
        exported["image_format"] = "jpg"
        exported_rows.append(exported)

    info = {
        "profile": f"jpeg{args.jpeg_quality}_{args.sequences_per_video}seq",
        "video_count": int(table[["dataset_id", "video_id"]].drop_duplicates().shape[0]),
        "sequence_count": len(selected),
        "sequences_per_video": args.sequences_per_video,
        "jpeg_quality": args.jpeg_quality,
        "image_size": int(table["image_size"].iloc[0]),
        "selection": "best window_score in each temporal stratum",
    }
    partial = args.archive.with_suffix(args.archive.suffix + ".partial")
    args.archive.parent.mkdir(parents=True, exist_ok=True)
    try:
        with zipfile.ZipFile(partial, "w", zipfile.ZIP_STORED, allowZip64=True) as bundle:
            bundle.writestr("manifests/sequence_manifest.csv", csv_bytes(exported_rows, fields))
            bundle.writestr("manifests/export_info.json", json.dumps(info, indent=2))
            iterator, pending, completed = iter(tasks), set(), 0
            with ProcessPoolExecutor(max_workers=args.workers) as executor:
                for _ in range(args.workers * 2):
                    try:
                        pending.add(executor.submit(encode_sequence, next(iterator)))
                    except StopIteration:
                        break
                while pending:
                    done, pending = wait(pending, return_when=FIRST_COMPLETED)
                    for future in done:
                        for name, content in future.result():
                            bundle.writestr(name, content)
                        completed += 1
                        if completed % 100 == 0 or completed == len(tasks):
                            print(f"Exported {completed}/{len(tasks)} sequences", flush=True)
                        try:
                            pending.add(executor.submit(encode_sequence, next(iterator)))
                        except StopIteration:
                            pass
        os.replace(partial, args.archive)
    except BaseException:
        if partial.exists():
            partial.unlink()
        raise

    with zipfile.ZipFile(args.archive) as bundle:
        bad = bundle.testzip()
    if bad:
        raise RuntimeError(f"ZIP integrity check failed at {bad}")
    print(json.dumps({"archive": str(args.archive), "bytes": args.archive.stat().st_size, **info}))


if __name__ == "__main__":
    main()
