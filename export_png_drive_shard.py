"""Create one deterministic, self-contained ZIP shard from the original PNG output."""

import argparse
import hashlib
import json
import os
import zipfile
from pathlib import Path

import pandas as pd


SPLIT_ORDER = {"train": 0, "val": 1, "test": 2}


def _group_key(dataset_id, group_id):
    return f"{dataset_id}\0{group_id}"


def _build_plan(table, source, target_bytes):
    sequence_root = source / "sequences"
    group_sizes = {}
    for index, row in enumerate(table.itertuples(index=False), 1):
        size = 0
        for relative in json.loads(row.frame_paths):
            size += (sequence_root / relative).stat().st_size
        key = _group_key(row.dataset_id, row.group_id)
        item = group_sizes.setdefault(
            key,
            {"dataset_id": row.dataset_id, "group_id": row.group_id, "split": row.split, "bytes": 0},
        )
        item["bytes"] += size
        if index % 2000 == 0 or index == len(table):
            print(f"Indexed {index}/{len(table)} sequences", flush=True)

    groups = sorted(
        group_sizes.values(),
        key=lambda item: (SPLIT_ORDER.get(item["split"], 99), item["dataset_id"], item["group_id"]),
    )
    shards, current = [], None
    for group in groups:
        if (
            current is None
            or current["split"] != group["split"]
            or (current["groups"] and current["bytes"] + group["bytes"] > target_bytes)
        ):
            current = {"part": len(shards) + 1, "split": group["split"], "bytes": 0, "groups": []}
            shards.append(current)
        current["groups"].append({"dataset_id": group["dataset_id"], "group_id": group["group_id"]})
        current["bytes"] += group["bytes"]
    return {
        "version": 1,
        "target_bytes": target_bytes,
        "total_source_bytes": sum(item["bytes"] for item in groups),
        "group_count": len(groups),
        "shards": shards,
    }


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description="Export one original-PNG Drive shard")
    parser.add_argument("--source", default="output", type=Path)
    parser.add_argument("--destination", default="drive-shards", type=Path)
    parser.add_argument("--part", required=True, type=int)
    parser.add_argument("--target-gib", default=10.0, type=float)
    parser.add_argument("--max-videos", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.part < 1 or args.target_gib <= 0:
        parser.error("part and target-gib must be positive")

    manifest_path = args.source / "manifests" / "sequence_manifest.csv"
    table = pd.read_csv(manifest_path)
    if args.max_videos is not None:
        video_keys = table[["dataset_id", "video_id"]].drop_duplicates().head(args.max_videos)
        table = table.merge(video_keys, on=["dataset_id", "video_id"], how="inner")
    args.destination.mkdir(parents=True, exist_ok=True)
    plan_path = args.destination / "png_shard_plan.json"
    if plan_path.exists():
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
    else:
        plan = _build_plan(table, args.source, int(args.target_gib * 1024**3))
        temporary = plan_path.with_suffix(".json.partial")
        temporary.write_text(json.dumps(plan, indent=2), encoding="utf-8")
        os.replace(temporary, plan_path)

    if args.part > len(plan["shards"]):
        raise ValueError(f"Part must be between 1 and {len(plan['shards'])}")
    shard = plan["shards"][args.part - 1]
    keys = {_group_key(item["dataset_id"], item["group_id"]) for item in shard["groups"]}
    selected = table[
        table.apply(lambda row: _group_key(row["dataset_id"], row["group_id"]) in keys, axis=1)
    ].copy()

    name = f"vcf-png-{shard['split']}-part-{args.part:03d}-of-{len(plan['shards']):03d}.zip"
    archive = args.destination / name
    partial = archive.with_suffix(".zip.partial")
    if archive.exists() or partial.exists():
        raise FileExistsError(f"Refusing to overwrite {archive} or its partial file")

    shard_info = {
        "part": args.part,
        "total_parts": len(plan["shards"]),
        "split": shard["split"],
        "group_count": len(shard["groups"]),
        "sequence_count": len(selected),
        "source_bytes": shard["bytes"],
        "format": "original PNG, ZIP_STORED",
    }
    try:
        with zipfile.ZipFile(partial, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as bundle:
            bundle.writestr("manifests/sequence_manifest.csv", selected.to_csv(index=False))
            bundle.writestr("manifests/shard_info.json", json.dumps(shard_info, indent=2))
            for index, row in enumerate(selected.itertuples(index=False), 1):
                for relative in json.loads(row.frame_paths):
                    bundle.write(args.source / "sequences" / relative, f"sequences/{Path(relative).as_posix()}")
                if index % 100 == 0 or index == len(selected):
                    print(f"Packed {index}/{len(selected)} sequences", flush=True)
        os.replace(partial, archive)
    except BaseException:
        if partial.exists():
            partial.unlink()
        raise

    with zipfile.ZipFile(archive) as bundle:
        bad = bundle.testzip()
    if bad:
        raise RuntimeError(f"ZIP integrity check failed at {bad}")
    checksum = _sha256(archive)
    checksum_path = archive.with_suffix(archive.suffix + ".sha256")
    checksum_path.write_text(f"{checksum}  {archive.name}\n", encoding="ascii")
    print(json.dumps({"archive": str(archive), "bytes": archive.stat().st_size, "sha256": checksum, **shard_info}))


if __name__ == "__main__":
    main()
