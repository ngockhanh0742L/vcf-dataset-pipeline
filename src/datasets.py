"""Dataset adapter registry producing one canonical video record."""

from collections import defaultdict
from dataclasses import replace
import hashlib
from pathlib import Path

import pandas as pd

from src.vcf import VideoSample, assign_split, parse_vcf_video
from src.video_reader import list_videos


def _discover_vcf(source, config_split):
    strict = bool(getattr(source, "strict_layout", True))
    samples = []
    rejected = []
    for video_path in list_videos(source.root):
        try:
            samples.append(
                parse_vcf_video(video_path, source.root, config_split, dataset_id=source.id)
            )
        except ValueError as exc:
            rejected.append({"dataset_id": source.id, "path": video_path, "error": str(exc)})
    if rejected and strict:
        raise ValueError(
            f"Dataset '{source.id}' rejected {len(rejected)} path(s); "
            "fix its VCF layout or set strict_layout: false"
        )
    return samples, rejected


def _optional_text(row, name):
    value = row.get(name, "")
    return "" if pd.isna(value) else str(value)


def _discover_manifest(source, config_split):
    table = pd.read_csv(source.manifest)
    required = {"path", "label", "group_id"}
    missing = sorted(required.difference(table.columns))
    if missing:
        raise ValueError(
            f"Dataset '{source.id}' manifest is missing columns: {', '.join(missing)}"
        )

    root = Path(source.root).resolve()
    samples = []
    rejected = []
    for row_index, row in table.iterrows():
        raw_path = Path(str(row["path"])).expanduser()
        path = raw_path if raw_path.is_absolute() else root / raw_path
        path = path.resolve()
        if not path.is_file():
            rejected.append(
                {
                    "dataset_id": source.id,
                    "path": str(path),
                    "error": f"Manifest row {row_index}: video does not exist",
                }
            )
            continue
        try:
            relative = path.relative_to(root).as_posix()
        except ValueError:
            relative = path.name
        label = int(row["label"])
        if label not in {0, 1}:
            raise ValueError(
                f"Dataset '{source.id}' row {row_index} has non-binary label {label}"
            )
        if pd.isna(row["group_id"]):
            raise ValueError(f"Dataset '{source.id}' row {row_index} has empty group_id")
        raw_group = str(row["group_id"]).strip().lower()
        if not raw_group:
            raise ValueError(f"Dataset '{source.id}' row {row_index} has empty group_id")
        group_id = f"{source.id}:{raw_group}"
        video_id = _optional_text(row, "video_id") or relative
        class_name = _optional_text(row, "class_name") or ("real" if label == 0 else "fake")
        explicit_split = _optional_text(row, "split").lower()
        if explicit_split and explicit_split not in {"train", "val", "test"}:
            raise ValueError(
                f"Dataset '{source.id}' row {row_index} has invalid split {explicit_split!r}"
            )
        samples.append(
            VideoSample(
                path=str(path),
                video_id=video_id,
                compression=_optional_text(row, "compression"),
                gen_method=_optional_text(row, "gen_method"),
                resolution=_optional_text(row, "resolution"),
                background_type=_optional_text(row, "background_type"),
                video_name=_optional_text(row, "video_name") or path.stem,
                group_id=group_id,
                label=label,
                class_name=class_name,
                split=explicit_split or assign_split(group_id, config_split),
                dataset_id=source.id,
                media_type=_optional_text(row, "media_type"),
                split_locked=bool(explicit_split),
            )
        )
    if rejected and bool(getattr(source, "strict_layout", True)):
        raise ValueError(
            f"Dataset '{source.id}' rejected {len(rejected)} manifest row(s); "
            "fix missing paths or set strict_layout: false"
        )
    return samples, rejected


ADAPTERS = {"vcf": _discover_vcf, "manifest": _discover_manifest}


def _split_counts(size, ratios):
    exact = [size * ratio for ratio in ratios]
    counts = [int(value) for value in exact]
    remaining = size - sum(counts)
    order = sorted(
        range(len(ratios)), key=lambda index: (exact[index] - counts[index], -index), reverse=True
    )
    for index in order[:remaining]:
        counts[index] += 1
    return counts


def _assign_balanced_splits(samples, config_split):
    strategy = getattr(config_split, "strategy", "balanced_hash")
    if strategy != "balanced_hash":
        raise ValueError("Only split.strategy=balanced_hash is supported")
    ratios = (
        float(config_split.train_ratio),
        float(config_split.val_ratio),
        float(config_split.test_ratio),
    )
    locked_by_dataset = defaultdict(set)
    for sample in samples:
        locked_by_dataset[sample.dataset_id].add(sample.split_locked)
    mixed = [dataset_id for dataset_id, values in locked_by_dataset.items() if len(values) > 1]
    if mixed:
        raise ValueError(
            "A dataset cannot mix explicit and generated splits: " + ", ".join(sorted(mixed))
        )

    explicit_groups = defaultdict(set)
    for sample in samples:
        if sample.split_locked:
            explicit_groups[(sample.dataset_id, sample.group_id)].add(sample.split)
    leaking = [key for key, values in explicit_groups.items() if len(values) > 1]
    if leaking:
        raise ValueError(f"Explicit manifest splits leak {len(leaking)} group(s)")

    groups_by_dataset = defaultdict(set)
    for sample in samples:
        if not sample.split_locked:
            groups_by_dataset[sample.dataset_id].add(sample.group_id)

    assignments = {}
    split_names = ("train", "val", "test")
    for dataset_id, groups in groups_by_dataset.items():
        ordered = sorted(
            groups,
            key=lambda group_id: hashlib.sha256(
                f"{config_split.seed}:{dataset_id}:{group_id}".encode("utf-8")
            ).digest(),
        )
        counts = _split_counts(len(ordered), ratios)
        offset = 0
        for split_name, count in zip(split_names, counts):
            for group_id in ordered[offset : offset + count]:
                assignments[(dataset_id, group_id)] = split_name
            offset += count
    return [
        sample
        if sample.split_locked
        else replace(sample, split=assignments[(sample.dataset_id, sample.group_id)])
        for sample in samples
    ]


def discover_datasets(config, logger, selected_ids=None):
    selected = set(selected_ids or [])
    known = {source.id for source in config.datasets}
    unknown = selected.difference(known)
    if unknown:
        raise ValueError(f"Unknown dataset id(s): {', '.join(sorted(unknown))}")

    samples = []
    rejected = []
    for source in config.datasets:
        if selected and source.id not in selected:
            continue
        logger.info("Scanning dataset '%s' with %s adapter", source.id, source.adapter)
        discovered, source_rejected = ADAPTERS[source.adapter](source, config.split)
        samples.extend(discovered)
        rejected.extend(source_rejected)
        logger.info("Dataset '%s': %d valid videos", source.id, len(discovered))
    samples = _assign_balanced_splits(samples, config.split)
    samples.sort(key=lambda item: (item.dataset_id, item.video_id))
    keys = [(item.dataset_id, item.video_id) for item in samples]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate dataset_id/video_id pairs were discovered")
    return samples, rejected
