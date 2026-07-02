"""Build a manifest for Celeb-DF v2 with the official test split."""

import argparse
import csv
import hashlib
from pathlib import Path
import re


CLASS_DIRS = {
    "Celeb-real": {"label": 0, "class_name": "real", "gen_method": "celeb_real"},
    "YouTube-real": {"label": 0, "class_name": "real", "gen_method": "youtube_real"},
    "Celeb-synthesis": {"label": 1, "class_name": "fake", "gen_method": "celeb_synthesis"},
}

TEST_LABELS = {
    "Celeb-real": "1",
    "YouTube-real": "1",
    "Celeb-synthesis": "0",
}


def _relative(path, root):
    return path.relative_to(root).as_posix()


def _stable_order(seed, value):
    return hashlib.sha256(f"{seed}:{value}".encode("utf-8")).digest()


def _group_id(relative_path):
    return Path(relative_path).with_suffix("").as_posix().lower()


def _identity_tokens(relative_path):
    return re.findall(r"id\d+", Path(relative_path).stem)


def _trainval_group_ids(rows):
    identities = set()
    edges = []
    for row in rows:
        if row["split"] == "test":
            continue
        tokens = _identity_tokens(row["path"])
        identities.update(tokens)
        if row["label"] == 1 and len(tokens) >= 2:
            edges.append((tokens[0], tokens[1]))

    parent = {identity: identity for identity in identities}

    def find(identity):
        while parent[identity] != identity:
            parent[identity] = parent[parent[identity]]
            identity = parent[identity]
        return identity

    def union(first, second):
        first_root = find(first)
        second_root = find(second)
        if first_root != second_root:
            parent[second_root] = first_root

    for first, second in edges:
        union(first, second)

    component_members = {}
    for identity in identities:
        component_members.setdefault(find(identity), []).append(identity)
    component_name = {}
    for members in component_members.values():
        name = "identity-component/" + "-".join(sorted(members))
        for identity in members:
            component_name[identity] = name

    group_ids = {}
    for row in rows:
        base_group = _group_id(row["path"])
        if row["split"] == "test":
            group_ids[row["path"]] = "official-test/" + base_group
            continue
        tokens = _identity_tokens(row["path"])
        if tokens:
            group_ids[row["path"]] = component_name[tokens[0]]
        else:
            group_ids[row["path"]] = base_group
    return group_ids


def _read_official_test(root):
    test_path = root / "List_of_testing_videos.txt"
    rows = {}
    with test_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            parts = line.strip().split()
            if not parts:
                continue
            if len(parts) != 2:
                raise ValueError(f"Invalid test-list row {line_number}: {line!r}")
            listed_label, relative_path = parts
            class_dir = Path(relative_path).parts[0]
            expected_label = TEST_LABELS.get(class_dir)
            if expected_label is None:
                raise ValueError(f"Unknown class directory in test list: {relative_path}")
            if listed_label != expected_label:
                raise ValueError(
                    f"Unexpected Celeb-DF test label for {relative_path}: "
                    f"{listed_label}, expected {expected_label}"
                )
            if relative_path in rows:
                raise ValueError(f"Duplicate test-list path: {relative_path}")
            if not (root / relative_path).is_file():
                raise FileNotFoundError(f"Missing test-list video: {relative_path}")
            rows[relative_path] = True
    return rows


def _assign_train_val(rows, val_ratio, seed):
    group_ids = _trainval_group_ids(rows)
    for row in rows:
        row["group_id"] = group_ids[row["path"]]

    class_counts = {}
    group_counts = {}
    for row in rows:
        if row["split"] == "test":
            continue
        class_counts[row["class_name"]] = class_counts.get(row["class_name"], 0) + 1
        group_counts.setdefault(row["group_id"], {"fake": 0, "real": 0})
        group_counts[row["group_id"]][row["class_name"]] += 1

    val_groups = set()
    fake_target = round(class_counts.get("fake", 0) * val_ratio)
    real_target = round(class_counts.get("real", 0) * val_ratio)

    fake_groups = [
        (group_id, counts)
        for group_id, counts in group_counts.items()
        if counts["fake"]
    ]
    if fake_groups:
        fake_group, _ = min(
            fake_groups,
            key=lambda item: (
                abs(item[1]["fake"] - fake_target),
                abs(item[1]["real"] - real_target),
                _stable_order(seed, item[0]),
            ),
        )
        val_groups.add(fake_group)

    current_real = sum(group_counts[group_id]["real"] for group_id in val_groups)
    real_only_groups = [
        group_id
        for group_id, counts in group_counts.items()
        if counts["real"] and not counts["fake"] and group_id not in val_groups
    ]
    real_only_groups.sort(key=lambda item: _stable_order(seed, item))
    for group_id in real_only_groups:
        if current_real >= real_target:
            break
        val_groups.add(group_id)
        current_real += group_counts[group_id]["real"]

    for row in rows:
        if row["split"] != "test":
            row["split"] = "val" if row["group_id"] in val_groups else "train"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data/celeb-df-v2", type=Path)
    parser.add_argument("--output", default="data/celeb-df-v2/celebdf_manifest.csv", type=Path)
    parser.add_argument("--val-ratio", default=0.15, type=float)
    parser.add_argument("--seed", default=42, type=int)
    args = parser.parse_args()

    if not 0 <= args.val_ratio <= 1:
        raise ValueError("--val-ratio must be between 0 and 1")

    root = args.root.resolve()
    official_test = _read_official_test(root)
    rows = []
    for class_dir, metadata in CLASS_DIRS.items():
        for path in sorted((root / class_dir).glob("*.mp4")):
            relative_path = _relative(path, root)
            stem = path.stem
            rows.append(
                {
                    "path": relative_path,
                    "label": metadata["label"],
                    "group_id": "",
                    "split": "test" if relative_path in official_test else "",
                    "video_id": relative_path,
                    "class_name": metadata["class_name"],
                    "compression": "",
                    "gen_method": metadata["gen_method"],
                    "resolution": "",
                    "media_type": "mp4",
                    "background_type": class_dir,
                    "video_name": stem,
                }
            )

    discovered = {row["path"] for row in rows}
    missing = sorted(set(official_test).difference(discovered))
    if missing:
        raise FileNotFoundError(f"Test list contains {len(missing)} undiscovered videos")

    _assign_train_val(rows, args.val_ratio, args.seed)
    rows.sort(key=lambda row: (row["split"], row["label"], row["path"]))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "path",
        "label",
        "group_id",
        "split",
        "video_id",
        "class_name",
        "compression",
        "gen_method",
        "resolution",
        "media_type",
        "background_type",
        "video_name",
    ]
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    counts = {}
    for row in rows:
        key = (row["split"], row["class_name"])
        counts[key] = counts.get(key, 0) + 1
    for key in sorted(counts):
        print(f"{key[0]:5s} {key[1]:4s} {counts[key]}")
    print(f"Wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
