from collections import Counter
import json
import os
from pathlib import Path
import uuid

import cv2
import pandas as pd


MANIFEST_COLUMNS = [
    "sequence_id",
    "dataset_id",
    "video_id",
    "group_id",
    "label",
    "class_name",
    "split",
    "compression",
    "gen_method",
    "resolution",
    "media_type",
    "background_type",
    "video_name",
    "frame_paths",
    "start_time",
    "end_time",
    "start_frame",
    "end_frame",
    "candidate_fps",
    "model_fps",
    "seq_len",
    "image_size",
    "image_format",
    "avg_quality",
    "avg_motion",
    "duplicate_ratio",
    "missing_face_ratio",
    "repair_ratio",
    "window_score",
    "pipeline_version",
    "pipeline_fingerprint",
]


def _atomic_csv(dataframe, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    temp_path = f"{path}.tmp"
    dataframe.to_csv(temp_path, index=False)
    os.replace(temp_path, path)


def write_split_manifest(videos, path, preserve_existing=True):
    columns = [
        "dataset_id",
        "video_id",
        "group_id",
        "split",
        "label",
        "class_name",
        "compression",
        "gen_method",
        "resolution",
        "media_type",
        "background_type",
        "video_name",
    ]
    rows = [{column: getattr(video, column) for column in columns} for video in videos]
    dataframe = pd.DataFrame(rows, columns=columns)
    if preserve_existing and os.path.exists(path):
        existing = pd.read_csv(path)
        dataframe = pd.concat([existing, dataframe], ignore_index=True)
        dataframe = dataframe.drop_duplicates(["dataset_id", "video_id"], keep="last")
    if not dataframe.empty:
        dataframe = dataframe.sort_values(["dataset_id", "video_id"])
    _atomic_csv(dataframe, path)


def _sequence_id(sequence, video, config_pipeline, fingerprint):
    frames = sequence["frames"]
    identity = "|".join(
        [
            video.dataset_id,
            video.video_id,
            str(frames[0].frame_index),
            str(frames[-1].frame_index),
            str(config_pipeline.seq_len),
            str(config_pipeline.model_fps),
            str(config_pipeline.image_size),
            f"{frames[0].timestamp:.9f}",
            f"{frames[-1].timestamp:.9f}",
            fingerprint,
        ]
    )
    return str(uuid.uuid5(uuid.NAMESPACE_URL, identity))


def write_sequences(sequences, video, config_data, config_pipeline, fingerprint):
    image_format = config_data.output_format.lower().lstrip(".")
    if image_format not in {"png", "jpg", "jpeg"}:
        raise ValueError("output_format must be png, jpg, or jpeg")
    extension = "jpg" if image_format == "jpeg" else image_format
    rows = []

    for sequence in sequences:
        sequence_id = _sequence_id(sequence, video, config_pipeline, fingerprint)
        folder = Path(config_data.sequence_dir) / sequence_id
        folder.mkdir(parents=True, exist_ok=True)
        frame_paths = []
        for index, record in enumerate(sequence["frames"]):
            if record.face_image is None:
                raise ValueError("Accepted sequence contains an empty face frame")
            filename = f"frame_{index:02d}.{extension}"
            output_path = folder / filename
            image = cv2.cvtColor(record.face_image, cv2.COLOR_RGB2BGR)
            params = (
                [cv2.IMWRITE_PNG_COMPRESSION, 3]
                if extension == "png"
                else [
                    cv2.IMWRITE_JPEG_QUALITY,
                    int(getattr(config_data, "jpeg_quality", 95)),
                ]
            )
            if not cv2.imwrite(str(output_path), image, params):
                raise OSError(f"Could not write image: {output_path}")
            frame_paths.append(f"{sequence_id}/{filename}")

        frames = sequence["frames"]
        rows.append(
            {
                "sequence_id": sequence_id,
                "dataset_id": video.dataset_id,
                "video_id": video.video_id,
                "group_id": video.group_id,
                "label": video.label,
                "class_name": video.class_name,
                "split": video.split,
                "compression": video.compression,
                "gen_method": video.gen_method,
                "resolution": video.resolution,
                "media_type": video.media_type,
                "background_type": video.background_type,
                "video_name": video.video_name,
                "frame_paths": json.dumps(frame_paths),
                "start_time": frames[0].timestamp,
                "end_time": frames[-1].timestamp,
                "start_frame": frames[0].frame_index,
                "end_frame": frames[-1].frame_index,
                "candidate_fps": config_pipeline.candidate_fps,
                "model_fps": config_pipeline.model_fps,
                "seq_len": config_pipeline.seq_len,
                "image_size": config_pipeline.image_size,
                "image_format": extension,
                "avg_quality": sequence["mean_quality"],
                "avg_motion": sequence["mean_motion"],
                "duplicate_ratio": sequence["duplicate_ratio"],
                "missing_face_ratio": sequence["missing_face_ratio"],
                "repair_ratio": sequence["repair_ratio"],
                "window_score": sequence["window_score"],
                "pipeline_version": "vcf_etl_v2",
                "pipeline_fingerprint": fingerprint,
            }
        )
    return rows


def read_manifest(path):
    if not os.path.exists(path):
        return pd.DataFrame(columns=MANIFEST_COLUMNS)
    dataframe = pd.read_csv(path)
    for column in MANIFEST_COLUMNS:
        if column not in dataframe:
            dataframe[column] = ""
    return dataframe.reindex(columns=MANIFEST_COLUMNS)


def write_manifest(rows, path, preserve_existing=True):
    new = pd.DataFrame(rows, columns=MANIFEST_COLUMNS)
    if preserve_existing and os.path.exists(path):
        new = pd.concat([read_manifest(path), new], ignore_index=True)
    if not new.empty:
        new = new.drop_duplicates("sequence_id", keep="last")
        new = new.sort_values(
            ["split", "label", "dataset_id", "video_id", "start_time"]
        )
    new = new.reindex(columns=MANIFEST_COLUMNS)
    _atomic_csv(new, path)
    return new


def build_summary(manifest, videos, rejected, failures):
    split_counts = Counter(manifest["split"].tolist()) if not manifest.empty else Counter()
    class_counts = Counter(manifest["class_name"].tolist()) if not manifest.empty else Counter()
    dataset_counts = Counter(manifest["dataset_id"].tolist()) if not manifest.empty else Counter()
    return {
        "discovered_videos": len(videos) + len(rejected),
        "valid_videos": len(videos),
        "rejected_paths": rejected,
        "failed_videos": failures,
        "sequence_count": int(len(manifest)),
        "sequences_by_split": dict(sorted(split_counts.items())),
        "sequences_by_class": dict(sorted(class_counts.items())),
        "sequences_by_dataset": dict(sorted(dataset_counts.items())),
        "note": "Use split as provided; never randomly split sequence rows.",
    }
