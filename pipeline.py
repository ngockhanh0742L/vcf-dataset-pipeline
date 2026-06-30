import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os

from src.config import load_config, pipeline_fingerprint
from src.utils import setup_logger


def _write_json_atomic(payload, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    temp_path = f"{path}.tmp"
    with open(temp_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    os.replace(temp_path, path)


def _process_video(video, config, logger, fingerprint):
    from src.face_processor import process_faces
    from src.manifest import write_sequences
    from src.sampler import sample_candidates
    from src.selector import select_model_frames
    from src.sequence_builder import build_sequences
    from src.video_reader import read_frames_with_timestamps

    frames_iter = read_frames_with_timestamps(
        video.path,
        use_timestamps=config.pipeline.use_timestamp_sampling,
        min_fps_warning=config.pipeline.min_source_fps_warning,
        logger=logger,
    )
    candidates = sample_candidates(frames_iter, config.pipeline.candidate_fps)
    face_records = process_faces(
        candidates,
        config.face,
        video_id=video.video_id,
        target_size=(config.pipeline.image_size, config.pipeline.image_size),
    )
    model_frames = select_model_frames(
        face_records, config.pipeline, config.face, config.quality, config.motion
    )
    sequences = build_sequences(model_frames, config.pipeline, config.motion)
    rows = write_sequences(
        sequences, video, config.data, config.pipeline, fingerprint
    )
    return rows, len(sequences)


def run_preprocess(
    config,
    logger,
    selected_ids=None,
    validate_only=False,
    resume=None,
    overwrite=False,
    max_videos=None,
    workers=None,
):
    from src.datasets import discover_datasets
    from src.manifest import (
        build_summary,
        read_manifest,
        write_manifest,
        write_split_manifest,
    )

    logger.info("Starting dataset preprocessing...")
    videos, rejected = discover_datasets(config, logger, selected_ids)
    if not videos:
        raise ValueError("No valid videos were discovered")
    if max_videos is not None:
        if max_videos <= 0:
            raise ValueError("max_videos must be positive")
        videos = videos[:max_videos]
    fingerprint = pipeline_fingerprint(config)
    logger.info("Found %d valid videos; pipeline fingerprint=%s", len(videos), fingerprint)
    if validate_only:
        split_counts = {}
        class_counts = {}
        group_splits = {}
        for video in videos:
            split_counts[video.split] = split_counts.get(video.split, 0) + 1
            class_counts[video.class_name] = class_counts.get(video.class_name, 0) + 1
            group_splits[(video.dataset_id, video.group_id)] = video.split
        group_counts = {}
        for split in group_splits.values():
            group_counts[split] = group_counts.get(split, 0) + 1
        result = {
            "valid_videos": len(videos),
            "rejected_paths": rejected,
            "videos_by_split": split_counts,
            "videos_by_class": class_counts,
            "groups_by_split": group_counts,
            "pipeline_fingerprint": fingerprint,
        }
        logger.info("Input validation passed: %s", result)
        return result

    if overwrite and resume:
        raise ValueError("overwrite and resume cannot both be enabled")
    preserve_existing = not overwrite
    write_split_manifest(
        videos, config.data.split_manifest_path, preserve_existing=preserve_existing
    )
    existing = read_manifest(config.data.manifest_path) if preserve_existing else None
    if existing is not None and not existing.empty:
        fingerprints = {
            value
            for value in existing["pipeline_fingerprint"].astype(str)
            if value and value.lower() != "nan"
        }
        if fingerprints and fingerprints != {fingerprint}:
            raise RuntimeError(
                "Existing manifest was produced by a different pipeline fingerprint; "
                "use --overwrite or choose another data.output_dir"
            )
    if overwrite:
        write_manifest([], config.data.manifest_path, preserve_existing=False)
    resume = bool(getattr(config.data, "resume", True)) if resume is None else resume
    if overwrite:
        resume = False
    completed = set()
    if resume and existing is not None and not existing.empty:
        matching = existing[existing["pipeline_fingerprint"].astype(str) == fingerprint]
        completed = set(zip(matching["dataset_id"], matching["video_id"]))
        logger.info("Resume enabled: %d videos already have accepted sequences", len(completed))

    rows = []
    processed_video_ids = []
    failures = []
    workers = int(getattr(config.pipeline, "workers", 1)) if workers is None else workers
    if workers <= 0:
        raise ValueError("workers must be positive")
    pending = [
        video
        for video in videos
        if (video.dataset_id, video.video_id) not in completed
    ]
    logger.info(
        "Processing %d pending videos with %d worker(s)", len(pending), workers
    )
    checkpoint_every = int(
        getattr(config.pipeline, "checkpoint_every_videos", 25)
    )

    def record_result(video, future, index):
        try:
            video_rows, sequence_count = future.result()
            rows.extend(video_rows)
            processed_video_ids.append(video.video_id)
            logger.info(
                "[%d/%d] %s: %d accepted sequences",
                index,
                len(pending),
                video.video_id,
                sequence_count,
            )
        except Exception as exc:
            failures.append(
                {
                    "dataset_id": video.dataset_id,
                    "video_id": video.video_id,
                    "error": str(exc),
                }
            )
            logger.exception("Failed to process %s", video.video_id)

        attempted = len(processed_video_ids) + len(failures)
        if rows and attempted % checkpoint_every == 0:
            write_manifest(rows, config.data.manifest_path, preserve_existing=True)
            logger.info("Checkpointed %d sequence rows", len(rows))
            rows.clear()

    if workers == 1:
        for index, video in enumerate(pending, start=1):
            class ImmediateFuture:
                def result(self):
                    return _process_video(video, config, logger, fingerprint)

            record_result(video, ImmediateFuture(), index)
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_video = {
                executor.submit(_process_video, video, config, logger, fingerprint): video
                for video in pending
            }
            for index, future in enumerate(as_completed(future_to_video), start=1):
                record_result(future_to_video[future], future, index)

    manifest = write_manifest(
        rows, config.data.manifest_path, preserve_existing=True
    )
    summary = build_summary(manifest, videos, rejected, failures)
    summary["pipeline_fingerprint"] = fingerprint
    _write_json_atomic(summary, config.data.summary_path)
    logger.info(
        "Done: %d sequences from %d/%d videos (%d failures).",
        len(manifest),
        len(processed_video_ids),
        len(videos),
        len(failures),
    )
    if failures and config.data.fail_on_video_error:
        raise RuntimeError(
            f"Preprocessing completed with {len(failures)} failed video(s); "
            f"see {config.data.summary_path}"
        )
    return summary


def main():
    parser = argparse.ArgumentParser(description="Prepare video datasets for downstream training")
    parser.add_argument("--config", default="config.yaml", help="Path to config file")
    parser.add_argument("--log-file", help="Optional persistent log file")
    parser.add_argument(
        "--dataset", action="append", dest="datasets", help="Dataset id to process; repeatable"
    )
    parser.add_argument(
        "--validate-only", action="store_true", help="Validate inputs without writing output"
    )
    parser.add_argument(
        "--max-videos", type=int, help="Process only the first N videos (smoke tests)"
    )
    parser.add_argument(
        "--workers", type=int, help="Override pipeline worker count"
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--no-resume", action="store_true", help="Process all videos and upsert rows")
    mode.add_argument("--overwrite", action="store_true", help="Create fresh manifests")
    args = parser.parse_args()

    if args.log_file:
        os.makedirs(os.path.dirname(args.log_file) or ".", exist_ok=True)
    logger = setup_logger("pipeline", args.log_file)
    run_preprocess(
        load_config(args.config),
        logger,
        selected_ids=args.datasets,
        validate_only=args.validate_only,
        resume=False if args.no_resume else None,
        overwrite=args.overwrite,
        max_videos=args.max_videos,
        workers=args.workers,
    )


if __name__ == "__main__":
    main()
