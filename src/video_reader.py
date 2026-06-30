import os

import cv2


VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv")


def list_videos(raw_video_dir):
    videos = []
    for root, _, files in os.walk(raw_video_dir):
        for filename in files:
            if filename.lower().endswith(VIDEO_EXTENSIONS):
                videos.append(os.path.join(root, filename))
    return sorted(videos)


def read_frames_with_timestamps(
    video_path, use_timestamps=True, min_fps_warning=0.0, logger=None
):
    """Yield ``(frame, timestamp_seconds, frame_index)`` from a video."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if fps <= 0:
        fps = 30.0
        if logger:
            logger.warning("Invalid FPS for %s; using fallback %.1f", video_path, fps)
    elif logger and fps < min_fps_warning:
        logger.warning("Low source FPS %.2f for %s", fps, video_path)

    frame_index = 0
    previous_timestamp = -1.0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            timestamp = frame_index / fps
            if use_timestamps:
                container_timestamp = float(cap.get(cv2.CAP_PROP_POS_MSEC)) / 1000.0
                if container_timestamp >= 0 and container_timestamp > previous_timestamp:
                    timestamp = container_timestamp
            previous_timestamp = timestamp
            yield frame, timestamp, frame_index
            frame_index += 1
    finally:
        cap.release()
