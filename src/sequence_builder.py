import numpy as np


def build_sequences(model_frames, config_pipeline, config_motion):
    seq_len = int(config_pipeline.seq_len)
    hop = int(config_pipeline.window_hop)
    if seq_len <= 0 or hop <= 0:
        raise ValueError("seq_len and window_hop must be positive")
    if len(model_frames) < seq_len:
        return []

    accepted = []
    for start in range(0, len(model_frames) - seq_len + 1, hop):
        frames = model_frames[start : start + seq_len]
        missing_count = sum(frame.missing_face_flag for frame in frames)
        repair_count = sum(frame.repair_flag for frame in frames)
        comparable = [
            frame for frame in frames if frame.ssim_to_prev_selected is not None
        ]
        duplicate_count = sum(
            frame.ssim_to_prev_selected
            >= float(config_motion.ssim_redundant_threshold)
            for frame in comparable
        )

        missing_ratio = missing_count / seq_len
        repair_ratio = repair_count / seq_len
        duplicate_ratio = duplicate_count / max(1, len(comparable))
        mean_quality = float(np.mean([frame.quality_score for frame in frames]))
        mean_motion = float(np.mean([frame.motion_score for frame in comparable])) if comparable else 0.0

        if missing_ratio > float(config_motion.max_missing_face_ratio):
            continue
        if repair_ratio > float(config_motion.max_repair_ratio):
            continue
        if duplicate_ratio > float(config_motion.max_duplicate_ratio):
            continue
        if mean_motion < float(config_motion.min_motion_score):
            continue

        window_score = (
            0.60 * mean_quality
            + 0.25 * min(mean_motion / 0.2, 1.0)
            - 0.10 * duplicate_ratio
            - 0.05 * repair_ratio
        )
        accepted.append(
            {
                "frames": frames,
                "mean_quality": mean_quality,
                "mean_motion": mean_motion,
                "duplicate_ratio": duplicate_ratio,
                "missing_face_ratio": missing_ratio,
                "repair_ratio": repair_ratio,
                "window_score": window_score,
            }
        )

    max_sequences = int(config_pipeline.max_sequences_per_video)
    if max_sequences <= 0 or len(accepted) <= max_sequences:
        return accepted

    chunk_size = len(accepted) / max_sequences
    stratified = []
    for chunk_index in range(max_sequences):
        start = int(chunk_index * chunk_size)
        end = int((chunk_index + 1) * chunk_size)
        chunk = accepted[start:end]
        if chunk:
            stratified.append(max(chunk, key=lambda item: item["window_score"]))
    return stratified
