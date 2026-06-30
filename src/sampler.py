def sample_candidates(frames_iter, candidate_fps):
    """Stream at most one frame for every candidate-FPS timestamp."""
    candidate_fps = float(candidate_fps)
    if candidate_fps <= 0:
        raise ValueError("candidate_fps must be positive")

    interval = 1.0 / candidate_fps
    next_timestamp = 0.0
    candidate_id = 0
    for frame, timestamp, frame_index in frames_iter:
        if timestamp + 1e-9 < next_timestamp:
            continue
        yield {
            "frame": frame,
            "timestamp": timestamp,
            "frame_idx": frame_index,
            "candidate_id": candidate_id,
        }
        candidate_id += 1
        while next_timestamp <= timestamp + 1e-9:
            next_timestamp += interval
