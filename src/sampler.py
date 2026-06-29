def sample_candidates(frames_iter, candidate_fps):
    """
    Samples frames from an iterator yielding (frame, timestamp, frame_idx).
    Extracts one candidate per candidate_time interval.
    candidate_time = k * (1 / candidate_fps)
    """
    candidate_interval = 1.0 / candidate_fps
    next_candidate_time = 0.0
    
    candidates = []
    
    for frame, timestamp, frame_idx in frames_iter:
        if timestamp >= next_candidate_time:
            # Simple nearest approach for streaming simulation:
            # Pick the first frame that crosses the target timestamp.
            candidates.append({
                'frame': frame,
                'timestamp': timestamp,
                'frame_idx': frame_idx,
                'candidate_id': len(candidates)
            })
            next_candidate_time += candidate_interval
            
    return candidates
