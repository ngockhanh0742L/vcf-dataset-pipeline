import numpy as np

def build_sequences(model_frames, config_pipeline):
    """
    Builds sequences of length seq_len from model_frames using sliding window hop.
    """
    seq_len = config_pipeline.seq_len
    hop = config_pipeline.window_hop_train
    
    sequences = []
    
    if len(model_frames) < seq_len:
        return sequences
        
    for i in range(0, len(model_frames) - seq_len + 1, hop):
        seq = model_frames[i:i+seq_len]
        
        # Calculate window score
        mean_quality = np.mean([f.quality_score for f in seq])
        mean_motion = np.mean([f.motion_score for f in seq])
        
        redundant_count = sum(1 for f in seq if f.ssim_to_prev_selected is not None and f.ssim_to_prev_selected >= 0.995)
        duplicate_ratio = redundant_count / seq_len
        
        missing_repaired_count = sum(1 for f in seq if f.repair_flag or f.missing_face_flag)
        missing_or_repaired_ratio = missing_repaired_count / seq_len
        
        window_score = (
            0.50 * mean_quality +
            0.25 * mean_motion -
            0.15 * duplicate_ratio -
            0.10 * missing_or_repaired_ratio
        )
        
        sequences.append({
            'frames': seq,
            'mean_quality': mean_quality,
            'mean_motion': mean_motion,
            'duplicate_ratio': duplicate_ratio,
            'missing_or_repaired_ratio': missing_or_repaired_ratio,
            'window_score': window_score
        })
        
    # Temporal stratification (max sequences per video)
    max_seq = config_pipeline.max_sequences_per_video
    if len(sequences) > max_seq:
        # Divide into max_seq chunks and pick best window_score in each
        chunk_size = len(sequences) / max_seq
        selected_seqs = []
        for c in range(max_seq):
            start = int(c * chunk_size)
            end = int((c + 1) * chunk_size)
            chunk = sequences[start:end]
            if chunk:
                best_in_chunk = max(chunk, key=lambda x: x['window_score'])
                selected_seqs.append(best_in_chunk)
        sequences = selected_seqs
        
    return sequences
