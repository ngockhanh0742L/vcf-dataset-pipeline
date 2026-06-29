import numpy as np

def select_model_frames(face_records, config_pipeline, config_quality, config_motion):
    """
    Selects one frame per model_fps bin from candidate frames.
    Returns a list of selected FaceRecords.
    """
    model_interval = 1.0 / config_pipeline.model_fps
    
    # Group candidates into bins
    bins = {}
    for record in face_records:
        bin_idx = int(record.timestamp / model_interval)
        if bin_idx not in bins:
            bins[bin_idx] = []
        bins[bin_idx].append(record)
        
    selected_frames = []
    last_selected = None
    
    max_bin = max(bins.keys()) if bins else -1
    
    for bin_idx in range(max_bin + 1):
        candidates = bins.get(bin_idx, [])
        
        if not candidates:
            # Missing bin, repair from last valid face
            if last_selected is not None:
                repaired = copy_record_for_repair(last_selected, bin_idx * model_interval)
                selected_frames.append(repaired)
            continue
            
        best_record = None
        best_score = -9999.0
        
        for record in candidates:
            if record.hard_invalid:
                continue
                
            from src.quality import compute_quality_score
            from src.motion import compute_motion_score
            
            compute_quality_score(record, config_quality)
            compute_motion_score(record, last_selected, config_motion)
            
            # Normalize motion to some reasonable scale (e.g., max expected motion ~ 0.2)
            norm_motion = min(record.motion_score / 0.2, 1.0)
            
            # Temporal center score: prefer candidates closer to center of bin
            bin_center = bin_idx * model_interval + (model_interval / 2)
            dist_to_center = abs(record.timestamp - bin_center)
            temporal_score = 1.0 - (dist_to_center / (model_interval / 2))
            
            selection_score = (
                0.65 * record.quality_score +
                0.25 * norm_motion +
                0.10 * temporal_score
            )
            
            if selection_score > best_score:
                best_score = selection_score
                best_record = record
                
        if best_record:
            selected_frames.append(best_record)
            last_selected = best_record
        else:
            # All candidates invalid, repair
            if last_selected is not None:
                repaired = copy_record_for_repair(last_selected, bin_idx * model_interval)
                selected_frames.append(repaired)
                
    return selected_frames

def copy_record_for_repair(record, new_timestamp):
    from src.face_processor import FaceRecord
    new_record = FaceRecord(record.video_id, record.frame_index, new_timestamp, record.candidate_id)
    new_record.face_image = record.face_image
    new_record.bbox = record.bbox
    new_record.repair_flag = True
    new_record.quality_score = record.quality_score
    new_record.motion_score = 0.0
    new_record.ssim_to_prev_selected = 1.0
    return new_record
