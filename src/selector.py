from src.face_processor import FaceRecord
from src.motion import compute_motion_score
from src.quality import compute_quality_score


def _missing_record(video_id, bin_index, timestamp):
    record = FaceRecord(video_id, -1, timestamp, bin_index)
    record.missing_face_flag = True
    record.hard_invalid = True
    return record


def _repair_record(record, bin_index, timestamp):
    repaired = FaceRecord(record.video_id, record.frame_index, timestamp, bin_index)
    repaired.face_image = record.face_image.copy()
    repaired.bbox = list(record.bbox) if record.bbox else None
    repaired.detect_confidence = record.detect_confidence
    repaired.repair_flag = True
    repaired.quality_score = record.quality_score
    repaired.motion_score = 0.0
    repaired.ssim_to_prev_selected = 1.0
    return repaired


def select_model_frames(
    face_records, config_pipeline, config_face, config_quality, config_motion
):
    """Select one quality-aware candidate per model-FPS time bin."""
    model_interval = 1.0 / float(config_pipeline.model_fps)
    bins = {}
    for record in face_records:
        bin_index = int(record.timestamp / model_interval)
        bins.setdefault(bin_index, []).append(record)

    if not bins:
        return []

    selected = []
    last_valid = None
    consecutive_repairs = 0
    max_repairs = int(config_face.max_consecutive_repairs)
    video_id = face_records[0].video_id

    for bin_index in range(max(bins) + 1):
        timestamp = bin_index * model_interval
        best_record = None
        best_score = float("-inf")
        for record in bins.get(bin_index, []):
            if not compute_quality_score(record, config_quality):
                continue
            compute_motion_score(record, last_valid)
            normalized_motion = min(record.motion_score / 0.2, 1.0)
            bin_center = timestamp + model_interval / 2
            temporal_score = max(
                0.0,
                1.0
                - abs(record.timestamp - bin_center) / (model_interval / 2),
            )
            score = (
                0.65 * record.quality_score
                + 0.25 * normalized_motion
                + 0.10 * temporal_score
            )
            if score > best_score:
                best_score = score
                best_record = record

        if best_record is not None:
            selected.append(best_record)
            last_valid = best_record
            consecutive_repairs = 0
        elif last_valid is not None and consecutive_repairs < max_repairs:
            selected.append(_repair_record(last_valid, bin_index, timestamp))
            consecutive_repairs += 1
        else:
            selected.append(_missing_record(video_id, bin_index, timestamp))
            consecutive_repairs += 1
    return selected
