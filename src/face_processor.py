import cv2


class FaceRecord:
    def __init__(self, video_id, frame_index, timestamp, candidate_id):
        self.video_id = video_id
        self.frame_index = frame_index
        self.timestamp = timestamp
        self.candidate_id = candidate_id
        self.face_image = None
        self.bbox = None
        self.detect_confidence = 0.0
        self.repair_flag = False
        self.missing_face_flag = False
        self.quality_score = 0.0
        self.motion_score = 0.0
        self.ssim_to_prev_selected = None
        self.hard_invalid = False


def _select_detection(detections, select_largest):
    if select_largest:
        return max(
            detections,
            key=lambda det: (
                det.location_data.relative_bounding_box.width
                * det.location_data.relative_bounding_box.height
            ),
        )
    return max(detections, key=lambda det: float(det.score[0]))


def process_faces(candidates, config_face, video_id, target_size):
    if config_face.detector != "mediapipe":
        raise ValueError(f"Unsupported face detector: {config_face.detector}")

    import mediapipe as mp

    detector = mp.solutions.face_detection.FaceDetection(
        model_selection=int(config_face.model_selection),
        min_detection_confidence=float(config_face.min_detection_confidence),
    )
    records = []
    try:
        for candidate in candidates:
            record = FaceRecord(
                video_id,
                candidate["frame_idx"],
                candidate["timestamp"],
                candidate["candidate_id"],
            )
            frame = candidate["frame"]
            height, width = frame.shape[:2]
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = detector.process(rgb_frame)

            if not results.detections:
                record.missing_face_flag = True
                record.hard_invalid = True
                records.append(record)
                continue

            detection = _select_detection(
                results.detections, bool(config_face.select_largest_face)
            )
            box = detection.location_data.relative_bounding_box
            relative_area = max(0.0, box.width) * max(0.0, box.height)
            if relative_area < float(config_face.min_face_box_ratio):
                record.hard_invalid = True

            xmin = int(box.xmin * width)
            ymin = int(box.ymin * height)
            box_width = int(box.width * width)
            box_height = int(box.height * height)
            center_x = xmin + box_width // 2
            center_y = ymin + box_height // 2
            crop_size = int(max(box_width, box_height) * float(config_face.crop_margin))
            x1 = max(0, center_x - crop_size // 2)
            y1 = max(0, center_y - crop_size // 2)
            x2 = min(width, center_x + crop_size // 2)
            y2 = min(height, center_y + crop_size // 2)

            if x2 - x1 <= 10 or y2 - y1 <= 10:
                record.missing_face_flag = True
                record.hard_invalid = True
            else:
                crop = rgb_frame[y1:y2, x1:x2]
                record.face_image = cv2.resize(crop, target_size)
                record.bbox = [x1, y1, x2, y2]
                record.detect_confidence = float(detection.score[0])
            records.append(record)
    finally:
        detector.close()
    return records
