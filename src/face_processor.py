import cv2
import numpy as np

class FaceRecord:
    def __init__(self, video_id, frame_index, timestamp, candidate_id):
        self.video_id = video_id
        self.frame_index = frame_index
        self.timestamp = timestamp
        self.candidate_id = candidate_id
        self.face_image = None
        self.bbox = None
        self.landmarks = None
        self.detect_confidence = 0.0
        self.repair_flag = False
        self.missing_face_flag = False
        self.quality_score = 0.0
        self.motion_score = 0.0
        self.ssim_to_prev_selected = None
        self.hard_invalid = False

def process_faces(candidates, config_face, video_id, target_size=(300, 300)):
    """
    Extracts faces from candidates.
    Returns a list of FaceRecords.
    """
    import mediapipe as mp
    mp_face_detection = mp.solutions.face_detection
    detector = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)
    
    face_records = []
    
    for cand in candidates:
        record = FaceRecord(
            video_id=video_id,
            frame_index=cand['frame_idx'],
            timestamp=cand['timestamp'],
            candidate_id=cand['candidate_id']
        )
        
        frame = cand['frame']
        h, w = frame.shape[:2]
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = detector.process(rgb_frame)
        
        if not results.detections:
            record.missing_face_flag = True
            record.hard_invalid = True
        else:
            # Select largest face
            best_det = None
            max_area = 0
            for det in results.detections:
                bboxC = det.location_data.relative_bounding_box
                area = bboxC.width * bboxC.height
                if area > max_area:
                    max_area = area
                    best_det = det
            
            if max_area < config_face.min_face_box_ratio:
                record.hard_invalid = True
            
            if best_det:
                bboxC = best_det.location_data.relative_bounding_box
                xmin = int(bboxC.xmin * w)
                ymin = int(bboxC.ymin * h)
                width = int(bboxC.width * w)
                height = int(bboxC.height * h)
                
                # Add 30% margin
                cx = xmin + width // 2
                cy = ymin + height // 2
                size = max(width, height)
                margin_size = int(size * 1.3)
                
                x1 = max(0, cx - margin_size // 2)
                y1 = max(0, cy - margin_size // 2)
                x2 = min(w, cx + margin_size // 2)
                y2 = min(h, cy + margin_size // 2)
                
                if x2 - x1 > 10 and y2 - y1 > 10:
                    face_crop = rgb_frame[y1:y2, x1:x2]
                    face_resized = cv2.resize(face_crop, target_size)
                    
                    record.face_image = face_resized
                    record.bbox = [x1, y1, x2, y2]
                    record.detect_confidence = best_det.score[0]
                else:
                    record.missing_face_flag = True
                    record.hard_invalid = True
            else:
                record.missing_face_flag = True
                record.hard_invalid = True
                
        face_records.append(record)
        
    detector.close()
    return face_records
