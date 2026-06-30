import cv2
from skimage.metrics import structural_similarity


def compute_motion_score(record, last_selected_record):
    if record.hard_invalid or record.face_image is None:
        record.motion_score = 0.0
        record.ssim_to_prev_selected = None
        return
    if last_selected_record is None or last_selected_record.face_image is None:
        record.motion_score = 0.0
        record.ssim_to_prev_selected = None
        return

    current = cv2.cvtColor(record.face_image, cv2.COLOR_RGB2GRAY)
    previous = cv2.cvtColor(last_selected_record.face_image, cv2.COLOR_RGB2GRAY)
    score = float(structural_similarity(current, previous, data_range=255))
    record.ssim_to_prev_selected = score
    record.motion_score = max(0.0, 1.0 - score)
