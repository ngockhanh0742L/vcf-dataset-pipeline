from skimage.metrics import structural_similarity as ssim
import cv2

def compute_motion_score(record, last_selected_record, config_motion):
    if record.hard_invalid or record.face_image is None:
        record.motion_score = 0.0
        record.ssim_to_prev_selected = None
        return
        
    if last_selected_record is None or last_selected_record.face_image is None:
        record.motion_score = 1.0 # Max motion if no previous
        record.ssim_to_prev_selected = 0.0
        return
        
    gray_current = cv2.cvtColor(record.face_image, cv2.COLOR_RGB2GRAY)
    gray_prev = cv2.cvtColor(last_selected_record.face_image, cv2.COLOR_RGB2GRAY)
    
    score, _ = ssim(gray_current, gray_prev, full=True)
    record.ssim_to_prev_selected = float(score)
    
    motion_score = 1.0 - score
    record.motion_score = float(motion_score)
