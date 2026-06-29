import cv2
import numpy as np

def compute_quality_score(record, config_quality):
    if record.hard_invalid:
        record.quality_score = 0.0
        return
        
    img = record.face_image
    if img is None:
        record.quality_score = 0.0
        return
        
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    # Blur
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    blur_score = np.clip(laplacian_var / (config_quality.min_blur * 2), 0, 1)
    
    # Brightness
    mean_intensity = np.mean(gray)
    if mean_intensity < config_quality.min_brightness or mean_intensity > config_quality.max_brightness:
        brightness_score = 0.0
    else:
        brightness_score = 1.0 - abs(mean_intensity - 128) / 128.0
        
    # Contrast
    contrast_std = np.std(gray)
    contrast_score = np.clip(contrast_std / (config_quality.min_contrast * 2), 0, 1)
    
    # Face size (approximate using max of bbox width/height compared to 300)
    # The bbox gives original image scale, but since we already have 300x300, 
    # we can use the bbox area over some standard.
    # Actually, we can just use the detect_confidence directly as a proxy for face size/quality as well.
    detect_score = record.detect_confidence
    
    # Simple weighted sum
    quality_score = 0.3 * blur_score + 0.2 * brightness_score + 0.2 * contrast_score + 0.3 * detect_score
    
    record.quality_score = float(np.clip(quality_score, 0, 1))
