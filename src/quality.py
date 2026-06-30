import cv2
import numpy as np


def compute_quality_score(record, config_quality):
    if record.hard_invalid or record.face_image is None:
        record.quality_score = 0.0
        return False

    gray = cv2.cvtColor(record.face_image, cv2.COLOR_RGB2GRAY)
    laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    mean_intensity = float(np.mean(gray))
    contrast_std = float(np.std(gray))

    passes = (
        laplacian_var >= float(config_quality.min_blur)
        and float(config_quality.min_brightness)
        <= mean_intensity
        <= float(config_quality.max_brightness)
        and contrast_std >= float(config_quality.min_contrast)
    )
    blur_score = np.clip(laplacian_var / (float(config_quality.min_blur) * 2), 0, 1)
    brightness_score = 1.0 - abs(mean_intensity - 127.5) / 127.5
    contrast_score = np.clip(
        contrast_std / (float(config_quality.min_contrast) * 2), 0, 1
    )
    record.quality_score = float(
        np.clip(
            0.3 * blur_score
            + 0.2 * brightness_score
            + 0.2 * contrast_score
            + 0.3 * record.detect_confidence,
            0,
            1,
        )
    )
    if bool(config_quality.enforce_thresholds) and not passes:
        record.hard_invalid = True
        return False
    return True
