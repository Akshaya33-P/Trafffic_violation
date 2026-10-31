import cv2
import numpy as np
from typing import Tuple, Optional

def apply_privacy_blur(image: np.ndarray, bbox: Tuple[int, int, int, int], object_type: str = "vehicle") -> np.ndarray:
    """
    Applies Gaussian privacy blur to privacy-sensitive regions:
    - For vehicles: Blurs the lower central portion where license plates typically appear.
    - For riders/persons: Blurs the head/face region.
    """
    if image is None or image.size == 0:
        return image

    img_out = image.copy()
    h_img, w_img = img_out.shape[:2]
    x1, y1, x2, y2 = bbox
    x1, y1 = max(0, int(x1)), max(0, int(y1))
    x2, y2 = min(w_img, int(x2)), min(h_img, int(y2))
    bw = x2 - x1
    bh = y2 - y1

    if bw <= 4 or bh <= 4:
        return img_out

    if object_type in ["car", "bus", "truck"]:
        # Blur lower 30% of the vehicle box where license plates reside
        py1 = int(y1 + 0.70 * bh)
        py2 = y2
        px1 = int(x1 + 0.20 * bw)
        px2 = int(x1 + 0.80 * bw)
        roi = img_out[py1:py2, px1:px2]
        if roi.size > 0:
            ksize = max(15, (roi.shape[1] // 3) | 1)
            img_out[py1:py2, px1:px2] = cv2.GaussianBlur(roi, (ksize, ksize), 30)

    elif object_type in ["motorcycle", "rider", "person"]:
        # Blur the upper 40% where face / rider head resides
        py1 = y1
        py2 = int(y1 + 0.40 * bh)
        px1 = int(x1 + 0.15 * bw)
        px2 = int(x1 + 0.85 * bw)
        roi = img_out[py1:py2, px1:px2]
        if roi.size > 0:
            ksize = max(15, (roi.shape[1] // 3) | 1)
            img_out[py1:py2, px1:px2] = cv2.GaussianBlur(roi, (ksize, ksize), 30)

    return img_out
