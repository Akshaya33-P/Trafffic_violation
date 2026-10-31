import os
import uuid
import cv2
import numpy as np
from typing import Tuple, Dict, Any, Optional
from src.utils.privacy import apply_privacy_blur

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EVIDENCE_DIR = os.path.join(PROJECT_ROOT, "evidence")
CROPS_DIR = os.path.join(EVIDENCE_DIR, "crops")
FULL_DIR = os.path.join(EVIDENCE_DIR, "full")

os.makedirs(CROPS_DIR, exist_ok=True)
os.makedirs(FULL_DIR, exist_ok=True)

def save_evidence(
    frame: np.ndarray,
    bbox: Tuple[int, int, int, int],
    violation_type: str,
    track_id: int,
    confidence: float,
    video_timestamp: float,
    object_type: str = "vehicle",
    apply_blur: bool = False
) -> Tuple[str, str]:
    """
    Saves:
    1. Cropped image of the violating vehicle/rider with margin.
    2. Full-frame image with the violating object clearly highlighted with bounding box,
       violation banner, track ID, and decision-support disclaimer watermark.
    Returns:
        (crop_relative_path, full_relative_path)
    """
    h_img, w_img = frame.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in bbox]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w_img, x2), min(h_img, y2)

    # Apply privacy blur if requested
    processed_frame = frame.copy()
    if apply_blur:
        processed_frame = apply_privacy_blur(processed_frame, (x1, y1, x2, y2), object_type)

    # 1. Save Cropped Vehicle/Subject with 15% margin
    margin_x = int(0.15 * (x2 - x1))
    margin_y = int(0.15 * (y2 - y1))
    cx1 = max(0, x1 - margin_x)
    cy1 = max(0, y1 - margin_y)
    cx2 = min(w_img, x2 + margin_x)
    cy2 = min(h_img, y2 + margin_y)

    crop_img = processed_frame[cy1:cy2, cx1:cx2].copy()
    if crop_img.size == 0:
        crop_img = processed_frame[y1:y2, x1:x2].copy()

    file_id = f"{uuid.uuid4().hex[:10]}_track_{track_id}_{int(video_timestamp*1000)}"
    crop_filename = f"crop_{file_id}.jpg"
    full_filename = f"full_{file_id}.jpg"

    crop_path = os.path.join(CROPS_DIR, crop_filename)
    full_path = os.path.join(FULL_DIR, full_filename)

    cv2.imwrite(crop_path, crop_img)

    # 2. Prepare Annotated Full Frame
    annotated_full = processed_frame.copy()

    # Draw vibrant highlight box around vehicle (Red for violations)
    box_color = (0, 0, 240)  # BGR Red
    cv2.rectangle(annotated_full, (x1, y1), (x2, y2), box_color, 3)

    # Top Banner with label, track ID, and confidence
    label = f"FLAGGED: {violation_type.upper()} | ID #{track_id} | Conf: {confidence:.2f} | Time: {video_timestamp:.1f}s"
    (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    bg_y1 = max(0, y1 - th - 12)
    bg_y2 = y1
    cv2.rectangle(annotated_full, (x1, bg_y1), (x1 + tw + 10, bg_y2), box_color, -1)
    cv2.putText(
        annotated_full,
        label,
        (x1 + 5, y1 - 6),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    # Mandatory Legal Decision-Support Disclaimer Watermark
    watermark = "DECISION-SUPPORT ONLY - PENDING HUMAN REVIEW"
    cv2.putText(
        annotated_full,
        watermark,
        (20, h_img - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 215, 255),  # Yellow-Amber
        2,
        cv2.LINE_AA
    )

    cv2.imwrite(full_path, annotated_full)

    # Store relative paths for database persistence
    rel_crop = os.path.relpath(crop_path, PROJECT_ROOT)
    rel_full = os.path.relpath(full_path, PROJECT_ROOT)

    return rel_crop, rel_full
