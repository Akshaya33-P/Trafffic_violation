import os
import cv2
import numpy as np
from typing import Tuple, Dict, Any, Optional

class PhoneCallDetector:
    """
    Tier A / B Safety Violation Detector: Mobile Phone Usage / Calling While Operating Vehicle.
    Analyzes:
    1. Direct 'cell phone' object detections (COCO class 67).
    2. Hand-to-Ear spatial posture: Evaluates whether a hand (skin-tone cluster)
       is elevated directly to the ear/temple level (upper 10%-32% of rider body)
       holding a compact device, indicating an active phone call while driving/riding.
    """
    def __init__(self, confidence_threshold: float = 0.65):
        self.confidence_threshold = confidence_threshold

    def check_phone_call(
        self,
        frame: np.ndarray,
        person_box: Tuple[float, float, float, float],
        detected_phone_boxes: Optional[list] = None
    ) -> Tuple[bool, float, Dict[str, Any], Optional[Tuple[int, int, int, int]]]:
        """
        Evaluates whether the person is holding a phone to their ear.
        Returns:
            (is_on_call, confidence, metadata, call_roi_box)
        """
        h_img, w_img = frame.shape[:2]
        px1, py1, px2, py2 = [int(v) for v in person_box]
        px1, py1 = max(0, px1), max(0, py1)
        px2, py2 = min(w_img, px2), min(h_img, py2)
        pw = px2 - px1
        ph = py2 - py1

        if pw < 15 or ph < 25:
            return False, 0.0, {"reason": "Low resolution"}, None

        # 1. Direct phone object overlap check
        if detected_phone_boxes:
            for pbox in detected_phone_boxes:
                cx1, cy1, cx2, cy2 = [int(v) for v in pbox]
                if px1 <= (cx1 + cx2)/2 <= px2 and py1 <= (cy1 + cy2)/2 <= py1 + int(0.40 * ph):
                    conf = 0.94
                    return True, conf, {"method": "direct_cell_phone_detector", "phone_box": pbox}, (cx1, cy1, cx2, cy2)

        # 2. Hand-to-Ear posture analysis (lateral ear zones at head height only)
        #
        # IMPORTANT: This heuristic is disabled for small crops (< 45px wide or < 60px tall)
        # because the ear zones become just 1-3 pixels wide and fill entirely with
        # background/skin tone, causing near-100% false positives.
        if pw < 45 or ph < 60:
            return False, 0.0, {"reason": "Person crop too small for reliable posture analysis"}, None

        rider_crop = frame[py1:py2, px1:px2]
        if rider_crop.size == 0:
            return False, 0.0, {"reason": "Empty crop"}, None

        # Guard: if the ENTIRE crop is overwhelmingly skin-toned (> 55 %), the
        # ear zones will be meaningless — skip posture analysis to avoid false positives
        # on tiny crops that land on a person's bare arm or face.
        hsv_full = cv2.cvtColor(rider_crop, cv2.COLOR_BGR2HSV)
        full_skin_mask = cv2.inRange(hsv_full, (0, 20, 40), (25, 210, 245))
        full_skin_frac = float(np.mean(full_skin_mask > 0))
        if full_skin_frac > 0.55:
            return False, 0.0, {"reason": f"Entire crop is skin ({full_skin_frac:.2f}); unreliable"}, None

        # Head/Ear level is strictly vertical 8% to 28% of torso height
        y_top = int(0.08 * ph)
        y_bottom = int(0.28 * ph)

        # Right ear zone: outer 35% of width
        right_ear = rider_crop[y_top:y_bottom, int(0.65 * pw):pw]
        # Left ear zone: outer 35% of width
        left_ear = rider_crop[y_top:y_bottom, 0:int(0.35 * pw)]

        def analyze_ear_zone(zone_img):
            if zone_img.size == 0:
                return 0.0, 0.0
            hsv = cv2.cvtColor(zone_img, cv2.COLOR_BGR2HSV)
            # Skin tone mask (hand holding phone)
            skin_mask = cv2.inRange(hsv, (0, 20, 40), (25, 210, 245))
            skin_frac = float(np.mean(skin_mask > 0))
            # Dark phone body / phone bezel
            phone_mask = cv2.inRange(hsv, (0, 0, 0), (180, 255, 65))
            phone_frac = float(np.mean(phone_mask > 0))
            return skin_frac, phone_frac

        r_skin, r_phone = analyze_ear_zone(right_ear)
        l_skin, l_phone = analyze_ear_zone(left_ear)

        # Hand raised to ear level holding phone.
        # Thresholds are intentionally strict to minimise false positives on
        # normal seated postures and auto-rickshaw passengers:
        #   skin  > 0.28  — a clearly raised hand/arm must fill most of the zone
        #   phone > 0.12  — a distinct dark phone body must be present
        is_right_call = (r_skin > 0.28 and r_phone > 0.12)
        is_left_call  = (l_skin > 0.28 and l_phone > 0.12)

        if is_right_call:
            conf = min(0.96, 0.75 + 0.6 * r_skin)
            roi_box = (
                px1 + int(0.60 * pw),
                py1 + y_top,
                px2,
                py1 + y_bottom
            )
            return True, round(conf, 3), {
                "side": "right_ear",
                "hand_skin_ratio": round(r_skin, 3),
                "phone_body_ratio": round(r_phone, 3),
                "method": "hand_to_ear_posture"
            }, roi_box

        elif is_left_call:
            conf = min(0.96, 0.75 + 0.6 * l_skin)
            roi_box = (
                px1,
                py1 + y_top,
                px1 + int(0.40 * pw),
                py1 + y_bottom
            )
            return True, round(conf, 3), {
                "side": "left_ear",
                "hand_skin_ratio": round(l_skin, 3),
                "phone_body_ratio": round(l_phone, 3),
                "method": "hand_to_ear_posture"
            }, roi_box

        return False, 0.0, {
            "r_skin": round(r_skin, 3),
            "l_skin": round(l_skin, 3),
            "is_on_call": False
        }, None
