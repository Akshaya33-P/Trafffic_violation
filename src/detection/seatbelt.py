import os
from typing import Tuple, Dict, Any, Optional
import numpy as np

class SeatbeltDetector:
    """
    Tier A (Experimental) Seatbelt Detector.
    
    Feasibility & Operational Boundary:
    Only operates under strict conditions:
    1. Vehicle type is Car / Taxi / Van.
    2. Frontal or front-quarter camera view (windshield clearly visible).
    3. Adequate daylight without extreme glare, heavy tint, or reflection.
    
    If windshield is obscured, glare is detected, or image resolution is too low,
    it returns 'indeterminate' rather than generating misleading false violations.
    """
    def __init__(self, confidence_threshold: float = 0.65, min_cabin_height_px: int = 40):
        self.confidence_threshold = confidence_threshold
        self.min_cabin_height_px = min_cabin_height_px

    def extract_driver_cabin_roi(
        self,
        vehicle_crop: np.ndarray
    ) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
        """
        Extracts driver cabin / windshield region (typically upper 20%-55% of vehicle crop).
        """
        if vehicle_crop is None or vehicle_crop.size == 0:
            return None, {"error": "Empty crop"}

        vh, vw = vehicle_crop.shape[:2]
        if vh < self.min_cabin_height_px or vw < self.min_cabin_height_px:
            return None, {"reason": "Vehicle resolution too low for windshield inspection"}

        # Windshield typically spans vertical [0.15 * vh to 0.55 * vh]
        # and horizontal [0.10 * vw to 0.90 * vw]
        y1 = int(0.15 * vh)
        y2 = int(0.55 * vh)
        x1 = int(0.10 * vw)
        x2 = int(0.90 * vw)

        cabin_roi = vehicle_crop[y1:y2, x1:x2]
        return cabin_roi, {"cabin_box": [x1, y1, x2, y2]}

    def check_glare_or_tint(self, cabin_roi: np.ndarray) -> bool:
        """
        Checks if the cabin region is dominated by specular reflection/glare or heavy tint.
        """
        import cv2
        gray = cv2.cvtColor(cabin_roi, cv2.COLOR_BGR2GRAY) if len(cabin_roi.shape) == 3 else cabin_roi
        # Fraction of saturated/blown-out white pixels (glare)
        glare_ratio = np.mean(gray > 240)
        # Fraction of pitch black pixels (heavy tint)
        tint_ratio = np.mean(gray < 15)
        return bool(glare_ratio > 0.35 or tint_ratio > 0.70)

    def predict_seatbelt(
        self,
        vehicle_crop: np.ndarray,
        vehicle_type: str = "car"
    ) -> Tuple[str, float, Dict[str, Any]]:
        """
        Evaluates driver seatbelt compliance.
        Returns:
            (status: "seatbelt_on" | "no_seatbelt" | "indeterminate", confidence, metadata)
        """
        if vehicle_type.lower() not in ["car", "taxi", "van", "suv"]:
            return "indeterminate", 0.0, {
                "reason": f"Seatbelt inspection not applicable to vehicle type: {vehicle_type}"
            }

        cabin_roi, meta = self.extract_driver_cabin_roi(vehicle_crop)
        if cabin_roi is None:
            return "indeterminate", 0.0, {
                "reason": meta.get("reason", "Unable to extract driver cabin ROI")
            }

        if self.check_glare_or_tint(cabin_roi):
            return "indeterminate", 0.0, {
                "reason": "Excessive windshield glare or opaque tint prevents reliable analysis"
            }

        # Analyze diagonal edge gradients across driver torso (seatbelt strap characteristic: ~45 deg diagonal band)
        import cv2
        gray = cv2.cvtColor(cabin_roi, cv2.COLOR_BGR2GRAY) if len(cabin_roi.shape) == 3 else cabin_roi
        # Sobel diagonal gradient response
        sobel_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        magnitude = np.hypot(sobel_x, sobel_y)
        angle = np.arctan2(sobel_y, sobel_x) * 180.0 / np.pi

        # Seatbelt angle is typically roughly 30 to 60 degrees diagonal
        diagonal_mask = ((angle >= 30) & (angle <= 65)) | ((angle >= -150) & (angle <= -115))
        strap_energy = float(np.mean(magnitude[diagonal_mask])) if np.any(diagonal_mask) else 0.0
        total_energy = float(np.mean(magnitude)) + 1e-5
        diagonal_ratio = strap_energy / total_energy

        # Threshold decision
        if diagonal_ratio > 1.25:
            conf = min(0.92, 0.65 + 0.20 * (diagonal_ratio - 1.25))
            return "seatbelt_on", round(conf, 3), {
                "experimental": True,
                "diagonal_ratio": round(diagonal_ratio, 2),
                "limits": "Visible windshield verified"
            }
        elif diagonal_ratio < 0.85:
            conf = min(0.88, 0.60 + 0.20 * (0.85 - diagonal_ratio))
            return "no_seatbelt", round(conf, 3), {
                "experimental": True,
                "diagonal_ratio": round(diagonal_ratio, 2),
                "limits": "Low diagonal strap gradient in driver ROI"
            }
        else:
            return "indeterminate", 0.0, {
                "experimental": True,
                "reason": "Borderline gradient energy; human confirmation required",
                "diagonal_ratio": round(diagonal_ratio, 2)
            }
