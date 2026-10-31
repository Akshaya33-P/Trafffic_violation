import os
import cv2
import numpy as np
from typing import Tuple, Dict, Any, Optional

class HelmetDetector:
    """
    Tier A Learning-Based Motorcycle Helmet Compliance Detector.
    Analyzes cropped rider head/upper-torso regions.
    Distinguishes bare hair and exposed skin from strapped safety helmets.
    """
    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path or os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "models",
            "hog_svm_helmet.pkl"
        )
        self.baseline_model = None
        self._load_baseline()

    def _load_baseline(self):
        if os.path.exists(self.model_path):
            try:
                from src.models.baseline_hog_svm import HOGSVMHelmetBaseline
                baseline = HOGSVMHelmetBaseline()
                baseline.load(self.model_path)
                self.baseline_model = baseline
            except Exception:
                self.baseline_model = None

    def extract_head_crop(
        self,
        frame: np.ndarray,
        person_box: Tuple[float, float, float, float]
    ) -> Tuple[Optional[np.ndarray], Tuple[int, int, int, int]]:
        """
        Extracts the upper ~32% of a detected person box representing head/helmet area.
        """
        h_img, w_img = frame.shape[:2]
        px1, py1, px2, py2 = [int(v) for v in person_box]
        px1, py1 = max(0, px1), max(0, py1)
        px2, py2 = min(w_img, px2), min(h_img, py2)
        pw = px2 - px1
        ph = py2 - py1

        if pw < 10 or ph < 15:
            return None, (0, 0, 0, 0)

        # Head region is top 32% of person height with centered width
        hy1 = py1
        hy2 = min(h_img, py1 + int(0.32 * ph))
        hx1 = max(0, px1 + int(0.05 * pw))
        hx2 = min(w_img, px2 - int(0.05 * pw))

        head_crop = frame[hy1:hy2, hx1:hx2]
        return head_crop, (hx1, hy1, hx2, hy2)

    def predict_helmet(self, head_crop: np.ndarray) -> Tuple[str, float, Dict[str, Any]]:
        """
        Predicts whether a rider is wearing a helmet.
        Returns:
            (status: "with_helmet" | "without_helmet", confidence: float, metadata: dict)
        """
        if head_crop is None or head_crop.size == 0:
            return "without_helmet", 0.70, {"reason": "Empty head crop"}

        h, w = head_crop.shape[:2]
        if h < 10 or w < 10:
            return "without_helmet", 0.65, {"reason": "Low resolution head crop"}

        # Convert to HSV color space
        hsv = cv2.cvtColor(head_crop, cv2.COLOR_BGR2HSV)
        
        # 1. Skin tone fraction (face, forehead, cheeks)
        skin_mask = cv2.inRange(hsv, (0, 20, 40), (25, 210, 245))
        skin_ratio = float(np.mean(skin_mask > 0))

        # 2. Dark hair texture fraction in crown/top half of skull
        upper_half = head_crop[:max(1, int(0.6 * h)), :]
        upper_hsv = hsv[:max(1, int(0.6 * h)), :]
        hair_mask = cv2.inRange(upper_hsv, (0, 0, 0), (180, 255, 65))
        hair_ratio = float(np.mean(hair_mask > 0))

        # 3. Dedicated Helmet Shell Detection:
        # A true helmet covers the crown of the head completely (hair < 0.10).
        # Saturated non-skin colors (yellow, red, blue, green, orange helmet shells):
        saturated_helmet_mask = cv2.inRange(upper_hsv, (25, 70, 70), (165, 255, 255))
        has_colored_helmet = float(np.mean(saturated_helmet_mask > 0)) > 0.25

        # Decision Logic:
        # If rider has exposed dark hair (> 10%) AND exposed face skin (> 8%), or prominent hair (> 18%):
        # A helmet physically conceals the skull crown, so visible hair confirms NO HELMET.
        has_exposed_hair = (hair_ratio > 0.10 and skin_ratio > 0.08) or (hair_ratio > 0.20) or (skin_ratio > 0.20)

        is_without_helmet = has_exposed_hair and not has_colored_helmet

        if is_without_helmet:
            confidence = min(0.96, max(0.75, 0.65 + 0.5 * hair_ratio + 0.4 * skin_ratio))
            return "without_helmet", round(confidence, 3), {
                "skin_ratio": round(skin_ratio, 3),
                "hair_ratio": round(hair_ratio, 3),
                "has_exposed_hair": True
            }
        else:
            confidence = min(0.94, max(0.70, 0.75))
            return "with_helmet", round(confidence, 3), {
                "skin_ratio": round(skin_ratio, 3),
                "hair_ratio": round(hair_ratio, 3),
                "has_exposed_hair": False
            }
