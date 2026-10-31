import math
from typing import List, Tuple, Dict, Any, Optional

class WrongSideDetector:
    """
    Detects vehicles traveling opposite to the configured lane/road direction vector.
    Calculates velocity vector from track history and computes angular divergence.
    """
    def __init__(
        self,
        allowed_vector: Tuple[float, float] = (0.0, 1.0),
        min_displacement_px: float = 18.0,
        divergence_angle_threshold_deg: float = 120.0,
        min_consistent_frames: int = 3
    ):
        self.allowed_vector = self._normalize(allowed_vector)
        self.min_displacement_px = min_displacement_px
        self.divergence_angle_threshold_deg = divergence_angle_threshold_deg
        self.min_consistent_frames = min_consistent_frames

    def _normalize(self, v: Tuple[float, float]) -> Tuple[float, float]:
        norm = math.hypot(v[0], v[1])
        if norm == 0:
            return (0.0, 1.0)
        return (v[0] / norm, v[1] / norm)

    def check_violation(
        self,
        track_history: List[Tuple[float, float, float]]  # List of (x, y, timestamp_sec)
    ) -> Tuple[bool, float, Dict[str, Any]]:
        """
        Args:
            track_history: List of (x, y, timestamp) ordered chronologically.
        Returns:
            (is_violation, confidence, metadata)
        """
        if len(track_history) < self.min_consistent_frames + 1:
            return False, 0.0, {"reason": "Insufficient track history"}

        # Use recent points to estimate current direction vector
        p_start = track_history[- (self.min_consistent_frames + 1)]
        p_end = track_history[-1]

        dx = p_end[0] - p_start[0]
        dy = p_end[1] - p_start[1]
        dt = p_end[2] - p_start[2] if len(p_end) > 2 and len(p_start) > 2 else 1.0

        displacement = math.hypot(dx, dy)
        if displacement < self.min_displacement_px:
            return False, 0.0, {
                "reason": "Stationary or low displacement",
                "displacement": round(displacement, 1)
            }

        vx, vy = dx / displacement, dy / displacement
        ux, uy = self.allowed_vector

        # Dot product = cos(theta)
        dot_product = vx * ux + vy * uy
        dot_product = max(-1.0, min(1.0, dot_product))
        angle_deg = math.degrees(math.acos(dot_product))

        is_violating = angle_deg >= self.divergence_angle_threshold_deg
        
        # Confidence scales from 0.50 at threshold up to 0.98 at 180 degrees
        confidence = 0.0
        if is_violating:
            diff = angle_deg - self.divergence_angle_threshold_deg
            span = 180.0 - self.divergence_angle_threshold_deg
            confidence = min(0.98, max(0.55, 0.55 + 0.43 * (diff / (span if span > 0 else 1.0))))

        return is_violating, round(confidence, 3), {
            "displacement_px": round(displacement, 1),
            "divergence_angle_deg": round(angle_deg, 1),
            "dot_product": round(dot_product, 3),
            "allowed_vector": list(self.allowed_vector),
            "motion_vector": [round(vx, 3), round(vy, 3)]
        }
