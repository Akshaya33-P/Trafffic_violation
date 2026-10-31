import math
from typing import List, Tuple, Dict, Any, Optional

def point_in_polygon(point: Tuple[float, float], polygon: List[Tuple[float, float]]) -> bool:
    """Ray casting algorithm to determine if a point is inside a polygon."""
    x, y = point
    n = len(polygon)
    inside = False
    p1x, p1y = polygon[0]
    for i in range(n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    return inside


class IllegalParkingDetector:
    """
    Detects vehicles remaining stationary inside a configured No-Parking polygon zone
    for longer than the minimum duration threshold (e.g., 5.0 seconds).
    """
    def __init__(
        self,
        no_parking_zone: List[Tuple[float, float]],
        min_duration_sec: float = 5.0,
        max_movement_px: float = 20.0
    ):
        self.no_parking_zone = no_parking_zone
        self.min_duration_sec = min_duration_sec
        self.max_movement_px = max_movement_px
        # State tracking: track_id -> {"entry_time": float, "initial_pos": (x, y), "last_pos": (x, y), "triggered": bool}
        self._vehicle_states: Dict[int, Dict[str, Any]] = {}

    def update_and_check(
        self,
        track_id: int,
        position: Tuple[float, float],  # (x, y) bottom-center or center
        timestamp_sec: float
    ) -> Tuple[bool, float, Dict[str, Any]]:
        """
        Args:
            track_id: Vehicle ID
            position: Current (x, y) coordinates
            timestamp_sec: Current video timestamp
        Returns:
            (is_violation, confidence, metadata)
        """
        if not self.no_parking_zone or len(self.no_parking_zone) < 3:
            return False, 0.0, {"reason": "No valid no-parking zone configured"}

        is_inside = point_in_polygon(position, self.no_parking_zone)

        if not is_inside:
            # Vehicle left the zone: clear state
            if track_id in self._vehicle_states:
                del self._vehicle_states[track_id]
            return False, 0.0, {"inside_zone": False}

        # Vehicle is inside zone
        if track_id not in self._vehicle_states:
            self._vehicle_states[track_id] = {
                "entry_time": timestamp_sec,
                "initial_pos": position,
                "last_pos": position,
                "triggered": False
            }
            return False, 0.0, {"inside_zone": True, "dwell_time": 0.0}

        state = self._vehicle_states[track_id]
        if state["triggered"]:
            # Already flagged for this stationary session
            return False, 0.0, {"reason": "Already flagged for this parking violation"}

        # Measure displacement from entry position
        init_x, init_y = state["initial_pos"]
        curr_x, curr_y = position
        displacement = math.hypot(curr_x - init_x, curr_y - init_y)

        if displacement > self.max_movement_px:
            # Vehicle is moving through the zone (not parked): reset start position
            state["entry_time"] = timestamp_sec
            state["initial_pos"] = position
            return False, 0.0, {
                "inside_zone": True,
                "moving": True,
                "displacement_px": round(displacement, 1)
            }

        dwell_time = timestamp_sec - state["entry_time"]
        if dwell_time >= self.min_duration_sec:
            state["triggered"] = True
            confidence = min(0.96, 0.70 + 0.05 * (dwell_time - self.min_duration_sec))
            return True, round(confidence, 3), {
                "inside_zone": True,
                "dwell_time_sec": round(dwell_time, 2),
                "min_duration_sec": self.min_duration_sec,
                "displacement_px": round(displacement, 1),
                "position": [round(curr_x, 1), round(curr_y, 1)]
            }

        return False, 0.0, {
            "inside_zone": True,
            "dwell_time_sec": round(dwell_time, 2),
            "remaining_sec": round(self.min_duration_sec - dwell_time, 2)
        }
