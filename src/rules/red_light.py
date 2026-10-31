from typing import List, Tuple, Dict, Any, Optional
import numpy as np

def ccw(A: Tuple[float, float], B: Tuple[float, float], C: Tuple[float, float]) -> bool:
    """Tests if 3 points are in counter-clockwise order."""
    return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

def segments_intersect(
    A: Tuple[float, float],
    B: Tuple[float, float],
    C: Tuple[float, float],
    D: Tuple[float, float]
) -> bool:
    """Returns True if line segment AB and line segment CD intersect."""
    return (ccw(A, C, D) != ccw(B, C, D)) and (ccw(A, B, C) != ccw(A, B, D))


class RedLightDetector:
    """
    Detects when a vehicle crosses a configured stop line while the signal state is RED.
    """
    def __init__(
        self,
        stop_line: Tuple[Tuple[float, float], Tuple[float, float]] = ((100.0, 400.0), (700.0, 400.0)),
        signal_state: str = "RED"
    ):
        self.stop_line = stop_line
        self.signal_state = signal_state.upper()
        # Track which track IDs have already triggered red-light violation to prevent duplicate alerts
        self._triggered_tracks = set()

    def set_signal_state(self, state: str):
        self.signal_state = state.upper()

    def check_violation(
        self,
        track_id: int,
        p_prev: Tuple[float, float],
        p_curr: Tuple[float, float],
        timestamp_sec: float,
        signal_state: Optional[str] = None
    ) -> Tuple[bool, float, Dict[str, Any]]:
        """
        Args:
            track_id: Integer track ID
            p_prev: (x, y) coordinates of vehicle bottom-center in previous frame
            p_curr: (x, y) coordinates of vehicle bottom-center in current frame
            timestamp_sec: Current video timestamp
            signal_state: Optional override for current frame's signal state
        """
        curr_state = (signal_state or self.signal_state).upper()

        if track_id in self._triggered_tracks:
            return False, 0.0, {"reason": "Already flagged for this vehicle track"}

        # Only evaluate if traffic signal is RED
        if curr_state != "RED":
            return False, 0.0, {"reason": f"Signal state is {curr_state}, not RED"}

        # Check segment intersection
        sl_p1, sl_p2 = self.stop_line
        crossed = segments_intersect(p_prev, p_curr, sl_p1, sl_p2)

        if crossed:
            self._triggered_tracks.add(track_id)
            confidence = 0.94
            return True, confidence, {
                "signal_state": curr_state,
                "stop_line": list(self.stop_line),
                "crossing_trajectory": [list(p_prev), list(p_curr)],
                "timestamp_sec": round(timestamp_sec, 2)
            }

        return False, 0.0, {"reason": "Did not cross stop line"}
