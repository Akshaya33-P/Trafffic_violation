from typing import List, Tuple, Dict, Any, Optional
from collections import defaultdict, deque

class TripleRidingDetector:
    """
    Detects triple riding (3 or more persons on a single motorcycle).
    Associates detected persons with motorcycle bounding boxes and applies
    temporal voting over consecutive frames to avoid single-frame false positives.
    """
    def __init__(
        self,
        rider_count_threshold: int = 3,
        min_consecutive_frames: int = 3,
        horizontal_slack_ratio: float = 0.15,
        vertical_overlap_min: float = 0.25
    ):
        self.rider_count_threshold = rider_count_threshold
        self.min_consecutive_frames = min_consecutive_frames
        self.horizontal_slack_ratio = horizontal_slack_ratio
        self.vertical_overlap_min = vertical_overlap_min
        
        # Track history for motorcycle rider counts: track_id -> deque of rider counts
        self._rider_history: Dict[int, deque] = defaultdict(lambda: deque(maxlen=10))
        self._triggered_tracks = set()

    def _is_rider_on_motorcycle(
        self,
        person_box: Tuple[float, float, float, float],
        moto_box: Tuple[float, float, float, float]
    ) -> bool:
        """
        Determines whether a detected person is physically mounted on the motorcycle.
        """
        px1, py1, px2, py2 = person_box
        mx1, my1, mx2, my2 = moto_box

        # Person center coordinates
        p_cx = (px1 + px2) / 2.0
        p_cy = (py1 + py2) / 2.0

        moto_w = mx2 - mx1
        moto_h = my2 - my1

        # Check horizontal bounds with slack
        slack = self.horizontal_slack_ratio * moto_w
        if not (mx1 - slack <= p_cx <= mx2 + slack):
            return False

        # Person must be located in the upper or middle body region of the motorcycle
        # Motorcycle bottom is wheels; rider torso/head sits near top and above
        # Check vertical span intersection
        inter_y1 = max(py1, my1 - 0.5 * moto_h)
        inter_y2 = min(py2, my2)
        inter_h = max(0.0, inter_y2 - inter_y1)
        person_h = max(1.0, py2 - py1)

        return (inter_h / person_h) >= self.vertical_overlap_min

    def check_motorcycle(
        self,
        moto_track_id: int,
        moto_box: Tuple[float, float, float, float],
        all_person_boxes: List[Tuple[float, float, float, float]],
        video_timestamp: float,
        is_still_image: bool = False
    ) -> Tuple[bool, float, Dict[str, Any]]:
        """
        Evaluates rider count on a specific motorcycle track.
        Returns:
            (is_violation, confidence, metadata)
        """
        associated_riders = []
        for p_box in all_person_boxes:
            if self._is_rider_on_motorcycle(p_box, moto_box):
                associated_riders.append(p_box)

        count = len(associated_riders)
        self._rider_history[moto_track_id].append(count)

        # For still images, evaluate directly without requiring multi-frame video history
        if is_still_image:
            if count >= self.rider_count_threshold:
                self._triggered_tracks.add(moto_track_id)
                confidence = min(0.97, 0.85 + 0.05 * (count - 3))
                return True, round(confidence, 3), {
                    "rider_count": count,
                    "consecutive_frames": 1,
                    "motorcycle_box": list(moto_box),
                    "rider_boxes": [list(b) for b in associated_riders],
                    "timestamp_sec": round(video_timestamp, 2)
                }
            return False, 0.0, {
                "rider_count": count,
                "motorcycle_box": list(moto_box),
                "rider_boxes": [list(b) for b in associated_riders],
                "is_violating": False
            }

        history = list(self._rider_history[moto_track_id])
        if len(history) < self.min_consecutive_frames:
            return False, 0.0, {
                "current_count": count,
                "history": history,
                "reason": "Accumulating temporal observations"
            }

        # Check sustained count over the recent consecutive window
        recent_counts = history[-self.min_consecutive_frames:]
        sustained_violation = all(c >= self.rider_count_threshold for c in recent_counts)

        if sustained_violation and moto_track_id not in self._triggered_tracks:
            self._triggered_tracks.add(moto_track_id)
            # Confidence scales with consistency and count
            max_c = max(recent_counts)
            confidence = min(0.97, 0.80 + 0.06 * (max_c - 3))
            return True, round(confidence, 3), {
                "rider_count": max_c,
                "consecutive_frames": len(recent_counts),
                "motorcycle_box": list(moto_box),
                "rider_boxes": [list(b) for b in associated_riders],
                "timestamp_sec": round(video_timestamp, 2)
            }

        return False, 0.0, {
            "current_count": count,
            "recent_counts": recent_counts,
            "rider_boxes": [list(b) for b in associated_riders],
            "is_violating": False
        }
