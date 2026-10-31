import os
import json
from typing import Dict, Any, List, Optional, Tuple

class SceneConfig:
    """
    Encapsulates scene-level geometric annotations:
    - Allowed travel direction vector / lane zones
    - Stop line coordinates & traffic light state / ROI
    - No-parking polygon zone & duration threshold
    """
    def __init__(
        self,
        name: str = "default_scene",
        allowed_direction_vector: Optional[List[float]] = None, # [dx, dy]
        allowed_direction_tolerance_deg: float = 90.0,
        stop_line: Optional[List[List[float]]] = None, # [[x1, y1], [x2, y2]]
        traffic_light_roi: Optional[List[float]] = None, # [x1, y1, x2, y2]
        traffic_light_state: str = "RED", # "RED", "GREEN", "AUTO"
        no_parking_zone: Optional[List[List[float]]] = None, # [[x1, y1], ...]
        min_parking_duration_sec: float = 5.0
    ):
        self.name = name
        self.allowed_direction_vector = allowed_direction_vector or [0.0, 1.0] # default moving downward
        self.allowed_direction_tolerance_deg = allowed_direction_tolerance_deg
        self.stop_line = stop_line or [[100.0, 400.0], [700.0, 400.0]]
        self.traffic_light_roi = traffic_light_roi
        self.traffic_light_state = traffic_light_state.upper()
        self.no_parking_zone = no_parking_zone
        self.min_parking_duration_sec = min_parking_duration_sec

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "allowed_direction_vector": self.allowed_direction_vector,
            "allowed_direction_tolerance_deg": self.allowed_direction_tolerance_deg,
            "stop_line": self.stop_line,
            "traffic_light_roi": self.traffic_light_roi,
            "traffic_light_state": self.traffic_light_state,
            "no_parking_zone": self.no_parking_zone,
            "min_parking_duration_sec": self.min_parking_duration_sec
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SceneConfig":
        return cls(
            name=data.get("name", "custom_scene"),
            allowed_direction_vector=data.get("allowed_direction_vector", [0.0, 1.0]),
            allowed_direction_tolerance_deg=data.get("allowed_direction_tolerance_deg", 90.0),
            stop_line=data.get("stop_line"),
            traffic_light_roi=data.get("traffic_light_roi"),
            traffic_light_state=data.get("traffic_light_state", "RED"),
            no_parking_zone=data.get("no_parking_zone"),
            min_parking_duration_sec=data.get("min_parking_duration_sec", 5.0)
        )

    def save_to_file(self, file_path: str):
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        with open(file_path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load_from_file(cls, file_path: str) -> "SceneConfig":
        with open(file_path, "r") as f:
            data = json.load(f)
        return cls.from_dict(data)
