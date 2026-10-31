import os
import time
import cv2
import numpy as np
from typing import Dict, List, Tuple, Any, Optional

from src.scene.config import SceneConfig
from src.rules.wrong_side import WrongSideDetector
from src.rules.red_light import RedLightDetector
from src.rules.illegal_parking import IllegalParkingDetector
from src.detection.triple_riding import TripleRidingDetector
from src.detection.seatbelt import SeatbeltDetector
from src.detection.helmet import HelmetDetector
from src.detection.phone_call import PhoneCallDetector
from src.utils.evidence import save_evidence
from src.database.models import ViolationRecord
from src.database.session import SessionLocal

# Default COCO vehicle classes mapping
VEHICLE_CLASS_IDS = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
    67: "cell_phone"
}

class TrafficViolationPipeline:
    """
    End-to-end pipeline integrating:
    - Vehicle Detection & Persistent Tracking
    - Tier A Detectors (Helmet, Triple Riding, Seatbelt)
    - Tier B Scene Rules (Wrong-side, Red-light, Illegal parking)
    - Visual Evidence Generation (Crop + Highlighted Full Frame)
    - Database Persistence (Decision-Support Records)
    """
    def __init__(
        self,
        scene_config: Optional[SceneConfig] = None,
        model_weights: str = "models/yolov8n.pt",
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        apply_privacy_blur: bool = False
    ):
        self.scene_config = scene_config or SceneConfig()
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.apply_privacy_blur = apply_privacy_blur
        self.model_weights = model_weights
        self.model = None

        # Initialize Rule & Classifier Engines
        self.wrong_side_detector = WrongSideDetector(
            allowed_vector=tuple(self.scene_config.allowed_direction_vector),
            divergence_angle_threshold_deg=180.0 - self.scene_config.allowed_direction_tolerance_deg
        )
        stop_line_tuple = (
            tuple(self.scene_config.stop_line[0]),
            tuple(self.scene_config.stop_line[1])
        ) if self.scene_config.stop_line else ((100.0, 400.0), (700.0, 400.0))
        self.red_light_detector = RedLightDetector(
            stop_line=stop_line_tuple,
            signal_state=self.scene_config.traffic_light_state
        )
        np_zone = [tuple(p) for p in self.scene_config.no_parking_zone] if self.scene_config.no_parking_zone else []
        self.illegal_parking_detector = IllegalParkingDetector(
            no_parking_zone=np_zone,
            min_duration_sec=self.scene_config.min_parking_duration_sec
        )
        self.triple_riding_detector = TripleRidingDetector()
        self.seatbelt_detector = SeatbeltDetector()
        self.helmet_detector = HelmetDetector()
        self.phone_call_detector = PhoneCallDetector()

        # Track histories: track_id -> list of (x_bottom, y_bottom, timestamp)
        self.track_histories: Dict[int, List[Tuple[float, float, float]]] = {}
        self._fallback_tracks: Dict[int, Tuple[float, float, float, float]] = {}
        self._next_fallback_id = 1

        self._load_detection_model()

    def reset(self):
        """Resets tracking state between distinct videos or still images."""
        self.track_histories.clear()
        self._fallback_tracks.clear()
        self._next_fallback_id = 1
        self.triple_riding_detector._rider_history.clear()
        self.triple_riding_detector._triggered_tracks.clear()
        self.red_light_detector._triggered_tracks.clear()
        self.illegal_parking_detector._vehicle_states.clear()
        # Reset YOLO ByteTrack internal state so track IDs don't carry over
        # between separate images or video clips.
        if self.model is not None:
            try:
                self.model.predictor = None
            except Exception:
                pass

    def _load_detection_model(self):
        """Loads Ultralytics YOLO model if available."""
        try:
            from ultralytics import YOLO
            # Look in models/yolov8n.pt first, then root yolov8n.pt
            weights = self.model_weights
            if not os.path.exists(weights):
                alt_weights = os.path.join(
                    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                    "models",
                    "yolov8n.pt"
                )
                if os.path.exists(alt_weights):
                    weights = alt_weights
            self.model = YOLO(weights)
        except Exception:
            self.model = None

    def process_frame(
        self,
        frame: np.ndarray,
        frame_idx: int,
        fps: float,
        source_name: str = "stream",
        is_still_image: bool = False
    ) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """
        Processes a single frame or still image:
        1. Detects & tracks vehicles and persons.
        2. Evaluates Tier A (Helmet, Triple Riding, Seatbelt) & Tier B (Wrong-side, Red-light, Parking).
        3. Saves evidence and logs to DB on violation.
        4. Renders visual annotations on the output frame.
        """
        timestamp_sec = frame_idx / fps if fps > 0 else frame_idx * 0.04
        annotated_frame = frame.copy()
        flagged_violations = []

        # 1. Run Detection & Tracking
        detections = self._detect_and_track(frame, is_still_image=is_still_image)

        vehicle_tracks = [d for d in detections if d["class_name"] in ["car", "bus", "truck", "motorcycle"]]
        person_boxes = [d["bbox"] for d in detections if d["class_name"] == "person"]
        phone_boxes = [d["bbox"] for d in detections if d["class_name"] == "cell_phone"]

        # 2. Update track history and evaluate rules
        for v in vehicle_tracks:
            tid = v["track_id"]
            x1, y1, x2, y2 = v["bbox"]
            cls_name = v["class_name"]
            conf = v["confidence"]

            # Bottom center point represents road contact position
            bc_x = (x1 + x2) / 2.0
            bc_y = float(y2)

            if tid not in self.track_histories:
                self.track_histories[tid] = []
            self.track_histories[tid].append((bc_x, bc_y, timestamp_sec))

            history = self.track_histories[tid]

            # --- RULE 1: Wrong-Side Driving (Videos only) ---
            if not is_still_image and len(history) >= 4:
                ws_viol, ws_conf, ws_meta = self.wrong_side_detector.check_violation(history)
                if ws_viol:
                    violation_data = self._record_violation(
                        frame=frame,
                        bbox=v["bbox"],
                        violation_type="wrong_side_driving",
                        tier="Tier B",
                        track_id=tid,
                        confidence=ws_conf,
                        timestamp_sec=timestamp_sec,
                        source_file=source_name,
                        object_type=cls_name,
                        meta=ws_meta
                    )
                    flagged_violations.append(violation_data)

            # --- RULE 2: Red-Light Violation (Videos only) ---
            if not is_still_image and len(history) >= 2:
                p_prev = (history[-2][0], history[-2][1])
                p_curr = (history[-1][0], history[-1][1])
                rl_viol, rl_conf, rl_meta = self.red_light_detector.check_violation(
                    track_id=tid,
                    p_prev=p_prev,
                    p_curr=p_curr,
                    timestamp_sec=timestamp_sec
                )
                if rl_viol:
                    violation_data = self._record_violation(
                        frame=frame,
                        bbox=v["bbox"],
                        violation_type="red_light_violation",
                        tier="Tier B",
                        track_id=tid,
                        confidence=rl_conf,
                        timestamp_sec=timestamp_sec,
                        source_file=source_name,
                        object_type=cls_name,
                        meta=rl_meta
                    )
                    flagged_violations.append(violation_data)

            # --- RULE 3: Illegal Parking (Videos only) ---
            if not is_still_image and self.scene_config.no_parking_zone:
                ip_viol, ip_conf, ip_meta = self.illegal_parking_detector.update_and_check(
                    track_id=tid,
                    position=(bc_x, bc_y),
                    timestamp_sec=timestamp_sec
                )
                if ip_viol:
                    violation_data = self._record_violation(
                        frame=frame,
                        bbox=v["bbox"],
                        violation_type="illegal_parking",
                        tier="Tier B",
                        track_id=tid,
                        confidence=ip_conf,
                        timestamp_sec=timestamp_sec,
                        source_file=source_name,
                        object_type=cls_name,
                        meta=ip_meta
                    )
                    flagged_violations.append(violation_data)

                # --- TIER A: Motorcycle Violations (Triple Riding & Helmet Compliance) ---
            if cls_name == "motorcycle" and self._is_valid_motorcycle(v["bbox"], frame.shape[1], vehicle_tracks, conf):
                # 1. Triple Riding Check
                tr_viol, tr_conf, tr_meta = self.triple_riding_detector.check_motorcycle(
                    moto_track_id=tid,
                    moto_box=v["bbox"],
                    all_person_boxes=person_boxes,
                    video_timestamp=timestamp_sec,
                    is_still_image=is_still_image
                )
                if tr_viol:
                    violation_data = self._record_violation(
                        frame=frame,
                        bbox=v["bbox"],
                        violation_type="triple_riding",
                        tier="Tier A",
                        track_id=tid,
                        confidence=tr_conf,
                        timestamp_sec=timestamp_sec,
                        source_file=source_name,
                        object_type="motorcycle",
                        meta=tr_meta
                    )
                    flagged_violations.append(violation_data)
                    # Draw bold notification on vehicle
                    cv2.putText(
                        annotated_frame,
                        f"FLAGGED: TRIPLE RIDING ({tr_meta.get('rider_count', 3)} RIDERS)",
                        (int(x1), max(25, int(y1) - 28)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 0, 255),
                        2
                    )

                # 2. Helmet vs No-Helmet Evaluation on all riders
                associated_riders = tr_meta.get("rider_boxes", [])
                if not associated_riders:
                    for pbox in person_boxes:
                        if self.triple_riding_detector._is_rider_on_motorcycle(pbox, v["bbox"]):
                            associated_riders.append(pbox)

                for r_idx, r_box in enumerate(associated_riders):
                    head_crop, head_coords = self.helmet_detector.extract_head_crop(frame, r_box)
                    if head_crop is not None and head_crop.size > 0:
                        h_status, h_conf, h_meta = self.helmet_detector.predict_helmet(head_crop)
                        hx1, hy1, hx2, hy2 = head_coords
                        if h_status == "without_helmet":
                            # Red Box on Head
                            cv2.rectangle(annotated_frame, (hx1, hy1), (hx2, hy2), (0, 0, 255), 2)
                            cv2.putText(annotated_frame, "NO HELMET", (hx1, max(15, hy1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
                            
                            h_viol_data = self._record_violation(
                                frame=frame,
                                bbox=r_box,
                                violation_type="helmet_not_worn",
                                tier="Tier A",
                                track_id=tid,
                                confidence=h_conf,
                                timestamp_sec=timestamp_sec,
                                source_file=source_name,
                                object_type="rider",
                                meta=h_meta
                            )
                            flagged_violations.append(h_viol_data)
                        else:
                            # Green Box on Head
                            cv2.rectangle(annotated_frame, (hx1, hy1), (hx2, hy2), (0, 255, 0), 2)
                            cv2.putText(annotated_frame, "HELMET", (hx1, max(15, hy1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

                    # 3. Mobile Phone Usage / Calling While Operating Vehicle
                    is_call, call_conf, call_meta, call_roi = self.phone_call_detector.check_phone_call(
                        frame=frame,
                        person_box=r_box,
                        detected_phone_boxes=phone_boxes
                    )
                    if is_call:
                        call_viol_data = self._record_violation(
                            frame=frame,
                            bbox=r_box,
                            violation_type="phone_call_while_driving",
                            tier="Tier A",
                            track_id=tid,
                            confidence=call_conf,
                            timestamp_sec=timestamp_sec,
                            source_file=source_name,
                            object_type="rider",
                            meta=call_meta
                        )
                        flagged_violations.append(call_viol_data)
                        if call_roi is not None:
                            cv2.rectangle(annotated_frame, (call_roi[0], call_roi[1]), (call_roi[2], call_roi[3]), (255, 0, 255), 2)
                            cv2.putText(annotated_frame, "ON CALL", (call_roi[0], max(15, call_roi[1] - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 255), 2)

            # --- TIER A: Seatbelt (Experimental on Cars) ---
            if cls_name == "car" and (x2 - x1) > 80 and (y2 - y1) > 80:
                car_crop = frame[max(0, int(y1)):min(frame.shape[0], int(y2)), max(0, int(x1)):min(frame.shape[1], int(x2))]
                if car_crop.size > 0:
                    sb_status, sb_conf, sb_meta = self.seatbelt_detector.predict_seatbelt(car_crop, vehicle_type="car")
                    if sb_status == "no_seatbelt" and sb_conf >= 0.70:
                        violation_data = self._record_violation(
                            frame=frame,
                            bbox=v["bbox"],
                            violation_type="no_seatbelt",
                            tier="Tier A",
                            track_id=tid,
                            confidence=sb_conf,
                            timestamp_sec=timestamp_sec,
                            source_file=source_name,
                            object_type="car",
                            meta=sb_meta
                        )
                        flagged_violations.append(violation_data)

        # 3. Draw Scene Configurations and Track Annotations
        self._render_scene_and_tracks(annotated_frame, detections, flagged_violations)

        return annotated_frame, flagged_violations

    def _detect_and_track(self, frame: np.ndarray, is_still_image: bool = False) -> List[Dict[str, Any]]:
        """Performs detection and tracking using YOLO + ByteTrack, or fallback."""
        results = []
        if self.model is not None:
            try:
                c_thresh = min(0.20, self.conf_threshold) if is_still_image else self.conf_threshold
                iou_thresh = max(0.60, self.iou_threshold) if is_still_image else self.iou_threshold
                tracks = self.model.track(
                    frame,
                    persist=True,
                    verbose=False,
                    conf=c_thresh,
                    iou=iou_thresh,
                    tracker="bytetrack.yaml"
                )
                for trk in tracks:
                    if trk.boxes is None:
                        continue
                    boxes = trk.boxes.xyxy.cpu().numpy()
                    track_ids = trk.boxes.id.int().cpu().numpy() if trk.boxes.id is not None else list(range(1, len(boxes) + 1))
                    cls_ids = trk.boxes.cls.int().cpu().numpy()
                    confs = trk.boxes.conf.cpu().numpy()

                    for bbox, tid, cid, c in zip(boxes, track_ids, cls_ids, confs):
                        c_name = VEHICLE_CLASS_IDS.get(int(cid), "vehicle")
                        results.append({
                            "track_id": int(tid),
                            "bbox": (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])),
                            "class_id": int(cid),
                            "class_name": c_name,
                            "confidence": float(c)
                        })
                if len(results) > 0:
                    return results
            except Exception:
                pass

        # Fallback: Simple color/motion tracker for test videos or CPU environments
        return self._fallback_detect_and_track(frame)

    def _fallback_detect_and_track(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """Lightweight computer vision fallback detector when YOLO weights are initializing."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        results = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 1200:
                x, y, w, h = cv2.boundingRect(cnt)
                aspect = w / float(h)
                cls_name = "motorcycle" if aspect < 0.6 else "car"
                tid = self._match_or_create_fallback_track((x, y, x + w, y + h))
                results.append({
                    "track_id": tid,
                    "bbox": (float(x), float(y), float(x + w), float(y + h)),
                    "class_id": 3 if cls_name == "motorcycle" else 2,
                    "class_name": cls_name,
                    "confidence": 0.85
                })
        return results

    def _match_or_create_fallback_track(self, bbox: Tuple[float, float, float, float]) -> int:
        bx1, by1, bx2, by2 = bbox
        bcx, bcy = (bx1 + bx2) / 2.0, (by1 + by2) / 2.0

        best_id = None
        min_dist = 60.0

        for tid, prev_box in list(self._fallback_tracks.items()):
            px1, py1, px2, py2 = prev_box
            pcx, pcy = (px1 + px2) / 2.0, (py1 + py2) / 2.0
            dist = np.hypot(bcx - pcx, bcy - pcy)
            if dist < min_dist:
                min_dist = dist
                best_id = tid

        if best_id is not None:
            self._fallback_tracks[best_id] = bbox
            return best_id

        new_id = self._next_fallback_id
        self._next_fallback_id += 1
        self._fallback_tracks[new_id] = bbox
        return new_id

    def _is_valid_motorcycle(
        self,
        bbox: Tuple[float, float, float, float],
        frame_width: int,
        all_vehicle_detections: List[Dict[str, Any]],
        conf: float
    ) -> bool:
        """
        Returns True only if the detection is a genuine two-wheeled motorcycle.
        Rejects auto-rickshaws, three-wheelers, misdetections, and vehicles
        that overlap with cars/buses/trucks.

        Rules:
        1. Confidence must be >= 0.45 (rejects low-quality detections).
        2. Width must be <= 35% of frame width (auto-rickshaws are wider).
        3. Aspect ratio (w/h) must be <= 1.8 (two-wheelers are tall & narrow;
           auto-rickshaws are wide).
        4. Must NOT significantly overlap (> 40% of its own area) with a
           car / bus / truck bounding box (prevents misassociation).
        """
        x1, y1, x2, y2 = bbox
        w = x2 - x1
        h = y2 - y1
        if h <= 0:
            return False

        aspect = w / float(h)

        # Rule 1: minimum confidence
        if conf < 0.45:
            return False

        # Rule 2: not wider than 35% of frame
        if w > 0.35 * frame_width:
            return False

        # Rule 3: aspect ratio — genuine two-wheelers are narrow
        if aspect > 1.8:
            return False

        # Rule 4: must not overlap substantially with car/bus/truck
        moto_area = w * h
        for other in all_vehicle_detections:
            if other["class_name"] not in ("truck", "car", "bus"):
                continue
            ox1, oy1, ox2, oy2 = other["bbox"]
            ix1 = max(x1, ox1)
            iy1 = max(y1, oy1)
            ix2 = min(x2, ox2)
            iy2 = min(y2, oy2)
            inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
            if moto_area > 0 and (inter / moto_area) > 0.40:
                return False

        return True

    def _record_violation(
        self,
        frame: np.ndarray,
        bbox: Tuple[float, float, float, float],
        violation_type: str,
        tier: str,
        track_id: int,
        confidence: float,
        timestamp_sec: float,
        source_file: str,
        object_type: str,
        meta: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Saves visual evidence and records to database."""
        crop_path, full_path = save_evidence(
            frame=frame,
            bbox=bbox,
            violation_type=violation_type,
            track_id=track_id,
            confidence=confidence,
            video_timestamp=timestamp_sec,
            object_type=object_type,
            apply_blur=self.apply_privacy_blur
        )

        try:
            db = SessionLocal()
            record = ViolationRecord(
                violation_type=violation_type,
                tier=tier,
                video_timestamp=round(timestamp_sec, 2),
                track_id=track_id,
                confidence=round(confidence, 3),
                bbox_x1=round(bbox[0], 1),
                bbox_y1=round(bbox[1], 1),
                bbox_x2=round(bbox[2], 1),
                bbox_y2=round(bbox[3], 1),
                evidence_crop_path=crop_path,
                evidence_full_path=full_path,
                source_file=source_file,
                scene_config_name=self.scene_config.name,
                review_status="PENDING"
            )
            db.add(record)
            db.commit()
            db.refresh(record)
            record_dict = record.to_dict()
            db.close()
            return record_dict
        except Exception as e:
            return {
                "violation_type": violation_type,
                "tier": tier,
                "video_timestamp": timestamp_sec,
                "track_id": track_id,
                "confidence": confidence,
                "evidence_crop_path": crop_path,
                "evidence_full_path": full_path,
                "review_status": "PENDING",
                "error": str(e)
            }

    def _render_scene_and_tracks(
        self,
        frame: np.ndarray,
        detections: List[Dict[str, Any]],
        violations: List[Dict[str, Any]]
    ):
        """Draws geometry, overlays, and detection bounding boxes."""
        # Draw Stop Line
        if self.scene_config.stop_line and len(self.scene_config.stop_line) >= 2:
            p1 = tuple([int(v) for v in self.scene_config.stop_line[0]])
            p2 = tuple([int(v) for v in self.scene_config.stop_line[1]])
            line_color = (0, 0, 255) if self.scene_config.traffic_light_state == "RED" else (0, 255, 0)
            cv2.line(frame, p1, p2, line_color, 3)
            cv2.putText(
                frame,
                f"STOP LINE [{self.scene_config.traffic_light_state}]",
                (p1[0] + 5, p1[1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                line_color,
                2
            )

        # Draw No-Parking Zone Polygon
        if self.scene_config.no_parking_zone and len(self.scene_config.no_parking_zone) >= 3:
            pts = np.array(self.scene_config.no_parking_zone, np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame, [pts], isClosed=True, color=(0, 165, 255), thickness=2)
            cv2.putText(
                frame,
                "NO PARKING ZONE",
                (pts[0][0][0] + 5, pts[0][0][1] + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 165, 255),
                2
            )

        # Draw Bounding Boxes
        for d in detections:
            x1, y1, x2, y2 = [int(v) for v in d["bbox"]]
            tid = d["track_id"]
            c_name = d["class_name"]
            color = (255, 140, 0) if c_name == "motorcycle" else (0, 255, 128)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"#{tid} {c_name}"
            cv2.putText(frame, label, (x1, max(15, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
