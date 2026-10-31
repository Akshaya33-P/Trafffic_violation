import os
import pytest
import cv2
from src.detection.tracker import TrafficViolationPipeline
from src.scene.config import SceneConfig
from src.database.session import SessionLocal
from src.database.models import ViolationRecord

def test_pipeline_integration_on_sample_clip():
    clip_path = "data/illustrative_traffic_sample.mp4"
    assert os.path.exists(clip_path), "Sample clip must exist for integration test"

    # Set up scene configuration matching the clip geometry
    scene_cfg = SceneConfig(
        name="test_junction_integration",
        allowed_direction_vector=[0.0, 1.0],  # Downwards is allowed
        allowed_direction_tolerance_deg=60.0,
        stop_line=[[100.0, 400.0], [540.0, 400.0]],
        traffic_light_state="RED"
    )

    pipeline = TrafficViolationPipeline(
        scene_config=scene_cfg,
        conf_threshold=0.30,
        apply_privacy_blur=False
    )

    cap = cv2.VideoCapture(clip_path)
    assert cap.isOpened()

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_idx = 0
    all_violations = []

    # Process first 60 frames
    while cap.isOpened() and frame_idx < 60:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        ann_frame, viols = pipeline.process_frame(frame, frame_idx, fps, source_name="illustrative_sample")
        if viols:
            all_violations.extend(viols)

    cap.release()

    assert frame_idx >= 50, f"Expected at least 50 frames processed, got {frame_idx}"

    # Verify violations were flagged and evidence files created
    assert len(all_violations) > 0, "Pipeline should flag at least one violation on the test clip"
    
    first_viol = all_violations[0]
    assert "violation_type" in first_viol
    assert first_viol["review_status"] == "PENDING"
    
    # Check evidence files exist
    if "evidence_crop_path" in first_viol and first_viol["evidence_crop_path"]:
        crop_full_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), first_viol["evidence_crop_path"])
        assert os.path.exists(crop_full_path), f"Crop evidence image must exist on disk: {crop_full_path}"
