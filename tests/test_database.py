import pytest
import os
from datetime import datetime
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from src.database.models import Base, ViolationRecord, SceneConfigRecord

@pytest.fixture
def db_session():
    # Use in-memory SQLite database for test isolation
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    TestingSession = sessionmaker(bind=engine)
    session = TestingSession()
    yield session
    session.close()

def test_create_and_retrieve_violation(db_session):
    record = ViolationRecord(
        violation_type="helmet_not_worn",
        tier="Tier A",
        video_timestamp=12.4,
        track_id=42,
        confidence=0.88,
        bbox_x1=100.0,
        bbox_y1=150.0,
        bbox_x2=200.0,
        bbox_y2=300.0,
        evidence_crop_path="evidence/crops/crop_sample.jpg",
        evidence_full_path="evidence/full/full_sample.jpg",
        source_file="test_junction.mp4",
        scene_config_name="main_crossroad",
        review_status="PENDING"
    )
    db_session.add(record)
    db_session.commit()

    retrieved = db_session.query(ViolationRecord).filter_by(track_id=42).first()
    assert retrieved is not None
    assert retrieved.violation_type == "helmet_not_worn"
    assert retrieved.review_status == "PENDING"
    assert retrieved.confidence == 0.88

def test_human_review_status_update(db_session):
    record = ViolationRecord(
        violation_type="red_light_violation",
        tier="Tier B",
        video_timestamp=5.2,
        track_id=99,
        confidence=0.95,
        bbox_x1=50.0,
        bbox_y1=50.0,
        bbox_x2=100.0,
        bbox_y2=100.0,
        source_file="signal_cam1.mp4",
        review_status="PENDING"
    )
    db_session.add(record)
    db_session.commit()

    # Reviewer confirms violation
    record.review_status = "CONFIRMED"
    record.reviewer_name = "Officer_Smith"
    record.review_notes = "Clear violation crossing line during solid red"
    record.reviewed_at = datetime.utcnow()
    db_session.commit()

    updated = db_session.query(ViolationRecord).filter_by(id=record.id).first()
    assert updated.review_status == "CONFIRMED"
    assert updated.reviewer_name == "Officer_Smith"
    assert "solid red" in updated.review_notes

def test_scene_config_serialization(db_session):
    cfg_json = '{"allowed_direction_vector": [0, 1], "stop_line": [[0, 300], [600, 300]]}'
    scene = SceneConfigRecord(
        name="Sector_4_Camera",
        description="Northbound lane intersection",
        config_json=cfg_json
    )
    db_session.add(scene)
    db_session.commit()

    saved = db_session.query(SceneConfigRecord).filter_by(name="Sector_4_Camera").first()
    assert saved is not None
    data = saved.to_dict()
    assert data["name"] == "Sector_4_Camera"
    assert data["config"]["allowed_direction_vector"] == [0, 1]
