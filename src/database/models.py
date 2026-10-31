from datetime import datetime
import json
from sqlalchemy import Column, Integer, Float, String, DateTime, Text, Boolean, create_engine
from sqlalchemy.orm import declarative_base

Base = declarative_base()

class ViolationRecord(Base):
    __tablename__ = "violations"

    id = Column(Integer, primary_key=True, autoincrement=True)
    violation_type = Column(String(64), nullable=False, index=True)
    tier = Column(String(16), nullable=False)  # "Tier A" or "Tier B"
    video_timestamp = Column(Float, nullable=False)  # seconds into the video
    wall_clock_timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)
    track_id = Column(Integer, nullable=False, index=True)
    confidence = Column(Float, nullable=False)
    
    # Bounding Box coordinates [x1, y1, x2, y2]
    bbox_x1 = Column(Float, nullable=False)
    bbox_y1 = Column(Float, nullable=False)
    bbox_x2 = Column(Float, nullable=False)
    bbox_y2 = Column(Float, nullable=False)

    # Saved evidence paths
    evidence_crop_path = Column(String(512), nullable=True)
    evidence_full_path = Column(String(512), nullable=True)
    
    # Source footage and scene config linkage
    source_file = Column(String(256), nullable=False)
    scene_config_name = Column(String(128), nullable=True)
    
    # Human Review Workflow (Decision Support)
    review_status = Column(String(32), default="PENDING", index=True)  # PENDING, CONFIRMED, REJECTED, EDITED
    review_notes = Column(Text, nullable=True)
    reviewed_at = Column(DateTime, nullable=True)
    reviewer_name = Column(String(64), nullable=True)
    
    # Extra metadata (JSON string for extra context e.g. speed, rider count, signal state)
    extra_metadata = Column(Text, nullable=True)

    def to_dict(self):
        return {
            "id": self.id,
            "violation_type": self.violation_type,
            "tier": self.tier,
            "video_timestamp": round(self.video_timestamp, 2),
            "wall_clock_timestamp": self.wall_clock_timestamp.isoformat() if self.wall_clock_timestamp else None,
            "track_id": self.track_id,
            "confidence": round(self.confidence, 3),
            "bbox": [self.bbox_x1, self.bbox_y1, self.bbox_x2, self.bbox_y2],
            "evidence_crop_path": self.evidence_crop_path,
            "evidence_full_path": self.evidence_full_path,
            "source_file": self.source_file,
            "scene_config_name": self.scene_config_name,
            "review_status": self.review_status,
            "review_notes": self.review_notes,
            "reviewed_at": self.reviewed_at.isoformat() if self.reviewed_at else None,
            "reviewer_name": self.reviewer_name,
            "extra_metadata": json.loads(self.extra_metadata) if self.extra_metadata else {}
        }


class SceneConfigRecord(Base):
    __tablename__ = "scene_configs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(128), unique=True, nullable=False)
    description = Column(String(256), nullable=True)
    config_json = Column(Text, nullable=False)  # JSON representation of polygons, lines, directions
    reference_image_path = Column(String(512), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def to_dict(self):
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "config": json.loads(self.config_json) if self.config_json else {},
            "reference_image_path": self.reference_image_path,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }
