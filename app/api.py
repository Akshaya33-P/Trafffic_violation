import os
import shutil
import time
from typing import List, Optional
from datetime import datetime
from fastapi import FastAPI, UploadFile, File, Form, Depends, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session

from src.database.session import get_db, init_db
from src.database.models import ViolationRecord, SceneConfigRecord
from src.utils.validator import validate_media_file

UPLOAD_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Initialize database schema
init_db()

app = FastAPI(
    title="AI-Based Traffic Violation Detection and Classification System",
    description="Decision-Support System for Traffic Violations (All detections are possible violations requiring human review).",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ReviewUpdateRequest(BaseModel):
    review_status: str  # CONFIRMED, REJECTED, EDITED
    reviewer_name: str
    review_notes: Optional[str] = None


class SceneConfigCreateRequest(BaseModel):
    name: str
    description: Optional[str] = ""
    config: dict


@app.get("/")
def root():
    return {
        "system": "AI-Based Traffic Violation Detection & Classification System",
        "status": "operational",
        "role": "DECISION-SUPPORT TOOL",
        "disclaimer": "Every detection is a possible violation requiring human verification. Never issues automatic fines."
    }


@app.post("/api/upload")
async def upload_media(file: UploadFile = File(...)):
    """
    Uploads and validates image or video.
    Rejects corrupted or oversized files with clear messages.
    """
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    is_valid, err_msg, meta = validate_media_file(file_path)
    if not is_valid:
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=400, detail=f"File validation failed: {err_msg}")

    return {
        "status": "success",
        "filename": file.filename,
        "saved_path": file_path,
        "metadata": meta
    }


@app.get("/api/violations")
def get_violations(
    status: Optional[str] = None,
    violation_type: Optional[str] = None,
    tier: Optional[str] = None,
    limit: int = 100,
    offset: int = 0,
    db: Session = Depends(get_db)
):
    query = db.query(ViolationRecord)
    if status:
        query = query.filter(ViolationRecord.review_status == status.upper())
    if violation_type:
        query = query.filter(ViolationRecord.violation_type == violation_type)
    if tier:
        query = query.filter(ViolationRecord.tier == tier)

    total_count = query.count()
    records = query.order_by(ViolationRecord.id.desc()).offset(offset).limit(limit).all()

    return {
        "total": total_count,
        "limit": limit,
        "offset": offset,
        "records": [r.to_dict() for r in records]
    }


@app.patch("/api/violations/{violation_id}/review")
def review_violation(
    violation_id: int,
    payload: ReviewUpdateRequest,
    db: Session = Depends(get_db)
):
    valid_statuses = {"PENDING", "CONFIRMED", "REJECTED", "EDITED"}
    new_status = payload.review_status.upper()
    if new_status not in valid_statuses:
        raise HTTPException(status_code=400, detail=f"Invalid review status. Must be one of {valid_statuses}")

    record = db.query(ViolationRecord).filter(ViolationRecord.id == violation_id).first()
    if not record:
        raise HTTPException(status_code=404, detail="Violation record not found")

    record.review_status = new_status
    record.reviewer_name = payload.reviewer_name
    record.review_notes = payload.review_notes
    record.reviewed_at = datetime.utcnow()
    db.commit()

    return {
        "status": "success",
        "updated_record": record.to_dict()
    }


@app.get("/api/stats")
def get_stats(db: Session = Depends(get_db)):
    """
    Returns aggregated dashboard statistics for violations and human review queue.
    """
    all_violations = db.query(ViolationRecord).all()
    total = len(all_violations)

    by_type = {}
    by_status = {"PENDING": 0, "CONFIRMED": 0, "REJECTED": 0, "EDITED": 0}
    by_tier = {"Tier A": 0, "Tier B": 0}
    confidences = []

    for v in all_violations:
        by_type[v.violation_type] = by_type.get(v.violation_type, 0) + 1
        st = v.review_status.upper() if v.review_status else "PENDING"
        by_status[st] = by_status.get(st, 0) + 1
        tier_key = v.tier if v.tier else "Tier A"
        by_tier[tier_key] = by_tier.get(tier_key, 0) + 1
        confidences.append(v.confidence)

    avg_conf = round(float(sum(confidences) / len(confidences)), 3) if confidences else 0.0

    return {
        "total_detections": total,
        "by_violation_type": by_type,
        "by_review_status": by_status,
        "by_tier": by_tier,
        "average_confidence": avg_conf,
        "confirmed_count": by_status["CONFIRMED"]
    }
