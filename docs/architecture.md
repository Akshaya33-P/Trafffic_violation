# System Architecture & Technical Specifications

## 1. System Overview
The **AI-Based Traffic Violation Detection and Classification System** is an assistive, human-in-the-loop decision-support platform. It processes urban traffic camera video streams or uploaded clips, tracks multi-class road vehicles, checks spatial-temporal violation criteria across two tiers, logs evidence frames with privacy blurring, and presents an interactive human-audit dashboard.

```
                      +-----------------------------+
                      |   Traffic Video / Camera    |
                      +--------------+--------------+
                                     |
                                     v
                      +-----------------------------+
                      | Media Validator & Decoder   |
                      +--------------+--------------+
                                     |
                                     v
                      +-----------------------------+
                      | Multi-Object Tracking Engine|
                      | (YOLOv8n + ByteTrack)       |
                      +--------------+--------------+
                                     |
           +-------------------------+-------------------------+
           |                                                   |
           v                                                   v
+-----------------------+                           +-----------------------+
|  Tier A (Data-Learned)|                           |  Tier B (Scene Rules) |
| - Helmet / No-Helmet  |                           | - Wrong-Side Driving  |
| - Triple Riding Count |                           | - Red-Light Violation |
| - Seatbelt (Experim.) |                           | - Illegal Parking     |
+----------+------------+                           +-----------+-----------+
           |                                                   |
           +-------------------------+-------------------------+
                                     |
                                     v
                      +-----------------------------+
                      | Visual Evidence Generator   |
                      | - Vehicle Crop (15% margin) |
                      | - Full Frame + Legal Banner |
                      | - Privacy Blur (Plates/Face)|
                      +--------------+--------------+
                                     |
                                     v
                      +-----------------------------+
                      |   SQLAlchemy ORM Layer      |
                      |  (SQLite Dev / MySQL Prod)  |
                      +--------------+--------------+
                                     |
                    +----------------+----------------+
                    |                                 |
                    v                                 v
     +----------------------------+     +----------------------------+
     | FastAPI REST Backend Engine|     | Streamlit Review Dashboard |
     | - /api/violations          |     | - Human Audit & Confirm    |
     | - /api/stats               |     | - Scene Geometry Editor    |
     | - /api/report/export       |     | - CSV & PDF Reports Export |
     +----------------------------+     +----------------------------+
```

---

## 2. Core Modules Breakdown

### 2.1 Media Validator (`src/utils/validator.py`)
- Enforces strict file-size bounds ($\le 250\text{ MB}$).
- Verifies format headers (`.mp4`, `.avi`, `.mov`, `.mkv`, `.jpg`, `.png`).
- Decodes the initial frame stream to guarantee the container is neither truncated nor corrupted.

### 2.2 Tracking & Association (`src/detection/tracker.py`)
- Runs multi-class object detection on vehicle classes: `car`, `motorcycle`, `bus`, `truck`, and `person`.
- Associates detections across frames using ByteTrack (`bytetrack.yaml`) or spatial-temporal IoU tracking.
- Maintains historical trajectory points `(bc_x, bc_y, timestamp)` where `(bc_x, bc_y)` is the vehicle bottom-center (road-contact point).

### 2.3 Tier A Learning-Based Violations
1. **Helmet vs. No-Helmet (`src/models/`)**:
   - Compares a traditional ML baseline (**HOG + SVM** on 64x64 crops) against a deep learning detector.
   - Tested on strictly unseen scene splits to prevent data leakage.
2. **Triple Riding (`src/detection/triple_riding.py`)**:
   - Groups detected `person` boxes with `motorcycle` track envelopes.
   - Requires sustained $\ge 3$ count over consecutive frames to reject transient noise.
3. **Seatbelt Compliance (`src/detection/seatbelt.py`)**:
   - Extracts driver cabin ROI.
   - Flags "indeterminate" when specular glare, heavy tint, or oblique camera angles prevent conclusive visibility.

### 2.4 Tier B Geometry & Calibration Rules (`src/rules/`)
1. **Wrong-Side Driving (`src/rules/wrong_side.py`)**:
   - Normalizes allowed lane vector $\vec{u}$ and vehicle velocity vector $\vec{v}$.
   - Tests angular divergence $\theta = \arccos(\frac{\vec{u} \cdot \vec{v}}{|\vec{u}| |\vec{v}|})$. If $\theta \ge 120^\circ$ sustained, triggers violation.
2. **Red-Light Running (`src/rules/red_light.py`)**:
   - Segments stop line $L_1 L_2$ and vehicle motion vector $P_{t-1} P_t$.
   - Triggers when line segments intersect while the signal state is `RED`.
3. **Illegal Parking (`src/rules/illegal_parking.py`)**:
   - Point-in-polygon ray casting inside configured no-parking boundary.
   - Dwell timer accumulates while vehicle displacement remains $\le 20\text{ px}$. Triggers when dwell $\ge T_{min}$ (default 5.0 s).

### 2.5 Privacy & Audit Logging (`src/utils/privacy.py`, `src/utils/evidence.py`)
- Blurs lower-third vehicle plate regions and human faces when privacy blur is active.
- Writes full contextual evidence frame watermarked with: `"DECISION-SUPPORT ONLY - PENDING HUMAN REVIEW"`.
- Persists record to SQLite / MySQL with initial state `PENDING`.
