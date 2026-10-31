# AI-Based Traffic Violation Detection and Classification System Using Deep Learning

> **DECISION-SUPPORT SYSTEM DISCLAIMER**:  
> This system is an assistive **decision-support tool** for human traffic officers and reviewers. Every flagged incident is recorded strictly as a **"possible violation"** that requires human verification. The system **never** issues automatic fines or asserts legal certainty.

---

## 1. Project Overview & Feasibility Tiers

This project is a modular, high-performance computer vision and deep learning prototype that detects vehicles, tracks them across frames, and flags potential traffic violations with visual evidence and human-in-the-loop verification.

### **Tier A: Learning-Based Violations (Data-Driven)**
1. **Motorcycle Rider Without Helmet**:
   - Compares a traditional ML baseline (**HOG + Support Vector Machine**) against a fine-tuned deep learning model on identical, scene-split test data (zero frame-leakage).
2. **Triple Riding on Motorcycle**:
   - Tracks motorcycle bounding boxes and associates mounted rider detections.
   - Employs temporal voting over consecutive frames to reject single-frame false positives.
3. **Driver Without Seatbelt (EXPERIMENTAL)**:
   - Evaluates driver cabin ROI gradients under strict conditions (clear, un-tinted frontal windshield and daylight). Borderline or obscured views are flagged honestly as `indeterminate`.

### **Tier B: Geometry & Tracking Violations (Requires Scene Calibration)**
4. **Wrong-Side Driving**:
   - Direction vector divergence calculation ($\vec{v} \cdot \vec{u}$) against configured lane travel direction.
5. **Red-Light Running**:
   - Detects trajectory intersection across a user-defined stop line during an active `RED` signal state.
6. **Illegal Parking**:
   - Point-in-polygon dwell tracking within a defined no-parking polygon exceeding time limit $T_{min}$ (default: 5.0 s).

---

## 2. Licensing Audit & Permissive Alternatives
- **Ultralytics YOLOv8**: Licensed under **AGPL-3.0** (Open-source copyleft). If used in closed-source proprietary commercial hosting, AGPL terms apply.
- **Permissive Alternative**: The architecture includes a native OpenCV / Torchvision modular backend allowing seamless swapping to Apache-2.0 or BSD-3 models.
- **Datasets**: Roboflow Motorcycle Helmet and Public Datasets are shared under **CC BY 4.0**.

---

## 3. Quickstart & Installation Guide (Beginner-Friendly)

### Step 1: Open Terminal & Activate the Virtual Environment
```bash
cd /home/thota-neelima/Desktop/Traffic_project
source .venv/bin/activate
```

### Step 2: Run All Automated Unit & Integration Tests
```bash
pytest tests/ -v
```
*Expected output: 20 passed tests covering all rule edge cases and video decoding.*

### Step 3: Run the Reproducible Evaluation Benchmark
```bash
python evaluation/run_evaluation.py
```
*Measures and records genuine accuracy, Macro-F1, confusion matrices, and CPU latency (ms/frame and FPS) on your machine.*

### Step 4: Launch the Streamlit Interactive Dashboard
```bash
streamlit run app/ui.py --server.port 8501
```
Open your browser and navigate to: **`http://localhost:8501`**

### Step 5: (Optional) Run the FastAPI REST Backend
```bash
uvicorn app.api:app --reload --port 8000
```
API Documentation (Swagger UI) is available at: **`http://localhost:8000/docs`**

---

## 4. How to Train in Google Colab (Free GPU)

We have provided a ready-to-run Jupyter notebook:
👉 **[notebooks/train_colab.ipynb](notebooks/train_colab.ipynb)**

1. Visit [colab.research.google.com](https://colab.research.google.com).
2. Click **Upload** and upload `notebooks/train_colab.ipynb`.
3. Set runtime to **T4 GPU** (`Runtime > Change runtime type > T4 GPU`).
4. Click **Runtime > Run all**.
5. Colab will fine-tune the model, evaluate on the unseen test set, and automatically download `best.pt`.
6. Copy `best.pt` into your local `models/` directory.

---

## 5. Dashboard Features & Review Queue
1. **🎬 Video Processing**: Upload video footage, visualize tracking IDs, road lines, and real-time alerts.
2. **📋 Human Review Queue**: Filter flagged violations by status (`PENDING`, `CONFIRMED`, `REJECTED`, `EDITED`), inspect side-by-side vehicle crops and full-frame context, and record officer audit notes.
3. **📐 Scene Configuration**: Draw and configure directional vectors, stop lines, no-parking zones, and traffic signal states per camera location.
4. **📊 Analytics Dashboard**: Live metrics, violation distributions, review status breakdowns, and one-click export to **CSV** and **PDF** reports.
5. **📈 Model Evaluation & Latency**: View verified CPU latency and metric cards.

---

## 6. Privacy & Safety Policy
- **License Plate & Facial Identification**: Strictly out of scope. Automatic privacy blurring is provided to redact license plates and human faces in evidence images.
- **Data Retention**: All evidence is saved locally with configurable retention lifetimes.
