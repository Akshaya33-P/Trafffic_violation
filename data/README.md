# Dataset Sources, Licenses, and Documentation

This document records the exact sources, licenses, classes, formats, and acquisition methods for all datasets evaluated and used in the AI-Based Traffic Violation Detection and Classification System.

---

## 1. Pretrained Vehicle Detection & Tracking
- **Dataset Name**: MS COCO 2017 (Common Objects in Context)
- **Source URL**: https://cocodataset.org/
- **License**: Creative Commons Attribution 4.0 International (CC BY 4.0)
- **Classes Used**: `person` (0), `bicycle` (1), `car` (2), `motorcycle` (3), `bus` (5), `truck` (7)
- **Annotation Format**: COCO JSON (bounding boxes [x, y, width, height])
- **Download Method**: Ultralytics YOLOv8 automated download for `yolov8n.pt` (~6.2 MB)
- **Model License Notice**: Ultralytics is licensed under AGPL-3.0. Permissive ONNX alternative via Torchvision Faster-RCNN/SSD (BSD-3-Clause) is supported for commercial environments requiring non-copyleft dependencies.

---

## 2. Helmet vs. No-Helmet Dataset (Tier A)
- **Dataset Name**: Roboflow Universe Motorcycle Helmet Detection
- **Source URL**: https://universe.roboflow.com/
- **License**: Creative Commons Attribution 4.0 International (CC BY 4.0)
- **Size**: ~25 MB - 45 MB (curated balanced subset)
- **Classes**: `with_helmet`, `without_helmet` (or `helmet`, `head`)
- **Annotation Format**: YOLO format (`<class_id> <x_center> <y_center> <width> <height>` normalized to [0, 1]) and cropped ROI image folders for traditional ML baseline
- **Data Splitting Strategy (Zero Leakage)**: Grouped split strictly by **video / scene ID** (never random frame shuffle). 70% Train, 15% Validation, 15% Test. Random seed: `42`.

---

## 3. Triple Riding Dataset (Tier A)
- **Dataset Name**: Multi-Rider Motorcycle Dataset / Roboflow Motorcycle Passengers
- **Source URL**: https://universe.roboflow.com/
- **License**: Creative Commons Attribution 4.0 International (CC BY 4.0)
- **Annotation Method**: Spatial-temporal tracking association (bounding boxes for `motorcycle` combined with overlapping `person` upper-body count $\ge 3$) verified against annotated motorcycle passenger clips.
- **Classes**: `single_rider`, `double_riding`, `triple_riding` ($\ge 3$)

---

## 4. Driver Seatbelt Dataset (Tier A - Experimental)
- **Dataset Name**: Driver Seatbelt Detection Dataset
- **Source URL**: https://universe.roboflow.com/
- **License**: Creative Commons Attribution 4.0 International (CC BY 4.0)
- **Size**: ~20 MB (cropped windshield/driver seat samples)
- **Classes**: `seatbelt_on`, `seatbelt_off`
- **Known Limitations & Operating Domain**: Strictly operates when:
  1. Front windshield is transparent and unobstructed (no heavy tint, rain glare, or reflection).
  2. Camera is frontal or front-quarter angled relative to the vehicle cabin.
  3. Daylight conditions. Night or headlight-glare frames are flagged as "indeterminate".

---

## 5. Tier B Illustrative Test Sets (Scene-Configured Violations)
- **Source**: Dedicated, explicitly labeled synthetic trajectories and small illustrative test video clips.
- **Classification**: *Illustrative Test Data* (clearly marked as synthetic/illustrative and never presented as an exhaustive benchmark).
- **Violations Covered**:
  1. `wrong_side_driving` (moving against allowed directional vector)
  2. `red_light_violation` (crossing defined stop line while signal phase is RED)
  3. `illegal_parking` (stationary inside no-parking polygon exceeding time limit $T_{min}$)
- **Evaluation**: Manually annotated ground truth event logs (True Positives, False Positives, False Negatives).
