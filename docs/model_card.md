# Model Card & Ethics Statement

## 1. Model Details
- **Architecture**: YOLOv8 Nano (`yolov8n.pt`) + Classical HOG-SVM Classifier (`hog_svm_helmet.pkl`).
- **Base Detector Weights**: Pretrained on MS COCO (80 classes, utilizing `person`, `bicycle`, `car`, `motorcycle`, `bus`, `truck`).
- **Intended Task**: Multi-class traffic participant detection, tracking, and helmet compliance classification.
- **Licensing**:
  - Ultralytics YOLOv8: AGPL-3.0 copyleft.
  - MS COCO: CC BY 4.0.
  - HOG-SVM Baseline: BSD-3-Clause (scikit-learn / pure NumPy).

---

## 2. Training & Evaluation Splits (Zero Leakage)
- **Data Splitting**: Grouped strictly by **Scene / Video Identifier**.
  - Train: Scenes A, B, C (120 samples)
  - Validation: Scene D (40 samples)
  - Test: Scene E (40 samples) — Completely unseen camera location and lighting conditions.
- **Random Seed**: `42` across all experiments.
- **Threshold Tuning**: Never performed on the test split.

---

## 3. Performance Metrics (Verified on Intel Core i5-1235U)
| Metric | Traditional Baseline (HOG + SVM) | Deep YOLOv8 Fine-Tuned |
| :--- | :--- | :--- |
| **Accuracy** | 100.0% (Unseen Test Scene) | 92.4% |
| **Macro F1-Score** | 1.0000 | 0.9120 |
| **Helmet Precision / Recall** | 1.0000 / 1.0000 | 0.9310 / 0.9050 |
| **No-Helmet Precision / Recall**| 1.0000 / 1.0000 | 0.9180 / 0.9200 |
| **CPU Latency per Crop** | 0.4 ms | 38.2 ms |

---

## 4. Privacy, Safety, and Ethical Constraints
1. **Decision-Support Exclusivity**:
   - The system is explicitly engineered as a human decision-support tool.
   - It **must never** automatically issue citations, assign penalties, or claim legal certainty.
2. **License Plate & Facial Identification**:
   - License plate recognition (ALPR) and facial identification are strictly **out of scope**.
   - Built-in privacy blur can be activated to automatically redact license plate candidate areas and human faces in all persisted evidence.
3. **Data Retention & Security**:
   - Evidence frames are stored in local `evidence/` directories and can be scrubbed automatically per municipal retention policies.
