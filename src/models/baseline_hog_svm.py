import os
import pickle
from typing import List, Tuple, Dict, Any, Optional
import numpy as np

class HOGSVMHelmetBaseline:
    """
    Traditional Machine Learning Baseline for Helmet vs. No-Helmet Classification.
    Uses Histogram of Oriented Gradients (HOG) texture/edge feature extraction
    paired with a Support Vector Machine (Linear/RBF SVM) or Random Forest classifier.
    """
    def __init__(self, target_size: Tuple[int, int] = (64, 64)):
        self.target_size = target_size
        self.model = None
        self.scaler = None
        self.classes = ["helmet", "no_helmet"]

    def extract_hog_features(self, image: np.ndarray) -> np.ndarray:
        """
        Extracts HOG descriptor from a cropped RGB/BGR rider head image.
        Uses pure NumPy Dalal-Triggs HOG implementation for total portability.
        """
        import cv2
        if image is None or image.size == 0:
            return np.zeros(1764, dtype=np.float32)

        # Convert to Grayscale & resize to fixed target resolution (64, 64)
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        resized = cv2.resize(gray, self.target_size, interpolation=cv2.INTER_AREA).astype(np.float32)

        # 1. Compute horizontal and vertical gradients
        gx = np.zeros_like(resized)
        gy = np.zeros_like(resized)
        gx[:, 1:-1] = resized[:, 2:] - resized[:, :-2]
        gy[1:-1, :] = resized[2:, :] - resized[:-2, :]

        # 2. Compute magnitude and orientation in [0, 180) degrees
        mag = np.hypot(gx, gy)
        ang = np.degrees(np.arctan2(gy, gx)) % 180.0

        # 3. Cell histograms: 8x8 cells across 64x64 image -> 8x8 cells
        cell_size = 8
        n_cells_y = 64 // cell_size
        n_cells_x = 64 // cell_size
        nbins = 9
        bin_width = 180.0 / nbins

        hist = np.zeros((n_cells_y, n_cells_x, nbins), dtype=np.float32)

        for cy in range(n_cells_y):
            for cx in range(n_cells_x):
                c_mag = mag[cy*cell_size:(cy+1)*cell_size, cx*cell_size:(cx+1)*cell_size]
                c_ang = ang[cy*cell_size:(cy+1)*cell_size, cx*cell_size:(cx+1)*cell_size]

                bin_idx = (c_ang / bin_width).astype(int) % nbins
                for b in range(nbins):
                    hist[cy, cx, b] = np.sum(c_mag[bin_idx == b])

        # 4. Block normalization (2x2 cells per block with L2-Hys)
        blocks = []
        for by in range(n_cells_y - 1):
            for bx in range(n_cells_x - 1):
                block = hist[by:by+2, bx:bx+2, :].flatten()
                norm = np.linalg.norm(block) + 1e-5
                block_norm = block / norm
                # Clip to 0.2 and re-normalize (L2-Hys)
                block_norm = np.clip(block_norm, 0.0, 0.2)
                block_norm = block_norm / (np.linalg.norm(block_norm) + 1e-5)
                blocks.append(block_norm)

        return np.concatenate(blocks).astype(np.float32)

    def train(
        self,
        X_images: List[np.ndarray],
        y_labels: List[int],
        classifier_type: str = "svm"
    ) -> Dict[str, Any]:
        """
        Extracts HOG features from images and trains the classical classifier.
        y_labels: 0 for 'helmet', 1 for 'no_helmet'.
        """
        from sklearn.svm import SVC
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import StandardScaler

        print(f"Extracting HOG features for {len(X_images)} training samples...")
        features = [self.extract_hog_features(img) for img in X_images]
        X = np.array(features)
        y = np.array(y_labels)

        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        if classifier_type.lower() == "random_forest":
            self.model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")
        else:
            self.model = SVC(kernel="rbf", probability=True, C=1.0, random_state=42, class_weight="balanced")

        self.model.fit(X_scaled, y)
        train_acc = float(self.model.score(X_scaled, y))
        return {
            "num_samples": len(X),
            "train_accuracy": round(train_acc, 4),
            "classifier_type": classifier_type
        }

    def evaluate(
        self,
        X_test_images: List[np.ndarray],
        y_test: List[int]
    ) -> Dict[str, Any]:
        """
        Evaluates classifier on unseen test set using Precision, Recall, F1, and Confusion Matrix.
        """
        from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, accuracy_score

        if self.model is None or self.scaler is None:
            raise ValueError("Baseline model has not been trained or loaded yet.")

        features = [self.extract_hog_features(img) for img in X_test_images]
        X = np.array(features)
        X_scaled = self.scaler.transform(X)

        y_pred = self.model.predict(X_scaled)
        y_prob = self.model.predict_proba(X_scaled)[:, 1] if hasattr(self.model, "predict_proba") else None

        acc = float(accuracy_score(y_test, y_pred))
        precision, recall, f1, support = precision_recall_fscore_support(
            y_test, y_pred, average=None, labels=[0, 1], zero_division=0
        )
        macro_f1 = float(np.mean(f1))
        cm = confusion_matrix(y_test, y_pred, labels=[0, 1]).tolist()

        return {
            "accuracy": round(acc, 4),
            "macro_f1": round(macro_f1, 4),
            "per_class": {
                "helmet": {
                    "precision": round(float(precision[0]), 4),
                    "recall": round(float(recall[0]), 4),
                    "f1_score": round(float(f1[0]), 4),
                    "support": int(support[0])
                },
                "no_helmet": {
                    "precision": round(float(precision[1]), 4),
                    "recall": round(float(recall[1]), 4),
                    "f1_score": round(float(f1[1]), 4),
                    "support": int(support[1])
                }
            },
            "confusion_matrix": cm,
            "classes": ["helmet", "no_helmet"]
        }

    def predict(self, crop_image: np.ndarray) -> Tuple[str, float]:
        """
        Predicts whether a cropped rider head/upper body is wearing a helmet.
        Returns:
            (label, confidence)
        """
        if self.model is None or self.scaler is None:
            return "unknown", 0.0

        feat = self.extract_hog_features(crop_image)
        scaled = self.scaler.transform(feat.reshape(1, -1))
        pred_idx = int(self.model.predict(scaled)[0])
        prob = float(self.model.predict_proba(scaled)[0][pred_idx]) if hasattr(self.model, "predict_proba") else 0.85
        label = self.classes[pred_idx]
        return label, round(prob, 3)

    def save(self, filepath: str):
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        with open(filepath, "wb") as f:
            pickle.dump({"model": self.model, "scaler": self.scaler, "target_size": self.target_size}, f)

    def load(self, filepath: str):
        with open(filepath, "rb") as f:
            data = pickle.load(f)
            self.model = data["model"]
            self.scaler = data["scaler"]
            self.target_size = data["target_size"]
