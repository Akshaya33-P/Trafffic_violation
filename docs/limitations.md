# System Limitations and Operational Boundaries

This document honestly outlines the real-world operational constraints and known failure modes of the AI-Based Traffic Violation Detection and Classification System.

---

## 1. Environmental & Lighting Constraints
- **Low Light & Night Conditions**: Headlight glare at night causes severe specular saturation, degrading vehicle boundary detection and obscuring rider silhouettes.
- **Inclement Weather (Heavy Rain / Fog)**: Rain streaks on camera lenses distort edge gradients, leading to potential false positives in direction estimation and tracker ID switches.

---

## 2. Experimental Status: Driver Seatbelt Detection
- **Feasibility Tier**: Tier A (Experimental).
- **Hard Constraints**:
  1. Operates only on vehicles with transparent, un-tinted front windshields.
  2. Camera pitch must provide a frontal or front-quarter view into the cabin. Overhead traffic pole cameras (looking down at roof level) cannot resolve driver cabin interiors.
  3. Reflections of clouds, trees, and street signs on windshield glass can mimic or mask diagonal seatbelt straps.
- **Fail-Safe Mechanism**: The module yields `indeterminate` for borderline or obscured frames, preventing spurious violation flags.

---

## 3. Tier B Calibration Dependency
- **Scene Calibration Requirement**:
  - Wrong-side driving, red-light violations, and illegal parking cannot function accurately without explicit scene geometry configuration (stop line coordinates, directional vectors, no-parking polygons).
  - Attempting to run Tier B detection on raw, uncalibrated footage will lead to arbitrary or incorrect alerts.

---

## 4. Multi-Object Occlusion in Dense Traffic
- When multiple motorcycles or large buses cluster closely, standard 2D bounding box IoU association can temporarily merge distinct riders, occasionally skewing rider count estimates.
- Temporal smoothing over $\ge 3$ consecutive frames is utilized to minimize single-frame count spikes.
