# Hand Gesture Recognition using YOLO and OpenCV

This project implements real-time hand gesture recognition using a YOLO-based object detector and OpenCV for gesture classification based on contour analysis. It detects hands, extracts ROIs, preprocesses them, counts fingers, and classifies simple gestures.

---

## Features

- Real-time detection with webcam input
- YOLO-based hand detection (normal, tiny, prn, v4-tiny)
- ROI preprocessing (Gaussian blur + Otsu threshold)
- Finger counting via convexity defects
- Gesture smoothing
- Command-line configuration

---

## Requirements
Install dependencies:
```bash
pip install opencv-python numpy

```

## Structure
```hand_gesture_recognition_project/
├── hand_gesture_recognition.py     # Main detection script
├── yolo.py                         # YOLO wrapper class
├── models/                         # Model configs and weights
│   ├── cross-hands.cfg
│   └── cross-hands.weights         <-- not uploaded
├── download-models.py              # Script to download weights
├── README.md
└── .gitignore
```




## Weights file download
You can download the `.weights` file directly from the [Releases](https://github.com/isalkic1/hand_gesture_recognition/releases):

- [cross-hands.weights](https://github.com/isalkic1/hand_gesture_recognition/releases/download/v1.0/cross-hands.weights)

Place it in the `models/` folder before running the code.


---
---

# MediaPipe-Based ASL Letter Detection

This script implements **real-time recognition of static American Sign Language (ASL)** letters using [MediaPipe Hands](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker) and OpenCV. It classifies letters based on geometric rules derived from hand landmark positions.

## Features

- 🔹 Static ASL letter detection (A–Y, excluding J and Z)
- 🔹 Real-time webcam input with FPS display
- 🔹 Uses finger orientation, angles, and distances for rule-based classification
- 🔹 Smoothing with a gesture buffer to reduce flickering

## How to Run

Install the required dependencies:

```bash
pip install opencv-python mediapipe
```
## Notes
1. Works best with one visible hand
2. Prints Fingers up state in console for debugging
3. Some complex letters (e.g., M, N, T) depend on subtle landmark differences

## Related File
mediapipe_sign_language_detection.py — ASL letter classifier using MediaPipe
