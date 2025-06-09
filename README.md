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
pip install opencv-python numpy

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




## Weights file- Download
You can download the `.weights` file directly from the [Releases](https://github.com/isalkic1/hand_gesture_recognition/releases):

- [cross-hands.weights](https://github.com/isalkic1/hand_gesture_recognition/releases/download/v1.0/cross-hands.weights)

Place it in the `models/` folder before running the code.
