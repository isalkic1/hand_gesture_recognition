import argparse
import cv2
import numpy as np
import math
from collections import deque
from yolo import YOLO
import sys

# Remove Jupyter-specific arguments
sys.argv = sys.argv[:1]

# Argument Parsing
ap = argparse.ArgumentParser()
ap.add_argument('-n', '--network', default="normal", choices=["normal", "tiny", "prn", "v4-tiny"],
                help='Network Type')
ap.add_argument('-d', '--device', type=int, default=0, help='Device to use')
ap.add_argument('-s', '--size', default=416, help='Size for yolo')
ap.add_argument('-c', '--confidence', default=0.2, help='Confidence for yolo')
ap.add_argument('-nh', '--hands', default=-1, help='Total number of hands to be detected per frame (-1 for all)')
args = ap.parse_args()

# Load YOLO
if args.network == "normal":
    print("loading yolo...")
    yolo = YOLO("models/cross-hands.cfg", "models/cross-hands.weights", ["hand"])
elif args.network == "prn":
    print("loading yolo-tiny-prn...")
    yolo = YOLO("models/cross-hands-tiny-prn.cfg", "models/cross-hands-tiny-prn.weights", ["hand"])
elif args.network == "v4-tiny":
    print("loading yolov4-tiny-prn...")
    yolo = YOLO("models/cross-hands-yolov4-tiny.cfg", "models/cross-hands-yolov4-tiny.weights", ["hand"])
else:
    print("loading yolo-tiny...")
    yolo = YOLO("models/cross-hands-tiny.cfg", "models/cross-hands-tiny.weights", ["hand"])

yolo.size = int(args.size)
yolo.confidence = float(args.confidence)

# Buffer for gesture smoothing
gesture_history = deque(maxlen=5)

# === Utility Functions ===

def preprocess_roi(roi):
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (9, 9), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh

def find_hand_contour(thresh, min_area=5000, aspect_ratio_range=(0.2, 2.0)):
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None

    # Filter contours based on area and aspect ratio
    valid_contours = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue  # Ignore small contours (likely noise)

        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = float(w) / h
        if aspect_ratio_range[0] <= aspect_ratio <= aspect_ratio_range[1]:
            valid_contours.append(cnt)

    if not valid_contours:
        return None

    # Return the most "hand-like" contour: largest of the valid ones
    return max(valid_contours, key=cv2.contourArea)


def calculate_angle(far, start, end):
    a = np.linalg.norm(start - end)
    b = np.linalg.norm(far - start)
    c = np.linalg.norm(far - end)
    if b * c == 0:
        return 180
    angle = math.acos((b**2 + c**2 - a**2) / (2 * b * c))
    return angle * 180 / math.pi

def count_fingers(contour, roi):
    hull = cv2.convexHull(contour, returnPoints=False)
    if len(hull) < 3:
        return 0
    defects = cv2.convexityDefects(contour, hull)
    if defects is None:
        return 0
    finger_count = 0
    for i in range(defects.shape[0]):
        s, e, f, _ = defects[i, 0]
        start = contour[s][0]
        end = contour[e][0]
        far = contour[f][0]
        angle = calculate_angle(np.array(far), np.array(start), np.array(end))
        if angle <= 90:
            finger_count += 1
            cv2.circle(roi, tuple(far), 5, (0, 0, 255), -1)
    return min(finger_count + 1, 5)

def classify_gesture(finger_count):
    if finger_count == 0:
        return "Fist"
    elif finger_count == 5:
        return "Palm"
    elif finger_count == 3:
        return "3 fingers"
    else:
        return f"{finger_count} fingers"

def smooth_prediction(new_gesture):
    gesture_history.append(new_gesture)
    return max(set(gesture_history), key=gesture_history.count)

# === Webcam Input and Processing Loop ===

print("starting webcam...")
vc = cv2.VideoCapture(args.device)

if vc.isOpened():
    rval, frame = vc.read()
else:
    rval = False

while rval:
    frame = cv2.flip(frame, 1)
    width, height, inference_time, results = yolo.inference(frame)

    cv2.putText(frame, f'{round(1/inference_time,2)} FPS', (15,15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 2)
    results.sort(key=lambda x: x[2])

    hand_count = len(results)
    if args.hands != -1:
        hand_count = int(args.hands)

    for detection in results[:hand_count]:
        id, name, confidence, x, y, w, h = detection
        cx = x + (w / 2)
        cy = y + (h / 2)

        color = (0, 255, 255)
        #cv2.rectangle(frame, (x, y - 70), (x + w + 10, y + h), color, 2)
        text = f"{name} ({round(confidence, 2)})"
        cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # === EXPANDED ROI LOGIC ===
        padding = int(0.2 * max(w, h))  # 20% of the larger hand dimension
        x = max(0, x - padding)
        y = max(0, y - padding)
        x_end = min(frame.shape[1], x + w + 2 * padding)
        y_end = min(frame.shape[0], y + h + 2 * padding)
        roi = frame[y:y_end, x:x_end]

        if roi.size == 0:
            continue

        thresh = preprocess_roi(roi)
        contour = find_hand_contour(thresh)

        if contour is not None and cv2.contourArea(contour) > 1000:
            fingers = count_fingers(contour, roi)
            gesture = classify_gesture(fingers)
            smoothed_gesture = smooth_prediction(gesture)

            cv2.putText(roi, smoothed_gesture, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.drawContours(roi, [contour], -1, (0, 255, 0), 2)

        cv2.imshow("Hand ROI", roi)
        cv2.imshow("Threshold", thresh)

    cv2.imshow("YOLO + Gesture Recognition", frame)
    rval, frame = vc.read()

    key = cv2.waitKey(20)
    if key == 27:
        break

cv2.destroyAllWindows()
vc.release()
