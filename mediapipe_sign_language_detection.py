import cv2
import mediapipe as mp
import time
import math
from collections import deque

class HandTrackingDynamic:
    def __init__(self, mode=False, maxHands=1, detectionCon=0.5, trackCon=0.5):
        self.__mode__ = mode
        self.__maxHands__ = maxHands
        self.__detectionCon__ = detectionCon
        self.__trackCon__ = trackCon
        self.handsMp = mp.solutions.hands
        self.hands = self.handsMp.Hands(
            static_image_mode=self.__mode__,
            max_num_hands=self.__maxHands__,
            min_detection_confidence=self.__detectionCon__,
            min_tracking_confidence=self.__trackCon__
        )
        self.mpDraw = mp.solutions.drawing_utils
        self.tipIds = [4, 8, 12, 16, 20]
        self.lmsList = []
        self.bbox = None

    def findFingers(self, frame, draw=True):
        imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(frame, handLms, self.handsMp.HAND_CONNECTIONS)
        return frame

    def findPosition(self, frame, handNo=0, draw=True):
        xList, yList = [], []
        self.lmsList = []
        self.bbox = None
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            h, w, _ = frame.shape
            for id, lm in enumerate(myHand.landmark):
                cx, cy = int(lm.x * w), int(lm.y * h)
                xList.append(cx)
                yList.append(cy)
                self.lmsList.append([id, cx, cy])
                if draw:
                    cv2.circle(frame, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
            xmin, xmax = min(xList), max(xList)
            ymin, ymax = min(yList), max(yList)
            self.bbox = (xmin, ymin, xmax, ymax)
            if draw:
                cv2.rectangle(frame, (xmin-20, ymin-20), (xmax+20, ymax+20), (0, 255, 0), 2)
        return self.lmsList, self.bbox

    def findFingerUp(self):
        if not self.lmsList:
            return []
        thumb_x = self.lmsList[4][1]
        index_x = self.lmsList[8][1]
        orientation = 1 if (index_x - thumb_x) > 0 else -1
        ix, iy = self.lmsList[8][1], self.lmsList[8][2]
        px, py = self.lmsList[6][1], self.lmsList[6][2]
        use_x = abs(ix - px) > abs(iy - py)
        fingers = []
        # Thumb always along X
        ip_x = self.lmsList[3][1]
        fingers.append(1 if orientation * (thumb_x - ip_x) > 0 else 0)
        # Other fingers
        for i in range(1, 5):
            tip, pip = self.tipIds[i], self.tipIds[i]-2
            tx, ty = self.lmsList[tip][1], self.lmsList[tip][2]
            px2, py2 = self.lmsList[pip][1], self.lmsList[pip][2]
            if use_x:
                fingers.append(1 if orientation * (tx - px2) > 0 else 0)
            else:
                fingers.append(1 if py2 - ty > 0 else 0)
        return fingers

    def distance(self, id1, id2):
        x1, y1 = self.lmsList[id1][1], self.lmsList[id1][2]
        x2, y2 = self.lmsList[id2][1], self.lmsList[id2][2]
        return math.hypot(x2-x1, y2-y1)

    def classifyStaticASL(self):
        fingers = self.findFingerUp()
        print("Fingers up flags:", fingers)

        if len(fingers) != 5:
            return '...'

        # ——— E: “all fingers down” but thumb tucked under the curled fingers ———
        if fingers == [0, 0, 0, 0, 0]:
            thb_x, thb_y = self.lmsList[4][1], self.lmsList[4][2]
            idx_tip_y = self.lmsList[8][2]
            if thb_y > idx_tip_y:
                return 'E'

        # O
        if fingers == [0, 0, 0, 0, 0] and self.bbox:
            tips = [self.lmsList[i] for i in self.tipIds]
            xs = [p[1] for p in tips]
            ys = [p[2] for p in tips]
            xmin, ymin, xmax, ymax = self.bbox
            hand_w, hand_h = xmax - xmin, ymax - ymin
            if max(xs)-min(xs) < 0.3*hand_w and max(ys)-min(ys) < 0.3*hand_h:
                return 'O'

        # T: all fingers curled, thumb tucked between index & middle
        if fingers == [0, 0, 0, 0, 0]:
            idx_pip_x = self.lmsList[6][1]
            mid_pip_x = self.lmsList[10][1]
            thb_tip_x = self.lmsList[4][1]
            if min(idx_pip_x, mid_pip_x) < thb_tip_x < max(idx_pip_x, mid_pip_x):
                return 'T'

        # S: all fingers curled, thumb tucked in front of the fist
        if fingers == [0, 0, 0, 0, 0]:
            thb_tip_x = self.lmsList[4][1]
            thb_ip_x  = self.lmsList[3][1]
            thb_tip_y = self.lmsList[4][2]
            idx_tip_y = self.lmsList[8][2]
            if thb_tip_x < thb_ip_x and abs(thb_tip_y - idx_tip_y) < 30:
                return 'S'

        # M: all five flags zero (all fingers down), but thumb pinched BETWEEN ring‐PIP & pinky‐PIP,
        #    AND thumb tip lies ABOVE the index TIP (so it isn’t under the fist like in “E”).
        if fingers == [0, 0, 0, 0, 0]:
            thb_x, thb_y = self.lmsList[4][1], self.lmsList[4][2]
            ring_pip_x   = self.lmsList[14][1]
            pinky_pip_x  = self.lmsList[18][1]
            idx_tip_y    = self.lmsList[8][2]
            between_x = min(ring_pip_x, pinky_pip_x) < thb_x < max(ring_pip_x, pinky_pip_x)
            above_idx = (thb_y < idx_tip_y)

            if between_x and above_idx:
                return 'M'


        # N: middle & ring up, thumb tucked between them
        if fingers == [0, 0, 0, 0, 0]:
            thb_x      = self.lmsList[4][1]
            mid_pip_x  = self.lmsList[10][1]
            ring_pip_x = self.lmsList[14][1]
            if min(mid_pip_x, ring_pip_x) < thb_x < max(mid_pip_x, ring_pip_x):
                return 'N'

        # A
        if fingers == [0, 0, 0, 0, 0]:
            if self.lmsList[4][1] > self.lmsList[5][1] and abs(self.lmsList[4][2] - self.lmsList[5][2]) < 40:
                return 'A'

        # B
        if fingers == [0,1,1,1,1]:
            return 'B'

        # C
        if fingers == [1,1,1,1,1]:
            if self.distance(4,8) < 70 and self.distance(8,12) < 80 and self.distance(12,16) < 80:
                return 'C'

        # D
        if fingers == [0, 1, 0, 0, 0] and self.distance(4,10) < 40:
            return 'D'

        # X: only index finger up, but hooked – use Y‐coordinates of TIP, PIP, and MCP
        if fingers == [1, 1, 0, 0, 0]:
            # Grab Y coords for index TIP (id=8), PIP (id=6), and MCP (id=5)
            tip_y = self.lmsList[8][2]
            pip_y = self.lmsList[6][2]
            mcp_y = self.lmsList[5][2]

            if tip_y > pip_y: #and abs(pip_y - mcp_y) < 40:
                return 'X'

        # F
        if fingers == [1, 0, 1, 1, 1] and self.distance(4,8) < 30:
            return 'F'

        # K: thumb between index and middle, Y-aligned simply
        if fingers == [0,1,1,0,0]:
            idx_pip_x = self.lmsList[6][1]
            mid_pip_x = self.lmsList[10][1]
            thb_x     = self.lmsList[4][1]
            between_x = min(idx_pip_x, mid_pip_x) < thb_x < max(idx_pip_x, mid_pip_x)
            idx_pip_y = self.lmsList[6][2]
            thb_y     = self.lmsList[4][2]
            level_y   = abs(thb_y - idx_pip_y) < 30
            if between_x and level_y:
                return 'K'

        # L: index up & other fingers down, thumb pointing right
        if fingers[1] == 1 and fingers[2:] == [0,0,0]:
            idx_up    = self.lmsList[8][2] < self.lmsList[6][2]   # tip above PIP
            thb_right = self.lmsList[4][1] > self.lmsList[3][1]   # tip to the right of IP
            if idx_up and thb_right:
                return 'L'

        # P
        if fingers == [1,1,0,0,0]:
            ix, iy    = self.lmsList[8][1], self.lmsList[8][2]
            pip_x, pip_y = self.lmsList[6][1], self.lmsList[6][2]
            if abs(iy - pip_y) < 40 and self.lmsList[12][2] > self.lmsList[10][2] and self.distance(4,12) < 30:
                thumb_x = self.lmsList[4][1]
                orientation = 1 if (ix - thumb_x) > 0 else -1
                def is_curled_x(t,p,th=20): return orientation * (self.lmsList[p][1] - self.lmsList[t][1]) > th
                if is_curled_x(16,14) and is_curled_x(20,18):
                    return 'P'

        # Q: thumb & index folded downwards, other three fingers curled
        idx_down = self.lmsList[8][2] > self.lmsList[6][2]
        thb_down = self.lmsList[4][2] > self.lmsList[3][2]
        sep      = abs(self.lmsList[8][2] - self.lmsList[4][2]) > 10
        if idx_down and thb_down and sep:
            return 'Q'

        # G
        if fingers == [1,1,0,0,0]:
            ix, iy      = self.lmsList[8][1], self.lmsList[8][2]
            pip_x, pip_y = self.lmsList[6][1], self.lmsList[6][2]
            if abs(iy - pip_y) < 40:
                thumb_x = self.lmsList[4][1]
                xs = [self.lmsList[j][1] for j in [1,2,3,4]]
                if all(xs[i] < xs[i+1] for i in range(3)) or all(xs[i] > xs[i+1] for i in range(3)):
                    orientation = 1 if (ix - thumb_x) > 0 else -1
                    def is_curled_x(t,p,th=20): return orientation * (self.lmsList[p][1] - self.lmsList[t][1]) > th
                    if is_curled_x(12,10) and is_curled_x(16,14) and is_curled_x(20,18):
                        return 'G'

        # H
        if fingers == [1,1,1,0,0]:
            ix        = self.lmsList[8][1]
            thumb_x   = self.lmsList[4][1]
            orientation = 1 if (ix - thumb_x) > 0 else -1
            def is_ext_x(t,p,th=20): return orientation * (self.lmsList[t][1] - self.lmsList[p][1]) > th
            if is_ext_x(8,6) and is_ext_x(12,10):
                def is_curled_x(t,p,th=20): return orientation * (self.lmsList[p][1] - self.lmsList[t][1]) > th
                if is_curled_x(16,14) and is_curled_x(20,18):
                    return 'H'

        # Y vs I: pinky only shape, distinguish by thumb position
        if fingers[1:4] == [0, 0, 0] and fingers[4] == 1:
            pinky_up     = self.lmsList[20][2] < self.lmsList[18][2]
            thumb_spread = abs(self.lmsList[4][1] - self.lmsList[3][1])
            if pinky_up:
                if thumb_spread > 20:
                    return 'Y'
                else:
                    return 'I'

        # R: index & middle up, index crosses over middle
        if fingers == [0,1,1,0,0]:
            thumb_x = self.lmsList[4][1]
            idx_x   = self.lmsList[8][1]
            mid_x   = self.lmsList[12][1]
            orientation = 1 if (idx_x - thumb_x) > 0 else -1
            if orientation * (idx_x - mid_x) < 0:
                return 'R'

        # U: index & middle up with no gap (fingers together)
        if fingers == [0, 1, 1, 0, 0]:
            idx_x = self.lmsList[8][1]
            mid_x = self.lmsList[12][1]
            if abs(idx_x - mid_x) < 30:
                return 'U'

        # V
        if fingers == [0,1,1,0,0]:
            idx_x = self.lmsList[8][1]
            mid_x = self.lmsList[12][1]
            if abs(idx_x - mid_x) > 30:
                return 'V'

        # W
        if fingers == [0,1,1,1,0]:
            return 'W'

        return '...'


def main():
    cap = cv2.VideoCapture(0)
    detector = HandTrackingDynamic()
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if not cap.isOpened():
        print("Cannot open camera")
        return

    N = 5
    buffer = deque(maxlen=N)
    last_display = '...'
    ptime = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        frame = detector.findFingers(frame)
        lmsList, bbox = detector.findPosition(frame)
        letter = detector.classifyStaticASL()
        buffer.append(letter)
        if len(buffer) == N and len(set(buffer)) == 1:
            last_display = buffer[0]

        cv2.putText(frame, f'Slovo: {last_display}', (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
        ctime = time.time()
        fps = 1 / (ctime - ptime + 1e-6)
        ptime = ctime
        cv2.putText(frame, f'FPS: {int(fps)}', (10, 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
        cv2.imshow('ASL Detector', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
