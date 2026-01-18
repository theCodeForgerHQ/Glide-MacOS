import cv2
import pyautogui
import threading
import time
from control import enabled
from mediapipe.solutions.hands import Hands

hands = Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.5)

screen_width, screen_height = pyautogui.size()

def map_coords(x, y):
    return int(x * screen_width), int(y * screen_height)

def gesture_loop():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)
        if results.multi_hand_landmarks and enabled:
            hand = results.multi_hand_landmarks[0]
            x = hand.landmark[8].x
            y = hand.landmark[8].y
            mx, my = map_coords(x, y)
            pyautogui.moveTo(mx, my)
        time.sleep(0.01)

def start_gesture_thread():
    t = threading.Thread(target=gesture_loop, daemon=True)
    t.start()
