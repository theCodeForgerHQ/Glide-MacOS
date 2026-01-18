import cv2
import numpy as np
import pyautogui
import threading
import time
from control import enabled
import mediapipe as mp

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

model_path = "./hand_landmarker.task"
screen_width, screen_height = pyautogui.size()

def map_coords(x, y):
    return int(x * screen_width), int(y * screen_height)

def result_callback(result: HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    if not enabled:
        return
    if result.hand_landmarks:
        hand = result.hand_landmarks[0]
        x = hand[8].x  # index fingertip
        y = hand[8].y
        mx, my = map_coords(x, y)
        pyautogui.moveTo(mx, my)

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.LIVE_STREAM,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    result_callback=result_callback
)
landmarker = HandLandmarker.create_from_options(options)

def to_mediapipe_image(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=np.asarray(rgb))

def gesture_loop():
    cap = cv2.VideoCapture(0)
    timestamp = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        mp_image = to_mediapipe_image(frame)
        landmarker.detect_async(mp_image, timestamp)
        timestamp += int(1000 / 30)
        time.sleep(0.01)

def start_gesture_thread():
    t = threading.Thread(target=gesture_loop, daemon=True)
    t.start()
