# import cv2
# import numpy as np
# import mediapipe as mp
# import time

# BaseOptions = mp.tasks.BaseOptions
# HandLandmarker = mp.tasks.vision.HandLandmarker
# HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
# HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
# VisionRunningMode = mp.tasks.vision.RunningMode

# model_path = "./hand_landmarker.task"

# cap = cv2.VideoCapture(0)
# frame_to_show = np.zeros((480, 640, 3), dtype=np.uint8)

# # Predefined connections for the 21 hand landmarks
# HAND_CONNECTIONS = [
#     (0, 1), (1, 2), (2, 3), (3, 4),       # Thumb
#     (0, 5), (5, 6), (6, 7), (7, 8),       # Index
#     (0, 9), (9, 10), (10, 11), (11, 12),  # Middle
#     (0, 13), (13, 14), (14, 15), (15, 16),# Ring
#     (0, 17), (17, 18), (18, 19), (19, 20) # Pinky
# ]

# def result_callback(result: HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
#     global frame_to_show
#     frame_to_show = np.array(output_image.numpy_view(), copy=True)
#     if result.hand_landmarks:
#         for hand in result.hand_landmarks:
#             # Draw joints
#             for lm in hand:
#                 x = max(0, min(int(lm.x * frame_to_show.shape[1]), frame_to_show.shape[1]-1))
#                 y = max(0, min(int(lm.y * frame_to_show.shape[0]), frame_to_show.shape[0]-1))
#                 cv2.circle(frame_to_show, (x, y), 5, (0, 255, 0), -1)
#             # Draw bones
#             for connection in HAND_CONNECTIONS:
#                 start_idx, end_idx = connection
#                 x0 = max(0, min(int(hand[start_idx].x * frame_to_show.shape[1]), frame_to_show.shape[1]-1))
#                 y0 = max(0, min(int(hand[start_idx].y * frame_to_show.shape[0]), frame_to_show.shape[0]-1))
#                 x1 = max(0, min(int(hand[end_idx].x * frame_to_show.shape[1]), frame_to_show.shape[1]-1))
#                 y1 = max(0, min(int(hand[end_idx].y * frame_to_show.shape[0]), frame_to_show.shape[0]-1))
#                 cv2.line(frame_to_show, (x0, y0), (x1, y1), (0, 255, 255), 2)

# options = HandLandmarkerOptions(
#     base_options=BaseOptions(model_asset_path=model_path),
#     running_mode=VisionRunningMode.LIVE_STREAM,
#     num_hands=1,
#     min_hand_detection_confidence=0.5,
#     min_tracking_confidence=0.5,
#     result_callback=result_callback
# )
# landmarker = HandLandmarker.create_from_options(options)

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         continue
#     mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
#     timestamp = int(time.time() * 1000)
#     landmarker.detect_async(mp_image, timestamp)
#     cv2.imshow("Hand Landmarker Preview", frame_to_show)
#     if cv2.waitKey(1) & 0xFF == ord("q"):
#         break

# cap.release()
# cv2.destroyAllWindows()

import cv2
import numpy as np
import mediapipe as mp
import time
from collections import deque

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

model_path = "./hand_landmarker.task"

cap = cv2.VideoCapture(0)
frame_to_show = np.zeros((480, 640, 3), dtype=np.uint8)
last_safe_frame = frame_to_show.copy()

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20)
]

x_history = deque(maxlen=12)
pinch_history = deque(maxlen=6)
current_gesture = ""

def distance(a, b):
    return np.linalg.norm(np.array([a.x - b.x, a.y - b.y]))

def detect_gestures(hand):
    global current_gesture
    palm_x = hand[0].x
    x_history.append(palm_x)

    tips = [hand[i] for i in [4, 8, 12, 16, 20]]
    center = np.mean([[lm.x, lm.y] for lm in tips], axis=0)
    spread = np.mean([np.linalg.norm(np.array([lm.x, lm.y]) - center) for lm in tips])
    pinch_history.append(spread)

    if len(x_history) == x_history.maxlen:
        dx = x_history[-1] - x_history[0]
        if dx > 0.15:
            current_gesture = "WAVE LEFT TO RIGHT"
            print(current_gesture)
            x_history.clear()
        elif dx < -0.15:
            current_gesture = "WAVE RIGHT TO LEFT"
            print(current_gesture)
            x_history.clear()

    if len(pinch_history) == pinch_history.maxlen:
        if pinch_history[0] > 0.06 and pinch_history[-1] < 0.025:
            current_gesture = "FINGERS TOGETHER"
            print(current_gesture)
            pinch_history.clear()
        elif pinch_history[0] < 0.025 and pinch_history[-1] > 0.06:
            current_gesture = "FINGERS APART"
            print(current_gesture)
            pinch_history.clear()

def result_callback(result: HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    global frame_to_show, last_safe_frame
    try:
        frame = np.array(output_image.numpy_view(), copy=True)
        if result.hand_landmarks:
            for hand in result.hand_landmarks:
                detect_gestures(hand)
                for lm in hand:
                    x = int(lm.x * frame.shape[1])
                    y = int(lm.y * frame.shape[0])
                    if 0 <= x < frame.shape[1] and 0 <= y < frame.shape[0]:
                        cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)
                for a, b in HAND_CONNECTIONS:
                    x0 = int(hand[a].x * frame.shape[1])
                    y0 = int(hand[a].y * frame.shape[0])
                    x1 = int(hand[b].x * frame.shape[1])
                    y1 = int(hand[b].y * frame.shape[0])
                    if 0 <= x0 < frame.shape[1] and 0 <= y0 < frame.shape[0] and 0 <= x1 < frame.shape[1] and 0 <= y1 < frame.shape[0]:
                        cv2.line(frame, (x0, y0), (x1, y1), (0, 255, 255), 2)
        frame_to_show = frame
        last_safe_frame = frame.copy()
    except Exception:
        frame_to_show = last_safe_frame.copy()

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.LIVE_STREAM,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    result_callback=result_callback
)

landmarker = HandLandmarker.create_from_options(options)

while True:
    ret, frame = cap.read()
    if not ret:
        continue
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    landmarker.detect_async(mp_image, int(time.time() * 1000))
    try:
        display = frame_to_show.copy()
        if current_gesture:
            cv2.putText(display, current_gesture, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        cv2.imshow("Hand Landmarker Preview", display)
    except cv2.error:
        pass
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
