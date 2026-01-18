import cv2
import numpy as np
import mediapipe as mp
import time

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

model_path = "./hand_landmarker.task"

cap = cv2.VideoCapture(0)
frame_to_show = np.zeros((480, 640, 3), dtype=np.uint8)

# Predefined connections for the 21 hand landmarks
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),       # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),       # Index
    (0, 9), (9, 10), (10, 11), (11, 12),  # Middle
    (0, 13), (13, 14), (14, 15), (15, 16),# Ring
    (0, 17), (17, 18), (18, 19), (19, 20) # Pinky
]

def result_callback(result: HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    global frame_to_show
    frame_to_show = np.array(output_image.numpy_view(), copy=True)
    if result.hand_landmarks:
        for hand in result.hand_landmarks:
            # Draw joints
            for lm in hand:
                x = max(0, min(int(lm.x * frame_to_show.shape[1]), frame_to_show.shape[1]-1))
                y = max(0, min(int(lm.y * frame_to_show.shape[0]), frame_to_show.shape[0]-1))
                cv2.circle(frame_to_show, (x, y), 5, (0, 255, 0), -1)
            # Draw bones
            for connection in HAND_CONNECTIONS:
                start_idx, end_idx = connection
                x0 = max(0, min(int(hand[start_idx].x * frame_to_show.shape[1]), frame_to_show.shape[1]-1))
                y0 = max(0, min(int(hand[start_idx].y * frame_to_show.shape[0]), frame_to_show.shape[0]-1))
                x1 = max(0, min(int(hand[end_idx].x * frame_to_show.shape[1]), frame_to_show.shape[1]-1))
                y1 = max(0, min(int(hand[end_idx].y * frame_to_show.shape[0]), frame_to_show.shape[0]-1))
                cv2.line(frame_to_show, (x0, y0), (x1, y1), (0, 255, 255), 2)

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
    timestamp = int(time.time() * 1000)
    landmarker.detect_async(mp_image, timestamp)
    cv2.imshow("Hand Landmarker Preview", frame_to_show)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
