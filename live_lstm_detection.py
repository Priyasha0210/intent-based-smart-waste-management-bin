import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from collections import deque

# ============================================================
# SETTINGS
# ============================================================

MODEL_PATH = "models/come_gesture_model.keras"

LANDMARK_MODEL = "models/hand_landmarker.task"

SEQUENCE_LENGTH = 30

FEATURES = 63

THRESHOLD = 0.5

# ============================================================
# LOAD MODEL
# ============================================================

print("="*50)
print("Loading Model...")
print("="*50)

model = tf.keras.models.load_model(MODEL_PATH)

print("Model Loaded Successfully!")

# ============================================================
# MEDIAPIPE
# ============================================================

BaseOptions = mp.tasks.BaseOptions

HandLandmarker = mp.tasks.vision.HandLandmarker

HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions

RunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(

    base_options=BaseOptions(
        model_asset_path=LANDMARK_MODEL
    ),

    running_mode=RunningMode.VIDEO,

    num_hands=1,

    min_hand_detection_confidence=0.6,

    min_hand_presence_confidence=0.6,

    min_tracking_confidence=0.6

)

detector = HandLandmarker.create_from_options(options)

# ============================================================
# CAMERA
# ============================================================

cap = cv2.VideoCapture(0)

timestamp_ms = 0

sequence = deque(maxlen=SEQUENCE_LENGTH)

# ============================================================
# DETECT HAND
# ============================================================

def detect(frame):

    global timestamp_ms

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    rgb = np.ascontiguousarray(rgb)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    timestamp_ms += 33

    result = detector.detect_for_video(
        mp_image,
        timestamp_ms
    )

    if not result.hand_landmarks:
        return None, None

    hand = result.hand_landmarks[0]

    landmarks = []

    for p in hand:

        landmarks.extend([
            p.x,
            p.y,
            p.z
        ])

    landmarks = np.array(
        landmarks,
        dtype=np.float32
    )

    return landmarks, hand

# ============================================================
# DRAW HAND
# ============================================================

def draw(frame, hand):

    h,w,_ = frame.shape

    for p in hand:

        x = int(p.x*w)

        y = int(p.y*h)

        cv2.circle(
            frame,
            (x,y),
            5,
            (0,255,0),
            -1
        )

print("\nReady!")
print("Press Q to Quit")
# ============================================================
# LIVE DETECTION
# ============================================================

while cap.isOpened():

    ret, frame = cap.read()

    if not ret:
        break

    frame = cv2.flip(frame, 1)

    landmarks, hand = detect(frame)

    prediction = "WAITING..."
    probability = 0.0
    color = (255,255,255)

    # ------------------------------------------
    # Hand Detected
    # ------------------------------------------

    if landmarks is not None:

        draw(frame, hand)

        sequence.append(landmarks)

        cv2.putText(
            frame,
            f"Frames: {len(sequence)}/{SEQUENCE_LENGTH}",
            (20,40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255,255,255),
            2
        )

        if len(sequence) == SEQUENCE_LENGTH:

            sample = np.array(sequence,dtype=np.float32)

            sample = np.expand_dims(sample,axis=0)

            prob = model.predict(sample,verbose=0)[0][0]

            probability = float(prob)
            print(f"Probability: {probability:.3f}")


            if probability >= THRESHOLD:

                prediction = "COME"

                color = (0,255,0)

            else:

                prediction = "NOT COME"

                color = (0,0,255)

    else:

        sequence.clear()

        prediction = "NO HAND"

        color = (255,255,0)

    # ------------------------------------------
    # DISPLAY
    # ------------------------------------------

    cv2.putText(
        frame,
        f"Prediction : {prediction}",
        (20,80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        color,
        2
    )

    cv2.putText(
        frame,
        f"Probability : {probability:.3f}",
        (20,120),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255,255,255),
        2
    )

    cv2.putText(
        frame,
        "Press Q to Quit",
        (20,160),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255,255,255),
        2
    )

    cv2.imshow(
        "COME Gesture Detection",
        frame
    )

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

# ============================================================
# CLEANUP
# ============================================================

cap.release()

cv2.destroyAllWindows()

detector.close()

print("\n===================================")
print("LIVE DETECTION FINISHED")
print("===================================")