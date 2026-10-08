import cv2
import mediapipe as mp
import numpy as np
import os
import time
import shutil

# ============================================================
# SETTINGS
# ============================================================

print("="*50)
print("GESTURE DATASET COLLECTOR")
print("="*50)

print("\nSelect Gesture Class")
print("1 --> COME")
print("2 --> NOT_COME")

choice = input("\nEnter choice: ")

if choice == "1":
    LABEL = "come"
elif choice == "2":
    LABEL = "not_come"
else:
    print("Invalid choice.")
    raise SystemExit

NUM_SAMPLES = 100

SEQUENCE_LENGTH = 30

RECORDING_TIME = 3.0

COUNTDOWN_TIME = 5

MODEL_PATH = "models/hand_landmarker.task"

DATASET_FOLDER = "gesture_dataset"

SAVE_FOLDER = os.path.join(DATASET_FOLDER, LABEL)

# ============================================================
# DELETE PREVIOUS DATA
# ============================================================

if os.path.exists(SAVE_FOLDER):

    print("\nDeleting previous", LABEL, "samples...")

    shutil.rmtree(SAVE_FOLDER)

os.makedirs(SAVE_FOLDER)

sample_number = 0

print("Folder ready.")

# ============================================================
# CHECK MODEL
# ============================================================

if not os.path.exists(MODEL_PATH):

    print("hand_landmarker.task not found!")

    raise SystemExit

# ============================================================
# MEDIAPIPE SETUP
# ============================================================

BaseOptions = mp.tasks.BaseOptions

HandLandmarker = mp.tasks.vision.HandLandmarker

HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions

RunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(

    base_options=BaseOptions(

        model_asset_path=MODEL_PATH

    ),

    running_mode=RunningMode.VIDEO,

    num_hands=1,

    min_hand_detection_confidence=0.6,

    min_hand_presence_confidence=0.6,

    min_tracking_confidence=0.6

)

detector = HandLandmarker.create_from_options(options)

# ============================================================
# OPEN CAMERA
# ============================================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():

    print("Camera not found.")

    detector.close()

    raise SystemExit

timestamp_ms = 0

# ============================================================
# HAND DETECTION
# ============================================================

def detect_hand(frame, timestamp):

    rgb = cv2.cvtColor(

        frame,

        cv2.COLOR_BGR2RGB

    )

    rgb = np.ascontiguousarray(rgb)

    mp_image = mp.Image(

        image_format=mp.ImageFormat.SRGB,

        data=rgb

    )

    result = detector.detect_for_video(

        mp_image,

        timestamp

    )

    if not result.hand_landmarks:

        return None, None

    hand = result.hand_landmarks[0]

    landmarks = []

    for point in hand:

        landmarks.extend([

            point.x,

            point.y,

            point.z

        ])

    landmarks = np.array(

        landmarks,

        dtype=np.float32

    )

    return landmarks, hand

# ============================================================
# DRAW LANDMARKS
# ============================================================

def draw_hand(frame, hand):

    h, w, _ = frame.shape

    for point in hand:

        x = int(point.x * w)

        y = int(point.y * h)

        cv2.circle(

            frame,

            (x, y),

            5,

            (0,255,0),

            -1

        )

print("\n====================================")
print("Collector Ready")
print("Gesture :", LABEL)
print("Samples :", NUM_SAMPLES)
print("Countdown :", COUNTDOWN_TIME)
print("Recording :", RECORDING_TIME)
print("====================================")

# ============================================================
# MAIN LOOP
# ============================================================

while cap.isOpened():

    if sample_number >= NUM_SAMPLES:
        break

    ret, frame = cap.read()

    if not ret:
        break

    frame = cv2.flip(frame, 1)

    cv2.putText(frame,
                f"Gesture : {LABEL}",
                (20,40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255,255,255),
                2)

    cv2.putText(frame,
                f"Samples : {sample_number}/{NUM_SAMPLES}",
                (20,80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255,255,255),
                2)

    cv2.putText(frame,
                "Press S to Record",
                (20,120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0,255,255),
                2)

    cv2.putText(frame,
                "Press Q to Quit",
                (20,160),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255,255,255),
                2)

    cv2.imshow("Gesture Collector", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

    if key != ord('s'):
        continue

    # ============================================
    # COUNTDOWN
    # ============================================

    for sec in range(COUNTDOWN_TIME,0,-1):

        ret,count_frame=cap.read()

        if not ret:
            break

        count_frame=cv2.flip(count_frame,1)

        cv2.putText(count_frame,
                    "GET READY",
                    (160,120),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.5,
                    (255,255,255),
                    4)

        cv2.putText(count_frame,
                    str(sec),
                    (280,300),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    5,
                    (0,0,255),
                    8)

        cv2.imshow("Gesture Collector",count_frame)

        cv2.waitKey(1000)

    print("\nRecording Sample",sample_number+1)

    all_landmarks=[]

    start=time.time()

    while (time.time()-start)<RECORDING_TIME:

        ret,record_frame=cap.read()

        if not ret:
            break

        record_frame=cv2.flip(record_frame,1)

        timestamp_ms+=33

        landmarks,hand=detect_hand(
            record_frame,
            timestamp_ms
        )

        elapsed=time.time()-start

        remaining=max(
            0,
            RECORDING_TIME-elapsed
        )

        if landmarks is not None:

            all_landmarks.append(
                landmarks
            )

            draw_hand(
                record_frame,
                hand
            )

            colour=(0,255,0)

        else:

            colour=(0,0,255)

        cv2.putText(record_frame,
                    f"Recording {remaining:.1f}s",
                    (20,40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    colour,
                    2)

        cv2.putText(record_frame,
                    f"Frames : {len(all_landmarks)}",
                    (20,80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255,255,255),
                    2)

        cv2.imshow(
            "Gesture Collector",
            record_frame
        )

        if (cv2.waitKey(1)&0xFF)==ord('q'):
            break
    # ============================================================
    # CHECK RECORDING
    # ============================================================

    print("Valid Frames :", len(all_landmarks))

    if len(all_landmarks) >= SEQUENCE_LENGTH:

        # Select 30 evenly spaced frames

        indices = np.linspace(
            0,
            len(all_landmarks)-1,
            SEQUENCE_LENGTH
        ).astype(int)

        sequence = np.array(
            [all_landmarks[i] for i in indices],
            dtype=np.float32
        )

        filename = os.path.join(
            SAVE_FOLDER,
            f"{LABEL}_{sample_number:04d}.npy"
        )

        np.save(
            filename,
            sequence
        )

        sample_number += 1

        print("\nSAVED :", filename)
        print("Shape :", sequence.shape)

        success = record_frame.copy()

        cv2.putText(
            success,
            "SAMPLE SAVED",
            (120,220),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.4,
            (0,255,0),
            4
        )

        cv2.putText(
            success,
            f"{sample_number}/{NUM_SAMPLES}",
            (230,290),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (255,255,255),
            3
        )

        cv2.imshow(
            "Gesture Collector",
            success
        )

        cv2.waitKey(1000)

    else:

        print("\nSample discarded.")
        print("Hand not visible for enough frames.")

        fail = record_frame.copy()

        cv2.putText(
            fail,
            "SAMPLE DISCARDED",
            (70,220),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0,0,255),
            4
        )

        cv2.putText(
            fail,
            "Keep hand visible",
            (150,280),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255,255,255),
            2
        )

        cv2.imshow(
            "Gesture Collector",
            fail
        )

        cv2.waitKey(1500)

# ============================================================
# CLEANUP
# ============================================================

cap.release()

cv2.destroyAllWindows()

detector.close()

print("\n======================================")
print("DATA COLLECTION COMPLETE")
print("======================================")
print("Gesture :", LABEL)
print("Samples Collected :", sample_number)
print("Saved Folder :", SAVE_FOLDER)
print("Each Sample Shape :", (30,63))
print("======================================")