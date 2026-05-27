import cv2
import mediapipe as mp
import numpy as np
import urllib.request
import os

# ─── Helpers ───────────────────────────────────────────────

def calculate_angle(a, b, c):
    """Calculate angle at point B between points A, B, C"""
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - \
              np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle

# ─── State Machine ─────────────────────────────────────────

POSE_VISIBLE   = "POSE_VISIBLE"
READY_TOP      = "READY_TOP"
DESCENDING     = "DESCENDING"
BOTTOM_HOLD    = "BOTTOM_HOLD"
ASCENDING      = "ASCENDING"
REP_COUNTED    = "REP_COUNTED"

class PushUpCounter:
    def __init__(self):
        self.state = POSE_VISIBLE
        self.rep_count = 0
        self.angle = 0

    def update(self, angle):
        self.angle = angle

        if self.state == POSE_VISIBLE:
            # Wait until person is in top position
            if angle > 155:
                self.state = READY_TOP

        elif self.state == READY_TOP:
            # Start descending when angle drops
            if angle < 140:
                self.state = DESCENDING

        elif self.state == DESCENDING:
            # Reached bottom when angle is small enough
            if angle < 90:
                self.state = BOTTOM_HOLD

        elif self.state == BOTTOM_HOLD:
            # Start ascending when angle increases
            if angle > 90:
                self.state = ASCENDING

        elif self.state == ASCENDING:
            # Rep completed when back to top
            if angle > 155:
                self.state = REP_COUNTED

        elif self.state == REP_COUNTED:
            self.rep_count += 1
            self.state = READY_TOP

        return self.state, self.rep_count

# ─── Main ──────────────────────────────────────────────────

# Download model if needed
model_path = 'pose_landmarker.task'
if not os.path.exists(model_path):
    print("Downloading pose model...")
    urllib.request.urlretrieve(
        'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task',
        model_path
    )
    print("Done!")

# Setup MediaPipe
BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.IMAGE
)

counter = PushUpCounter()
cap = cv2.VideoCapture(0)
print("Started! Do push-ups in front of camera. Press Q to quit.")

with PoseLandmarker.create_from_options(options) as landmarker:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        h, w, _ = frame.shape
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        )
        result = landmarker.detect(mp_image)

        if result.pose_landmarks:
            landmarks = result.pose_landmarks[0]

            # Get coordinates for left arm (shoulder, elbow, wrist)
            shoulder = [landmarks[11].x * w, landmarks[11].y * h]
            elbow    = [landmarks[13].x * w, landmarks[13].y * h]
            wrist    = [landmarks[15].x * w, landmarks[15].y * h]

            # Calculate elbow angle
            angle = calculate_angle(shoulder, elbow, wrist)

            # Update state machine
            state, reps = counter.update(angle)

            # Draw landmarks
            for lm in landmarks:
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)

            # Draw arm points bigger
            for point in [shoulder, elbow, wrist]:
                cv2.circle(frame, (int(point[0]), int(point[1])), 8, (0, 0, 255), -1)

            # Display info on screen
            cv2.putText(frame, f'Reps: {reps}',
                        (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
            cv2.putText(frame, f'State: {state}',
                        (30, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            cv2.putText(frame, f'Angle: {int(angle)}',
                        (30, 170), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        cv2.imshow('GymRival - Push Up Counter', frame)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()