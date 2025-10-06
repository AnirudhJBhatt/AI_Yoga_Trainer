import cv2
import mediapipe as mp
import numpy as np
import joblib
import warnings
import time
from pose_registry.registry import get_pose_corrections

warnings.filterwarnings("ignore")

# Load trained model
model = joblib.load("models/yoga_model.joblib")

# Mediapipe setup
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Keypoint angle mappings
key_point_indices = {
    "left_wrist_angle": (13, 15, 17),
    "right_wrist_angle": (14, 16, 18),
    "left_elbow_angle": (11, 13, 15),
    "right_elbow_angle": (12, 14, 16),
    "left_shoulder_angle": (13, 11, 23),
    "right_shoulder_angle": (14, 12, 24),
    "left_knee_angle": (23, 25, 27),
    "right_knee_angle": (24, 26, 28),
    "left_ankle_angle": (25, 27, 29),
    "right_ankle_angle": (26, 28, 30),
    "left_hip_angle": (11, 23, 25),
    "right_hip_angle": (12, 24, 26)
}

def calculate_angle(p1, p2, p3):
    v1 = np.array([p1[0]-p2[0], p1[1]-p2[1], p1[2]-p2[2]])
    v2 = np.array([p3[0]-p2[0], p3[1]-p2[1], p3[2]-p2[2]])
    if np.linalg.norm(v1) == 0 or np.linalg.norm(v2) == 0:
        return 0.0
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    return np.degrees(np.arccos(cos_angle))

def draw_landmarks_with_confidence(image, landmarks, width, height):
    for idx, lm in enumerate(landmarks):
        x, y = int(lm.x * width), int(lm.y * height)
        vis = lm.visibility
        color = (0, 255, 0) if vis > 0.8 else (0, 255, 255) if vis > 0.5 else (0, 0, 255)
        cv2.circle(image, (x, y), 6, color, -1)

cap = cv2.VideoCapture(0)
cv2.namedWindow("Yoga Pose Detection", cv2.WINDOW_NORMAL)
prev_time = time.time()


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)
    h, w, _ = frame.shape

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        draw_landmarks_with_confidence(frame, landmarks, w, h)

        angles = []
        angle_dict = {}

        for key, (p1_idx, p2_idx, p3_idx) in key_point_indices.items():
            if (landmarks[p1_idx].visibility > 0.5 and
                landmarks[p2_idx].visibility > 0.5 and
                landmarks[p3_idx].visibility > 0.5):
                p1 = [landmarks[p1_idx].x, landmarks[p1_idx].y, landmarks[p1_idx].z]
                p2 = [landmarks[p2_idx].x, landmarks[p2_idx].y, landmarks[p2_idx].z]
                p3 = [landmarks[p3_idx].x, landmarks[p3_idx].y, landmarks[p3_idx].z]
                angle = calculate_angle(p1, p2, p3)
            else:
                angle = 0.0
            angles.append(angle)
            angle_dict[key] = angle

        try:
            input_angles = np.array(angles).reshape(1, -1)
            pose_name = model.predict(input_angles)[0]
            corrections = get_pose_corrections(pose_name, angle_dict)
        except Exception as e:
            pose_name = "Error"
            corrections = [f"Prediction error: {e}"]

        cv2.putText(frame, f"Pose: {pose_name}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (50, 255, 50), 2)
        for i, correction in enumerate(corrections):
            cv2.putText(frame, correction, (10, 90 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 100, 255), 2)

        print(f"[{time.strftime('%H:%M:%S')}] Pose: {pose_name} | Corrections: {corrections}")

        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time
    cv2.putText(frame, f'FPS: {int(fps)}', (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    cv2.imshow("Yoga Pose Detection", frame)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()