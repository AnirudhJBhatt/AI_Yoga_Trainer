import cv2
import mediapipe as mp
import numpy as np
import time

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

# Checkbox state for each angle
angle_states = {key: False for key in key_point_indices}
checkbox_positions = {}

# Mouse callback to toggle checkbox state
def checkbox_click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        for joint, (bx, by) in checkbox_positions.items():
            if bx <= x <= bx + 20 and by <= y <= by + 20:
                angle_states[joint] = not angle_states[joint]

def calculate_angle(p1, p2, p3):
    v1 = np.array([p1[0] - p2[0], p1[1] - p2[1], p1[2] - p2[2]])
    v2 = np.array([p3[0] - p2[0], p3[1] - p2[1], p3[2] - p2[2]])
    if np.linalg.norm(v1) == 0 or np.linalg.norm(v2) == 0:
        return 0.0
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    return np.degrees(np.arccos(cos_angle))

# Draw checkbox panel on the frame
def draw_checkboxes(frame):
    x_start, y_start = 10, 10
    spacing = 30
    for idx, (joint, state) in enumerate(angle_states.items()):
        y_pos = y_start + idx * spacing
        checkbox_positions[joint] = (x_start, y_pos)

        # Draw filled checkbox
        color = (0, 255, 0) if state else (255, 255, 255)
        cv2.rectangle(frame, (x_start, y_pos), (x_start + 20, y_pos + 20), color, -1)
        cv2.rectangle(frame, (x_start, y_pos), (x_start + 20, y_pos + 20), (0, 0, 0), 2)

        # Draw label
        label = joint.replace("_angle", "").replace("_", " ").title()
        cv2.putText(frame, label, (x_start + 30, y_pos + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

cap = cv2.VideoCapture(1)
cv2.namedWindow("Joint Angle Visualization", cv2.WINDOW_NORMAL)
cv2.setMouseCallback("Joint Angle Visualization", checkbox_click)
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
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        for key, (p1_idx, p2_idx, p3_idx) in key_point_indices.items():
            if not angle_states.get(key, False):
                continue

            if (landmarks[p1_idx].visibility > 0.5 and
                landmarks[p2_idx].visibility > 0.5 and
                landmarks[p3_idx].visibility > 0.5):
                p1 = [landmarks[p1_idx].x * w, landmarks[p1_idx].y * h, landmarks[p1_idx].z]
                p2 = [landmarks[p2_idx].x * w, landmarks[p2_idx].y * h, landmarks[p2_idx].z]
                p3 = [landmarks[p3_idx].x * w, landmarks[p3_idx].y * h, landmarks[p3_idx].z]

                angle = calculate_angle(p1, p2, p3)
                x, y = int(landmarks[p2_idx].x * w), int(landmarks[p2_idx].y * h)

                # Draw angle arc and label
                cv2.line(frame, (int(p1[0]), int(p1[1])), (x, y), (255, 255, 0), 2)
                cv2.line(frame, (int(p3[0]), int(p3[1])), (x, y), (255, 255, 0), 2)
                cv2.circle(frame, (x, y), 6, (255, 0, 255), -1)
                # cv2.ellipse(frame, (x, y), (30, 30), 0, 0, int(angle), (255, 0, 0), 2)
                cv2.putText(frame, f"{key}: {int(angle)}°", (x + 10, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Draw panel of checkboxes
    draw_checkboxes(frame)

    # FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time
    cv2.putText(frame, f'FPS: {int(fps)}', (10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    cv2.imshow("Joint Angle Visualization", frame)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
