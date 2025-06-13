# pose_registry/dhanurasana.py

from .base_pose import BasePose

class Dhanurasana(BasePose):
    angles = {
        "left_knee_angle": (100, 130),
        "right_knee_angle": (100, 130),
        "left_hip_angle": (30, 60),
        "right_hip_angle": (30, 60),
        "left_shoulder_angle": (30, 60),
        "right_shoulder_angle": (30, 60),
    }

    def get_feedback(self, angles: dict) -> list:
        corrections = []
        for joint, (min_angle, max_angle) in self.angles.items():
            actual = angles.get(joint, 0.0)
            if not (min_angle <= actual <= max_angle):
                corrections.append(
                    f"Adjust your {joint.replace('_', ' ')}: {int(actual)}° (expected {min_angle}°–{max_angle}°)"
                )
        if not corrections:
            corrections.append("Great job! Hold the pose.")
        return corrections
