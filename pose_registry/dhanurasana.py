from .base_pose import BasePose

class Dhanurasana(BasePose):
    def get_feedback(self, angles: dict) -> list:
        feedback = []

        left_knee = angles.get("left_knee_angle", 0.0)
        right_knee = angles.get("right_knee_angle", 0.0)
        left_hip = angles.get("left_hip_angle", 0.0)
        right_hip = angles.get("right_hip_angle", 0.0)
        left_shoulder = angles.get("left_shoulder_angle", 0.0)
        right_shoulder = angles.get("right_shoulder_angle", 0.0)

        # Knee feedback
        if left_knee > 70 or right_knee > 70:
            feedback.append("Try to bend your knees more.")

        # Hip feedback
        if left_hip < 160 or right_hip < 160:
            feedback.append("Lift your thighs higher off the ground.")

        # Shoulder feedback
        if left_shoulder > 60 or right_shoulder > 60:
            feedback.append("Pull your shoulders back and open your chest.")

        if not feedback:
            feedback.append("Great form! Keep holding the pose.")

        return feedback
