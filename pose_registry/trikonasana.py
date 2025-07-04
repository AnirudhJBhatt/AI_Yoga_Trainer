from .base_pose import BasePose

class Trikonasana(BasePose):
    def get_feedback(self, angles: dict) -> list:
        feedback = []

        left_knee = angles.get("left_knee_angle", 0.0)
        right_knee = angles.get("right_knee_angle", 0.0)
        left_hip = angles.get("left_hip_angle", 0.0)
        right_hip = angles.get("right_hip_angle", 0.0)
        left_shoulder = angles.get("left_shoulder_angle", 0.0)
        right_shoulder = angles.get("right_shoulder_angle", 0.0)

        if left_knee < 170 or right_knee < 170:
            feedback.append("Straighten your front leg.")

        if abs(left_hip - 90) > 15 or abs(right_hip - 90) > 15:
            feedback.append("Widen your stance to align the hips.")

        if abs(left_shoulder - 90) > 15 or abs(right_shoulder - 90) > 15:
            feedback.append("Stack your shoulders vertically.")

        if not feedback:
            feedback.append("Excellent Trikonasana! Hold and breathe.")

        return feedback
