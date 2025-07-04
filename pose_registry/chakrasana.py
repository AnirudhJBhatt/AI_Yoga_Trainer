from .base_pose import BasePose

class Chakrasana(BasePose):
    def get_feedback(self, angles: dict) -> list:
        feedback = []

        left_knee = angles.get("left_knee_angle", 0.0)
        right_knee = angles.get("right_knee_angle", 0.0)
        left_shoulder = angles.get("left_shoulder_angle", 0.0)
        right_shoulder = angles.get("right_shoulder_angle", 0.0)

        if left_knee < 150 or right_knee < 150:
            feedback.append("Try to straighten your legs more.")

        if left_shoulder < 150 or right_shoulder < 150:
            feedback.append("Lift your chest and arch your back more.")

        if not feedback:
            feedback.append("Perfect Chakrasana! Keep breathing deeply.")

        return feedback
