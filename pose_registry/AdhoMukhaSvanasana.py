from .base_pose import BasePose

class AdhoMukhaSvanasana(BasePose):
    def get_feedback(self, angles: dict) -> list:
        feedback = []

        left_shoulder = angles.get("left_shoulder_angle", 0.0)
        right_shoulder = angles.get("right_shoulder_angle", 0.0)
        left_hip = angles.get("left_hip_angle", 0.0)
        right_hip = angles.get("right_hip_angle", 0.0)

        if left_shoulder < 160 or right_shoulder < 160:
            feedback.append("Straighten your arms to push your torso backward.")

        if left_hip > 110 or right_hip > 110:
            feedback.append("Try to raise your hips higher to form an inverted V shape.")

        if not feedback:
            feedback.append("Great form! Keep holding the pose.")

        return feedback