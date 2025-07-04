from .base_pose import BasePose

class Bhujangasana(BasePose):
    def get_feedback(self, angles: dict) -> list:
        feedback = []

        left_elbow = angles.get("left_elbow_angle", 0.0)
        right_elbow = angles.get("right_elbow_angle", 0.0)
        left_hip = angles.get("left_hip_angle", 0.0)
        right_hip = angles.get("right_hip_angle", 0.0)

        if left_elbow < 150 or right_elbow < 150:
            feedback.append("Straighten your arms more.")

        if left_hip > 70 or right_hip > 70:
            feedback.append("Lower your hips closer to the mat.")

        if not feedback:
            feedback.append("Well done! Beautiful Bhujangasana alignment.")

        return feedback
