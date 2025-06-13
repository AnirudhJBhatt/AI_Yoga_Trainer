# pose_registry/base_pose.py

class BasePose:
    def get_feedback(self, angles: dict) -> list:
        """
        Subclasses should implement pose-specific angle validation.
        Returns a list of feedback corrections.
        """
        raise NotImplementedError("Subclasses must implement get_feedback")
