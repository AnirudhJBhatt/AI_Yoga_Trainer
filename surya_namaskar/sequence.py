import time

class SuryaNamaskarFlow:
    def __init__(self, model, pose_sequence, threshold=0.8, hold_time=2):
        self.model = model
        self.pose_sequence = pose_sequence
        self.threshold = threshold
        self.hold_time = hold_time
        self.current_index = 0
        self.timer = None
        self.complete = False

    def update(self, landmarks):
        if self.complete:
            return "Surya Namaskar Completed"

        predicted_pose, confidence = self.model.predict([landmarks])

        expected_pose = self.pose_sequence[self.current_index]

        if predicted_pose == expected_pose and confidence >= self.threshold:
            if self.timer is None:
                self.timer = time.time()
            elif time.time() - self.timer >= self.hold_time:
                print(f"[✓] {expected_pose} held for {self.hold_time} sec.")
                self.current_index += 1
                self.timer = None
                if self.current_index == len(self.pose_sequence):
                    self.complete = True
                    return "Surya Namaskar Completed"
        else:
            self.timer = None

        return f"Hold {self.pose_sequence[self.current_index]}"
