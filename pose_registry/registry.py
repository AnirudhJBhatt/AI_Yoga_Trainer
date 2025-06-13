# pose_registry/registry.py

from .dhanurasana import Dhanurasana
# Import more poses here as you add them

# Registry mapping pose names to classes
pose_classes = {
    "Dhanurasana": Dhanurasana(),
    # Add others
}

def get_pose_corrections(pose_name, angles_dict):
    pose = pose_classes.get(pose_name)
    if pose is None:
        return [f"No correction rules defined for: {pose_name}"]
    return pose.get_feedback(angles_dict)
