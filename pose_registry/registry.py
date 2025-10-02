# pose_registry/registry.py

from .dhanurasana import Dhanurasana
from .AdhoMukhaSvanasana import AdhoMukhaSvanasana
from .bhujangasana import Bhujangasana
from .chakrasana import Chakrasana
from .trikonasana import Trikonasana
# Import more poses here as you add them

# Registry mapping pose names to classes
pose_classes = {
    "Dhanurasana": Dhanurasana(),
    "Adho Mukha Svanasana": AdhoMukhaSvanasana(),
    "Bhujangasana": Bhujangasana(),
    "Chakrasana": Chakrasana(),
    "Trikonasana": Trikonasana()
}

def get_pose_corrections(pose_name, angles_dict):
    pose = pose_classes.get(pose_name)
    if pose is None:
        return [f"No correction rules defined for: {pose_name}"]
    return pose.get_feedback(angles_dict)
