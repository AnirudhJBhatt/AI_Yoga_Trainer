# pose_registry/registry.py

from .dhanurasana import Dhanurasana
from .vrikshasana import Vrikshasana
from .adho_mukha_svanasana import AdhoMukhaSvanasana
from .bhujangasana import Bhujangasana
from .chakrasana import Chakrasana
from .trikonasana import Trikonasana

# Registry mapping pose names to classes

pose_classes = {
    "Dhanurasana": Dhanurasana(),
    "Vrikshasana": Vrikshasana(),
    "AdhoMukhaSvanasana": AdhoMukhaSvanasana(),
    "Bhujangasana": Bhujangasana(),
    "Chakrasana": Chakrasana(),
    "Trikonasana": Trikonasana(),
}

def get_pose_corrections(pose_name: str, angles: dict) -> list:
    pose = pose_classes.get(pose_name)
    if not pose:
        return []
    return pose.check_pose(angles)
