"""Adapters that normalize robot-specific episodes into universal experiences."""

from .r1pro_mujoco import R1ProMujocoAdapter
from .real_episode import RealEpisodeAdapter
try:
    from experience_system.ur5e_core import Wrapper1UR5eAdapter
except ModuleNotFoundError:  # pragma: no cover - script-root fallback
    from ur5e_core import Wrapper1UR5eAdapter

__all__ = ["R1ProMujocoAdapter", "RealEpisodeAdapter", "Wrapper1UR5eAdapter"]
