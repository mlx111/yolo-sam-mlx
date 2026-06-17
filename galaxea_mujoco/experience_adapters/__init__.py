"""Compatibility package for migrated experience adapters."""

from __future__ import annotations

from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CANONICAL = _REPO_ROOT / "experience_system" / "experience_adapters"

if _CANONICAL.exists():
    __path__ = [str(_CANONICAL), *[path for path in __path__ if path != str(_CANONICAL)]]

from .r1pro_mujoco import R1ProMujocoAdapter
from .real_episode import RealEpisodeAdapter
from .wrapper1_ur5e import Wrapper1UR5eAdapter

__all__ = ["R1ProMujocoAdapter", "RealEpisodeAdapter", "Wrapper1UR5eAdapter"]
