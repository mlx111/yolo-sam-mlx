"""UR5e-specific constants layered on top of the universal schema."""

from __future__ import annotations

from experience_core.schema import WRAPPER1_UR5E_NAMESPACE
from . import runtime_skills


UR5E_NAMESPACE = WRAPPER1_UR5E_NAMESPACE
UR5E_ROBOT_TYPE = "fixed_single_arm"
UR5E_BACKEND = "mujoco"

UR5E_ALLOWED_SKILLS = runtime_skills.allowed_actions()
