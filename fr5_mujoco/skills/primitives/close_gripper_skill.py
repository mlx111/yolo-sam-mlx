from __future__ import annotations

from typing import Any

from skills import recovery_steps
from skills.primitives._factory import FunctionBackedFR5Skill


def _close(experiment: Any, params: dict[str, Any]) -> None:
    recovery_steps.gripper_action(experiment, {**params, "state": 1})


class CloseGripperSkill(FunctionBackedFR5Skill):
    def __init__(self) -> None:
        super().__init__("close_gripper", _close)


def load_skill() -> CloseGripperSkill:
    return CloseGripperSkill()
