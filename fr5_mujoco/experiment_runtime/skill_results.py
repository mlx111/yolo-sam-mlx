"""Structured skill execution result records."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


@dataclass
class SkillResult:
    skill: str
    success: bool
    reason: str = ""
    phase: str = ""
    source: str = ""
    contact: dict[str, Any] = field(default_factory=dict)
    gripper_action: float | None = None
    tracked_body: str = ""
    pinch_distance: float | None = None
    object_pos: Any = None
    target_pos: Any = None
    target_rot: Any = None
    final_pos: Any = None
    final_rot: Any = None
    pos_error: float | None = None
    rot_error: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return _jsonable({
            "skill": self.skill,
            "success": bool(self.success),
            "reason": self.reason,
            "phase": self.phase,
            "source": self.source,
            "contact": self.contact,
            "gripper_action": self.gripper_action,
            "tracked_body": self.tracked_body,
            "pinch_distance": self.pinch_distance,
            "object_pos": self.object_pos,
            "target_pos": self.target_pos,
            "target_rot": self.target_rot,
            "final_pos": self.final_pos,
            "final_rot": self.final_rot,
            "pos_error": self.pos_error,
            "rot_error": self.rot_error,
            "extra": self.extra,
        })


def skill_result(*, skill: str, success: bool, reason: str = "", **kwargs: Any) -> SkillResult:
    fields = set(SkillResult.__dataclass_fields__)
    payload: dict[str, Any] = {"skill": skill, "success": success, "reason": reason}
    extra = dict(kwargs.pop("extra", {}) or {})
    for key, value in kwargs.items():
        if key in fields:
            payload[key] = value
        else:
            extra[key] = value
    payload["extra"] = extra
    return SkillResult(**payload)
