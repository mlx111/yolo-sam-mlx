from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import mujoco

from skills.torso_frame._common import side_gripper


@dataclass(frozen=True)
class TorsoFrameOpenGripperResult:
    success: bool
    side: str
    gripper_value: float
    message: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class R1ProTorsoFrameOpenGripperSkill:
    def execute_recovery_action(self, model: mujoco.MjModel, data: mujoco.MjData, params: dict[str, Any]) -> TorsoFrameOpenGripperResult:
        side = str(params.get("side", "left"))
        value = side_gripper(
            model,
            data,
            side,
            "open",
            gripper_steps=int(params.get("gripper_steps", 240)),
            direct_qpos=bool(params.get("direct_qpos", False)),
            hold_arm=True,
        )
        return TorsoFrameOpenGripperResult(success=True, side=side, gripper_value=float(value), message=f"gripper_value={value:.6f}")


def load_skill(path: str | Path | None = None) -> R1ProTorsoFrameOpenGripperSkill:
    return R1ProTorsoFrameOpenGripperSkill()
