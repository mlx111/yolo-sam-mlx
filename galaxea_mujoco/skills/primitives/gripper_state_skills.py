from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import mujoco
import numpy as np

from skills.primitives.object_manipulation_skills import ManipulationSide, _set_side_gripper


@dataclass(frozen=True)
class GripperStateResult:
    name: str
    success: bool
    side: ManipulationSide | str
    state: str
    value: float
    left_value: float | None = None
    right_value: float | None = None
    message: str = ""


class BaseGripperStateSkill:
    def __init__(self, config_path: str | Path | None = None):
        self.config = json.loads(Path(config_path).read_text()) if config_path is not None else {}
        self.name = self.config.get("name", self.__class__.__name__)


def _gripper_value(model: mujoco.MjModel, data: mujoco.MjData, side: ManipulationSide) -> float:
    values = []
    for index in (1, 2):
        joint_name = f"{side}_gripper_finger_joint{index}"
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id < 0:
            raise ValueError(f"MuJoCo joint not found: {joint_name}")
        values.append(float(data.qpos[model.jnt_qposadr[joint_id]]))
    return float(np.mean(values))


def _classify(value: float, *, open_threshold: float, closed_threshold: float) -> str:
    if value <= open_threshold:
        return "open"
    if value >= closed_threshold:
        return "closed"
    return "partial"


class DetectGripperStateSkill(BaseGripperStateSkill):
    def execute_recovery_action(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        params: dict,
        *,
        step_callback: Callable[[], None] | None = None,
    ) -> GripperStateResult:
        del step_callback
        mujoco.mj_forward(model, data)
        side = str(params.get("side", self.config.get("side", "left")))
        if side not in ("left", "right"):
            raise ValueError(f"Unsupported side: {side!r}")
        value = _gripper_value(model, data, side)
        open_threshold = float(params.get("open_threshold", self.config.get("open_threshold", 0.004)))
        closed_threshold = float(params.get("closed_threshold", self.config.get("closed_threshold", 0.020)))
        state = _classify(value, open_threshold=open_threshold, closed_threshold=closed_threshold)
        expected = params.get("expected_state", self.config.get("expected_state"))
        success = True if expected is None else state == str(expected)
        return GripperStateResult(
            self.name,
            success,
            side,
            state,
            value,
            message=f"{side}_gripper={state}, value={value:.6f}",
        )


class ResyncGrippersSkill(BaseGripperStateSkill):
    def execute_recovery_action(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        params: dict,
        *,
        step_callback: Callable[[], None] | None = None,
    ) -> GripperStateResult:
        command = str(params.get("command", self.config.get("command", "close")))
        steps = int(params.get("gripper_steps", self.config.get("gripper_steps", 240)))
        direct_qpos = bool(params.get("direct_qpos", self.config.get("direct_qpos", False)))
        left = _set_side_gripper(model, data, "left", command, steps=steps, direct_qpos=direct_qpos, step_callback=step_callback)
        right = _set_side_gripper(model, data, "right", command, steps=steps, direct_qpos=direct_qpos, step_callback=step_callback)
        value = float(max(abs(left), abs(right)))
        return GripperStateResult(
            self.name,
            True,
            "both",
            command,
            value,
            left_value=float(left),
            right_value=float(right),
            message=f"left={left:.6f}, right={right:.6f}, command={command}",
        )
