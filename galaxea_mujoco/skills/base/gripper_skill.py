from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import mujoco


GripperCommand = Literal["open", "close"]


@dataclass(frozen=True)
class GripperSkillConfig:
    open_value: float
    close_value: float
    joints: tuple[str, ...]
    actuators: tuple[str, ...]


class R1ProGripperSkill:
    """Control the R1Pro two-finger gripper by command or raw opening value."""

    def __init__(self, config: GripperSkillConfig):
        self.config = config

    @classmethod
    def from_json(cls, path: str | Path | None = None) -> "R1ProGripperSkill":
        config_path = Path(path) if path is not None else Path(__file__).with_suffix(".json")
        payload = json.loads(config_path.read_text())
        commands = payload["commands"]
        return cls(
            GripperSkillConfig(
                open_value=float(commands["open"]),
                close_value=float(commands["close"]),
                joints=tuple(payload["joints"]),
                actuators=tuple(payload["actuators"]),
            )
        )

    def command_value(self, command: GripperCommand | int | float) -> float:
        if isinstance(command, str):
            if command == "open":
                return self.config.open_value
            if command == "close":
                return self.config.close_value
            raise ValueError(f"Unsupported gripper command: {command!r}")
        return self.clamp(float(command))

    def clamp(self, value: float) -> float:
        low = min(self.config.open_value, self.config.close_value)
        high = max(self.config.open_value, self.config.close_value)
        return max(low, min(high, value))

    def set_ctrl(self, model: mujoco.MjModel, data: mujoco.MjData, command: GripperCommand | int | float) -> float:
        value = self.command_value(command)
        for actuator_name in self.config.actuators:
            actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name)
            if actuator_id != -1:
                data.ctrl[actuator_id] = value
        return value

    def set_qpos(self, model: mujoco.MjModel, data: mujoco.MjData, command: GripperCommand | int | float) -> float:
        value = self.command_value(command)
        for joint_name in self.config.joints:
            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            if joint_id != -1:
                qpos_id = model.jnt_qposadr[joint_id]
                data.qpos[qpos_id] = value
        return value

    def apply(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        command: GripperCommand | int | float,
        *,
        direct_qpos: bool = False,
    ) -> float:
        value = self.set_ctrl(model, data, command)
        if direct_qpos:
            self.set_qpos(model, data, value)
        return value

    def step_to(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        command: GripperCommand | int | float,
        *,
        steps: int = 120,
    ) -> float:
        value = self.set_ctrl(model, data, command)
        for _ in range(steps):
            mujoco.mj_step(model, data)
        return value

    def execute_recovery_action(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        params: dict,
        *,
        direct_qpos: bool = False,
    ) -> float:
        """Execute a recovery-plan gripper action where state 0=open and 1=close."""
        state = int(params.get("state", 0))
        command: GripperCommand = "close" if state == 1 else "open"
        return self.apply(model, data, command, direct_qpos=direct_qpos)


def load_skill(path: str | Path | None = None) -> R1ProGripperSkill:
    return R1ProGripperSkill.from_json(path)
