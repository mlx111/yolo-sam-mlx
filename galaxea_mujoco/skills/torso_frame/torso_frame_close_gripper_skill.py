from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from skills.base.arm_ik_skill import ARM_JOINTS
from skills.torso_frame._common import side_gripper


@dataclass(frozen=True)
class TorsoFrameCloseGripperResult:
    success: bool
    side: str
    gripper_value: float
    message: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class R1ProTorsoFrameCloseGripperSkill:
    def execute_recovery_action(self, model: mujoco.MjModel, data: mujoco.MjData, params: dict[str, Any]) -> TorsoFrameCloseGripperResult:
        side = str(params.get("side", "left"))
        arm_joint_names = ARM_JOINTS[side]
        pre_close_settle_steps = int(params.get("pre_close_settle_steps", 120))
        if pre_close_settle_steps > 0:
            arm_hold = np.asarray(
                [
                    float(data.qpos[model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)]])
                    for name in arm_joint_names
                ],
                dtype=np.float64,
            )
            for _ in range(pre_close_settle_steps):
                for name, target in zip(arm_joint_names, arm_hold):
                    actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{name}_pos")
                    if actuator_id >= 0:
                        low, high = model.actuator_ctrlrange[actuator_id]
                        data.ctrl[actuator_id] = np.clip(target, low, high)
                mujoco.mj_step(model, data)
        value = side_gripper(
            model,
            data,
            side,
            "close",
            gripper_steps=int(params.get("gripper_steps", 500)),
            direct_qpos=bool(params.get("direct_qpos", False)),
            hold_arm=True,
        )
        post_close_hold_steps = int(params.get("post_close_hold_steps", 0))
        if post_close_hold_steps > 0:
            for _ in range(post_close_hold_steps):
                self._hold_gripper_closed(model, data, side)
                mujoco.mj_step(model, data)
        return TorsoFrameCloseGripperResult(success=True, side=side, gripper_value=float(value), message=f"gripper_value={value:.6f}")

    @staticmethod
    def _hold_gripper_closed(model: mujoco.MjModel, data: mujoco.MjData, side: str) -> None:
        split_actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{side}_gripper_fingers_pos")
        if split_actuator_id >= 0:
            low, high = model.actuator_ctrlrange[split_actuator_id]
            data.ctrl[split_actuator_id] = np.clip(high, low, high)
            return
        for index in (1, 2):
            actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{side}_gripper_finger_joint{index}_pos")
            if actuator_id >= 0:
                low, high = model.actuator_ctrlrange[actuator_id]
                data.ctrl[actuator_id] = np.clip(high, low, high)


def load_skill(path: str | Path | None = None) -> R1ProTorsoFrameCloseGripperSkill:
    return R1ProTorsoFrameCloseGripperSkill()
