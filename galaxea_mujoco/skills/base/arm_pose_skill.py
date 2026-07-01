from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import mujoco
import numpy as np

from skills.base.arm_ik_skill import ArmMotionResult, ArmSide, R1ProArmIKSkill


@dataclass(frozen=True)
class ArmPoseSkillConfig:
    name: str
    side: ArmSide
    urdf_path: str
    default_steps: int
    default_settle_steps: int
    default_max_joint_step: float
    default_fail_threshold: float
    default_direct_qpos: bool
    default_orientation_threshold: float
    default_orientation_weight: float


class R1ProArmPoseSkill:
    """Fixed-side 6D arm pose skill backed by Pinocchio IK and MuJoCo actuators."""

    def __init__(self, config: ArmPoseSkillConfig):
        self.config = config
        self.ik_skill = R1ProArmIKSkill(config.urdf_path)

    @classmethod
    def from_json(cls, path: str | Path) -> "R1ProArmPoseSkill":
        config_path = Path(path)
        payload = json.loads(config_path.read_text())
        control_defaults = payload.get("control_defaults", {})
        side = payload["side"]
        if side not in ("left", "right"):
            raise ValueError(f"Unsupported arm side in {config_path}: {side!r}")
        return cls(
            ArmPoseSkillConfig(
                name=payload["name"],
                side=side,
                urdf_path=payload.get("urdf_path", "urdf/r1_pro_with_gripper.urdf"),
                default_steps=int(control_defaults.get("steps", 1500)),
                default_settle_steps=int(control_defaults.get("settle_steps", 3000)),
                default_max_joint_step=float(control_defaults.get("max_joint_step", 0.006)),
                default_fail_threshold=float(control_defaults.get("fail_threshold", 0.02)),
                default_direct_qpos=bool(control_defaults.get("direct_qpos", False)),
                default_orientation_threshold=float(control_defaults.get("orientation_threshold", 0.15)),
                default_orientation_weight=float(control_defaults.get("orientation_weight", 0.35)),
            )
        )

    def move_to_pose(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        target_pos: np.ndarray | list[float] | tuple[float, float, float],
        *,
        target_quat_wxyz: np.ndarray | list[float] | tuple[float, float, float, float],
        steps: int | None = None,
        settle_steps: int | None = None,
        direct_qpos: bool | None = None,
        stabilize: bool = True,
        lock_posture: bool = True,
        max_joint_step: float | None = None,
        fail_threshold: float | None = None,
        orientation_threshold: float | None = None,
        orientation_weight: float | None = None,
        step_callback: Callable[[], None] | None = None,
    ) -> ArmMotionResult:
        return self.ik_skill.move_to_position(
            model,
            data,
            self.config.side,
            np.asarray(target_pos, dtype=np.float64),
            target_xmat=_quat_wxyz_to_xmat(target_quat_wxyz),
            steps=self.config.default_steps if steps is None else steps,
            settle_steps=self.config.default_settle_steps if settle_steps is None else settle_steps,
            direct_qpos=self.config.default_direct_qpos if direct_qpos is None else bool(direct_qpos),
            stabilize=stabilize,
            closed_loop=True,
            cartesian_closed_loop=False,
            lock_posture=lock_posture,
            max_joint_step=self.config.default_max_joint_step if max_joint_step is None else max_joint_step,
            fail_threshold=self.config.default_fail_threshold if fail_threshold is None else fail_threshold,
            orientation_threshold=(
                self.config.default_orientation_threshold
                if orientation_threshold is None
                else float(orientation_threshold)
            ),
            orientation_weight=(
                self.config.default_orientation_weight
                if orientation_weight is None
                else float(orientation_weight)
            ),
            step_callback=step_callback,
        )

    def execute_recovery_action(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        params: dict,
        *,
        direct_qpos: bool | None = None,
    ) -> ArmMotionResult:
        target_pos = np.array([params["target_x"], params["target_y"], params["target_z"]], dtype=np.float64)
        target_quat_wxyz = params.get("target_quat_wxyz")
        if target_quat_wxyz is None:
            raise ValueError("Provide target_quat_wxyz for pose control")
        return self.move_to_pose(
            model,
            data,
            target_pos,
            target_quat_wxyz=target_quat_wxyz,
            steps=int(params.get("steps", self.config.default_steps)),
            settle_steps=int(params.get("settle_steps", self.config.default_settle_steps)),
            direct_qpos=bool(
                params.get("direct_qpos", self.config.default_direct_qpos if direct_qpos is None else direct_qpos)
            ),
            stabilize=bool(params.get("stabilize", True)),
            lock_posture=bool(params.get("lock_posture", True)),
            max_joint_step=float(params.get("max_joint_step", self.config.default_max_joint_step)),
            fail_threshold=float(params.get("fail_threshold", self.config.default_fail_threshold)),
            orientation_threshold=float(params.get("orientation_threshold", self.config.default_orientation_threshold)),
            orientation_weight=float(params.get("orientation_weight", self.config.default_orientation_weight)),
        )


def load_skill(path: str | Path) -> R1ProArmPoseSkill:
    return R1ProArmPoseSkill.from_json(path)


def _quat_wxyz_to_xmat(
    target_quat_wxyz: np.ndarray | list[float] | tuple[float, float, float, float],
) -> np.ndarray:
    quat = np.asarray(target_quat_wxyz, dtype=np.float64).reshape(4)
    norm = float(np.linalg.norm(quat))
    if norm < 1e-9:
        raise ValueError("target_quat_wxyz must be non-zero")
    w, x, y, z = quat / norm
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )
