from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import mujoco
import numpy as np

from skills.base.arm_ik_skill import ArmMotionResult, ArmSide, R1ProArmIKSkill


@dataclass(frozen=True)
class ArmMoveSkillConfig:
    name: str
    side: ArmSide
    urdf_path: str
    default_steps: int
    default_settle_steps: int
    default_max_joint_step: float
    default_fail_threshold: float
    default_direct_qpos: bool
    default_pose_segment_count: int
    default_pose_posture_gain: float


class R1ProArmMoveSkill:
    """Fixed-side arm TCP move skill backed by Pinocchio IK and MuJoCo actuators."""

    def __init__(self, config: ArmMoveSkillConfig):
        self.config = config
        self.ik_skill = R1ProArmIKSkill(config.urdf_path)

    @classmethod
    def from_json(cls, path: str | Path) -> "R1ProArmMoveSkill":
        config_path = Path(path)
        payload = json.loads(config_path.read_text())
        control_defaults = payload.get("control_defaults", {})
        side = payload["side"]
        if side not in ("left", "right"):
            raise ValueError(f"Unsupported arm side in {config_path}: {side!r}")
        return cls(
            ArmMoveSkillConfig(
                name=payload["name"],
                side=side,
                urdf_path=payload.get("urdf_path", "urdf/r1_pro_with_gripper.urdf"),
                default_steps=int(control_defaults.get("steps", 1500)),
                default_settle_steps=int(control_defaults.get("settle_steps", 3000)),
                default_max_joint_step=float(control_defaults.get("max_joint_step", 0.006)),
                default_fail_threshold=float(control_defaults.get("fail_threshold", 0.02)),
                default_direct_qpos=bool(control_defaults.get("direct_qpos", False)),
                default_pose_segment_count=max(int(control_defaults.get("pose_segment_count", 4)), 1),
                default_pose_posture_gain=float(control_defaults.get("pose_posture_gain", 0.12)),
            )
        )

    def move_to_position(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        target_pos: np.ndarray | list[float] | tuple[float, float, float],
        *,
        control_frame: str = "hand_tcp",
        steps: int | None = None,
        settle_steps: int | None = None,
        direct_qpos: bool | None = None,
        stabilize: bool = True,
        lock_posture: bool = True,
        max_joint_step: float | None = None,
        fail_threshold: float | None = None,
        step_callback: Callable[[], None] | None = None,
    ) -> ArmMotionResult:
        return self.move_to_pose(
            model,
            data,
            target_pos,
            control_frame=control_frame,
            steps=steps,
            settle_steps=settle_steps,
            direct_qpos=direct_qpos,
            stabilize=stabilize,
            lock_posture=lock_posture,
            max_joint_step=max_joint_step,
            fail_threshold=fail_threshold,
            step_callback=step_callback,
        )

    def move_to_pose(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        target_pos: np.ndarray | list[float] | tuple[float, float, float],
        *,
        target_quat_wxyz: np.ndarray | list[float] | tuple[float, float, float, float] | None = None,
        steps: int | None = None,
        settle_steps: int | None = None,
        direct_qpos: bool | None = None,
        stabilize: bool = True,
        lock_posture: bool = True,
        max_joint_step: float | None = None,
        fail_threshold: float | None = None,
        orientation_threshold: float = 0.15,
        orientation_weight: float | None = None,
        control_frame: str = "grasp_tool",
        pose_segment_count: int | None = None,
        pose_posture_gain: float | None = None,
        step_callback: Callable[[], None] | None = None,
    ) -> ArmMotionResult:
        return self.ik_skill.move_to_position(
            model,
            data,
            self.config.side,
            np.asarray(target_pos, dtype=np.float64),
            steps=self.config.default_steps if steps is None else steps,
            settle_steps=self.config.default_settle_steps if settle_steps is None else settle_steps,
            direct_qpos=self.config.default_direct_qpos if direct_qpos is None else bool(direct_qpos),
            stabilize=stabilize,
            closed_loop=False,
            cartesian_closed_loop=True,
            lock_posture=lock_posture,
            max_joint_step=self.config.default_max_joint_step if max_joint_step is None else max_joint_step,
            fail_threshold=self.config.default_fail_threshold if fail_threshold is None else fail_threshold,
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
        return self.move_to_pose(
            model,
            data,
            target_pos,
            target_quat_wxyz=params.get("target_quat_wxyz"),
            steps=int(params.get("steps", self.config.default_steps)),
            settle_steps=int(params.get("settle_steps", self.config.default_settle_steps)),
            direct_qpos=bool(params.get("direct_qpos", self.config.default_direct_qpos if direct_qpos is None else direct_qpos)),
            stabilize=bool(params.get("stabilize", True)),
            lock_posture=bool(params.get("lock_posture", True)),
            max_joint_step=float(params.get("max_joint_step", self.config.default_max_joint_step)),
            fail_threshold=float(params.get("fail_threshold", self.config.default_fail_threshold)),
            orientation_threshold=float(params.get("orientation_threshold", 0.15)),
            orientation_weight=(float(params["orientation_weight"]) if "orientation_weight" in params else None),
            control_frame=str(params.get("control_frame", "grasp_tool")),
            pose_segment_count=int(params.get("pose_segment_count", self.config.default_pose_segment_count)),
            pose_posture_gain=float(params.get("pose_posture_gain", self.config.default_pose_posture_gain)),
        )


def load_skill(path: str | Path) -> R1ProArmMoveSkill:
    return R1ProArmMoveSkill.from_json(path)


def _quat_wxyz_to_xmat(
    target_quat_wxyz: np.ndarray | list[float] | tuple[float, float, float, float] | None,
) -> np.ndarray | None:
    if target_quat_wxyz is not None:
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
    return None
