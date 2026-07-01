from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from skills.torso_frame._common import current_tcp_torso, local_to_world, torso_frame_pose
from skills.torso_frame.continuous_grasp_executor import R1ProTorsoFrameContinuousGraspExecutor


@dataclass(frozen=True)
class TorsoFrameLowerHeldObjectResult:
    success: bool
    side: str
    torso_frame: str
    lower_distance: float
    start_torso: list[float]
    target_torso: list[float]
    start_world: list[float]
    target_world: list[float]
    final_error: float
    stage_errors: dict[str, float]
    stage_orientation_errors: dict[str, float]
    message: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class R1ProTorsoFrameLowerHeldObjectSkill:
    """Lower the currently held object by moving the grasp TCP along torso -Z."""

    def execute_recovery_action(self, model: mujoco.MjModel, data: mujoco.MjData, params: dict[str, Any]) -> TorsoFrameLowerHeldObjectResult:
        side = str(params.get("side", "left"))
        torso_frame = str(params.get("torso_frame", "torso_link4"))
        lower_distance = self._lower_distance(params)

        mujoco.mj_forward(model, data)
        start_torso = current_tcp_torso(model, data, side, torso_frame)
        target_torso = start_torso + np.array([0.0, 0.0, -lower_distance], dtype=np.float64)
        start_world = self._to_world(model, data, torso_frame, start_torso)
        target_world = self._to_world(model, data, torso_frame, target_torso)

        executor = R1ProTorsoFrameContinuousGraspExecutor()
        target_xmat_world = executor._tcp_xmat_world(model, data, side, str(params.get("control_frame", "grasp_tool")))
        stage_errors: dict[str, float] = {}
        stage_orientation_errors: dict[str, float] = {}
        ok = executor._vertical_cartesian_lift_torso(
            model,
            data,
            side,
            f"{side}_lower_held_object",
            start_torso,
            target_xmat_world,
            torso_frame,
            stage_errors,
            stage_orientation_errors,
            lift_height=-lower_distance,
            segment_height=float(params.get("segment_height", 0.01)),
            segment_steps=int(params.get("segment_steps", max(int(params.get("waypoint_steps", 60)), 80))),
            fail_threshold=float(params.get("fail_threshold", 0.008)),
            orientation_weight=float(params.get("orientation_weight", 0.35)),
            orientation_threshold=float(params.get("orientation_threshold", 1.0)),
            max_joint_step=float(params.get("max_joint_step", 0.003)),
            step_callback=None,
            control_frame=str(params.get("control_frame", "grasp_tool")),
            hold_gripper_side=side,
        )

        final_error = float(stage_errors.get(f"{side}_lower_held_object", float("inf")))
        success_threshold = float(params.get("success_threshold", params.get("lower_success_threshold", params.get("fail_threshold", 0.008))))
        success = bool(ok or (np.isfinite(final_error) and final_error <= success_threshold))
        return TorsoFrameLowerHeldObjectResult(
            success=success,
            side=side,
            torso_frame=torso_frame,
            lower_distance=lower_distance,
            start_torso=np.round(start_torso, 6).tolist(),
            target_torso=np.round(target_torso, 6).tolist(),
            start_world=np.round(start_world, 6).tolist(),
            target_world=np.round(target_world, 6).tolist(),
            final_error=final_error,
            stage_errors=stage_errors,
            stage_orientation_errors=stage_orientation_errors,
            message=f"lower_distance={lower_distance:.6f}",
        )

    @staticmethod
    def _lower_distance(params: dict[str, Any]) -> float:
        value = float(params.get("lower_distance", params.get("lower_dz", 0.04)))
        if value < 0.0 or value > 0.08:
            raise ValueError(f"lower_distance must be in [0.0, 0.08] m: {value}")
        return value

    @staticmethod
    def _to_world(model: mujoco.MjModel, data: mujoco.MjData, torso_frame: str, target_torso: np.ndarray) -> np.ndarray:
        frame_pos, frame_xmat = torso_frame_pose(model, data, torso_frame)
        return local_to_world(frame_pos, frame_xmat, target_torso)


def load_skill(path: str | Path | None = None) -> R1ProTorsoFrameLowerHeldObjectSkill:
    return R1ProTorsoFrameLowerHeldObjectSkill()
