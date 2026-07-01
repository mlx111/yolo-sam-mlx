from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from skills.torso_frame._common import current_tcp_torso, follow_line_torso, grasp_target_with_correction, local_to_world, resolve_torso_target, torso_frame_pose


@dataclass(frozen=True)
class TorsoFrameApproachObjectResult:
    success: bool
    side: str
    torso_frame: str
    target_torso: list[float]
    target_world: list[float]
    grasp_torso: list[float]
    grasp_world: list[float]
    final_error: float
    stage_errors: dict[str, float]
    stage_orientation_errors: dict[str, float]
    message: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class R1ProTorsoFrameApproachObjectSkill:
    def execute_recovery_action(self, model: mujoco.MjModel, data: mujoco.MjData, params: dict[str, Any]) -> TorsoFrameApproachObjectResult:
        side = str(params.get("side", "left"))
        torso_frame = str(params.get("torso_frame", "torso_link4"))
        target_torso = resolve_torso_target(params)
        grasp_torso = grasp_target_with_correction(params)
        target_world = self._to_world(model, data, torso_frame, target_torso)
        grasp_world = self._to_world(model, data, torso_frame, grasp_torso)
        start_torso = current_tcp_torso(model, data, side, torso_frame)
        ok, stage_errors, stage_orientation_errors = follow_line_torso(
            model,
            data,
            side,
            "approach_object",
            start_torso,
            grasp_torso,
            torso_frame,
            params,
        )
        final_error = float(stage_errors.get("approach_object", float("inf")))
        success_threshold = float(params.get("success_threshold", params.get("approach_success_threshold", params.get("fail_threshold", 0.012))))
        return TorsoFrameApproachObjectResult(
            success=bool(np.isfinite(final_error) and final_error <= success_threshold),
            side=side,
            torso_frame=torso_frame,
            target_torso=np.round(target_torso, 6).tolist(),
            target_world=np.round(target_world, 6).tolist(),
            grasp_torso=np.round(grasp_torso, 6).tolist(),
            grasp_world=np.round(grasp_world, 6).tolist(),
            final_error=final_error,
            stage_errors=stage_errors,
            stage_orientation_errors=stage_orientation_errors,
            message=f"grasp_torso={np.round(grasp_torso, 6).tolist()}",
        )

    @staticmethod
    def _to_world(model: mujoco.MjModel, data: mujoco.MjData, torso_frame: str, target_torso: np.ndarray) -> np.ndarray:
        frame_pos, frame_xmat = torso_frame_pose(model, data, torso_frame)
        return local_to_world(frame_pos, frame_xmat, target_torso)


def load_skill(path: str | Path | None = None) -> R1ProTorsoFrameApproachObjectSkill:
    return R1ProTorsoFrameApproachObjectSkill()
