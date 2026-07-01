from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from skills.torso_frame._common import local_to_world, target_position_path, torso_frame_pose, vertical_lift_from_current_tcp


@dataclass(frozen=True)
class TorsoFrameLiftResult:
    success: bool
    side: str
    torso_frame: str
    start_torso: list[float]
    target_torso: list[float]
    start_world: list[float]
    target_world: list[float]
    final_error: float
    object_body: str
    object_start_world: list[float] | None
    object_final_world: list[float] | None
    object_lift_world: float | None
    object_lift_success: bool | None
    min_object_lift: float | None
    stage_errors: dict[str, float]
    stage_orientation_errors: dict[str, float]
    message: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class R1ProTorsoFrameLiftSkill:
    def execute_recovery_action(self, model: mujoco.MjModel, data: mujoco.MjData, params: dict[str, Any]) -> TorsoFrameLiftResult:
        side = str(params.get("side", "left"))
        torso_frame = str(params.get("torso_frame", "torso_link4"))
        lift_height = float(params.get("lift_height", params.get("lift_dz", 0.10)))
        object_body = self._resolve_object_body(params)
        object_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, object_body)
        object_start_world = None
        if object_id >= 0:
            mujoco.mj_forward(model, data)
            object_start_world = data.xpos[object_id].copy()
        ok, start_torso, target_torso, stage_errors, stage_orientation_errors = vertical_lift_from_current_tcp(
            model,
            data,
            side,
            torso_frame,
            params,
        )
        start_world = self._to_world(model, data, torso_frame, start_torso)
        target_world = self._to_world(model, data, torso_frame, target_torso)
        final_error = float(stage_errors.get(f"{side}_vertical_lift", float("inf")))
        object_final_world = None
        object_lift_world = None
        if object_id >= 0 and object_start_world is not None:
            mujoco.mj_forward(model, data)
            object_final_world = data.xpos[object_id].copy()
            object_lift_world = float(object_final_world[2] - object_start_world[2])
        min_object_lift = float(params.get("min_object_lift", max(0.0, lift_height - 0.005)))
        tcp_success_threshold = float(params.get("success_threshold", params.get("lift_success_threshold", 0.008)))
        object_lift_success = object_lift_world is not None and object_lift_world >= min_object_lift
        success = np.isfinite(final_error) and final_error <= tcp_success_threshold
        return TorsoFrameLiftResult(
            success=bool(success),
            side=side,
            torso_frame=torso_frame,
            start_torso=np.round(start_torso, 6).tolist(),
            target_torso=np.round(target_torso, 6).tolist(),
            start_world=np.round(start_world, 6).tolist(),
            target_world=np.round(target_world, 6).tolist(),
            final_error=final_error,
            object_body=object_body,
            object_start_world=np.round(object_start_world, 6).tolist() if object_start_world is not None else None,
            object_final_world=np.round(object_final_world, 6).tolist() if object_final_world is not None else None,
            object_lift_world=object_lift_world,
            object_lift_success=bool(object_lift_success) if object_lift_world is not None else None,
            min_object_lift=min_object_lift,
            stage_errors=stage_errors,
            stage_orientation_errors=stage_orientation_errors,
            message=f"object_lift_world={object_lift_world:.6f}" if object_lift_world is not None else f"lift_torso={float(lift_height):.6f}",
        )

    @staticmethod
    def _to_world(model: mujoco.MjModel, data: mujoco.MjData, torso_frame: str, target_torso: np.ndarray) -> np.ndarray:
        frame_pos, frame_xmat = torso_frame_pose(model, data, torso_frame)
        return local_to_world(frame_pos, frame_xmat, target_torso)

    @staticmethod
    def _resolve_object_body(params: dict[str, Any]) -> str:
        if params.get("object_body"):
            return str(params["object_body"])
        if params.get("position_input_path"):
            payload = json.loads(Path(params["position_input_path"]).read_text(encoding="utf-8"))
            if payload.get("object_body"):
                return str(payload["object_body"])
        if params.get("target_class"):
            path = target_position_path(str(params["target_class"]), params.get("_runtime_tmp_dir"))
            if path.exists():
                payload = json.loads(path.read_text(encoding="utf-8"))
                if payload.get("object_body"):
                    return str(payload["object_body"])
        return "target_cube"


def load_skill(path: str | Path | None = None) -> R1ProTorsoFrameLiftSkill:
    return R1ProTorsoFrameLiftSkill()
