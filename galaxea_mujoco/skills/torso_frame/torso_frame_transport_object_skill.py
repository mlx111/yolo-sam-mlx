from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from skills.torso_frame._common import current_tcp_torso, follow_line_torso, local_to_world, target_position_path, torso_frame_pose


@dataclass(frozen=True)
class TorsoFrameTransportObjectResult:
    success: bool
    side: str
    torso_frame: str
    start_torso: list[float]
    target_source_torso: list[float] | None
    target_z_mode: str
    place_offset_torso: list[float]
    target_torso: list[float]
    start_world: list[float]
    target_world: list[float]
    final_error: float
    object_body: str
    object_start_world: list[float] | None
    object_final_world: list[float] | None
    object_displacement_world: list[float] | None
    tcp_displacement_world: list[float]
    object_follow_error: float | None
    object_follow_success: bool | None
    stage_errors: dict[str, float]
    stage_orientation_errors: dict[str, float]
    message: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class R1ProTorsoFrameTransportObjectSkill:
    """Move the currently grasping TCP in torso coordinates while keeping the gripper closed."""

    def execute_recovery_action(self, model: mujoco.MjModel, data: mujoco.MjData, params: dict[str, Any]) -> TorsoFrameTransportObjectResult:
        side = str(params.get("side", "left"))
        torso_frame = str(params.get("torso_frame", "torso_link4"))
        object_body = str(params.get("object_body", "target_cube"))
        object_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, object_body)

        mujoco.mj_forward(model, data)
        object_start_world = data.xpos[object_id].copy() if object_id >= 0 else None
        start_torso = current_tcp_torso(model, data, side, torso_frame)
        target_torso, target_source_torso, target_z_mode, place_offset_torso = self._target_torso(start_torso, params)
        start_world = self._to_world(model, data, torso_frame, start_torso)
        target_world = self._to_world(model, data, torso_frame, target_torso)

        motion_params = dict(params)
        motion_params.setdefault("topdown_mode", "current")
        motion_params.setdefault("waypoint_steps", 120)
        motion_params.setdefault("max_joint_step", 0.003)
        motion_params.setdefault("fail_threshold", 0.01)
        ok, stage_errors, stage_orientation_errors = follow_line_torso(
            model,
            data,
            side,
            "transport_object",
            start_torso,
            target_torso,
            torso_frame,
            motion_params,
            hold_gripper_side=side,
        )

        final_error = float(stage_errors.get("transport_object", float("inf")))
        success_threshold = float(params.get("success_threshold", params.get("transport_success_threshold", params.get("fail_threshold", 0.01))))
        tcp_displacement_world = target_world - start_world

        object_final_world = None
        object_displacement_world = None
        object_follow_error = None
        object_follow_success = None
        if object_id >= 0 and object_start_world is not None:
            mujoco.mj_forward(model, data)
            object_final_world = data.xpos[object_id].copy()
            object_displacement_world = object_final_world - object_start_world
            follow_axes = np.array(
                [
                    bool(params.get("check_follow_x", True)),
                    bool(params.get("check_follow_y", True)),
                    bool(params.get("check_follow_z", False)),
                ],
                dtype=bool,
            )
            if np.any(follow_axes):
                object_follow_error = float(np.linalg.norm((object_displacement_world - tcp_displacement_world)[follow_axes]))
                object_follow_success = object_follow_error <= float(params.get("object_follow_threshold", 0.025))

        tcp_success = bool(ok or (np.isfinite(final_error) and final_error <= success_threshold))
        success = bool(tcp_success and (object_follow_success is not False))
        return TorsoFrameTransportObjectResult(
            success=success,
            side=side,
            torso_frame=torso_frame,
            start_torso=np.round(start_torso, 6).tolist(),
            target_source_torso=np.round(target_source_torso, 6).tolist() if target_source_torso is not None else None,
            target_z_mode=target_z_mode,
            place_offset_torso=np.round(place_offset_torso, 6).tolist(),
            target_torso=np.round(target_torso, 6).tolist(),
            start_world=np.round(start_world, 6).tolist(),
            target_world=np.round(target_world, 6).tolist(),
            final_error=final_error,
            object_body=object_body,
            object_start_world=np.round(object_start_world, 6).tolist() if object_start_world is not None else None,
            object_final_world=np.round(object_final_world, 6).tolist() if object_final_world is not None else None,
            object_displacement_world=np.round(object_displacement_world, 6).tolist() if object_displacement_world is not None else None,
            tcp_displacement_world=np.round(tcp_displacement_world, 6).tolist(),
            object_follow_error=object_follow_error,
            object_follow_success=object_follow_success,
            stage_errors=stage_errors,
            stage_orientation_errors=stage_orientation_errors,
            message=(
                f"object_follow_error={object_follow_error:.6f}"
                if object_follow_error is not None
                else f"transport_torso={np.round(target_torso, 6).tolist()}"
            ),
        )

    @staticmethod
    def _target_torso(start_torso: np.ndarray, params: dict[str, Any]) -> tuple[np.ndarray, np.ndarray | None, str, np.ndarray]:
        place_offset_torso = np.array(
            [
                float(params.get("place_offset_x", 0.0)),
                float(params.get("place_offset_y", 0.0)),
                float(params.get("place_offset_z", 0.0)),
            ],
            dtype=np.float64,
        )
        if "position_input_path" in params:
            target_source_torso = R1ProTorsoFrameTransportObjectSkill._position_target_torso(params)
            target_torso = target_source_torso + place_offset_torso
            target_z_mode = str(params.get("target_z_mode", "keep_current"))
            if bool(params.get("use_position_xy_only", True)) or target_z_mode == "keep_current":
                target_torso[2] = float(np.asarray(start_torso, dtype=np.float64).reshape(3)[2])
                target_z_mode = "keep_current"
            elif target_z_mode != "from_position":
                raise ValueError(f"Unsupported target_z_mode for position_input_path: {target_z_mode!r}")
            return target_torso, target_source_torso, target_z_mode, place_offset_torso
        if "target_class" in params:
            target_source_torso = R1ProTorsoFrameTransportObjectSkill._position_target_torso({
                **params,
                "position_input_path": str(target_position_path(str(params["target_class"]), params.get("_runtime_tmp_dir"))),
            })
            target_torso = target_source_torso + place_offset_torso
            target_torso[2] = float(np.asarray(start_torso, dtype=np.float64).reshape(3)[2])
            return target_torso, target_source_torso, "keep_current", place_offset_torso
        if "target_torso" in params:
            return np.asarray(params["target_torso"], dtype=np.float64).reshape(3), None, "explicit", place_offset_torso
        if all(key in params for key in ("target_x", "target_y", "target_z")):
            return np.array([float(params["target_x"]), float(params["target_y"]), float(params["target_z"])], dtype=np.float64), None, "explicit", place_offset_torso
        return R1ProTorsoFrameTransportObjectSkill._delta_target_torso(start_torso, params), None, "delta", place_offset_torso

    @staticmethod
    def _position_target_torso(params: dict[str, Any]) -> np.ndarray:
        path = Path(str(params["position_input_path"]))
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("success") is False:
            raise ValueError(f"Position input reports failure: {path}")
        key = str(params.get("position_key", "median_reference"))
        if key not in payload or payload[key] is None:
            raise KeyError(f"Position key not found in {path}: {key}")
        return np.asarray(payload[key], dtype=np.float64).reshape(3)

    @staticmethod
    def _delta_target_torso(start_torso: np.ndarray, params: dict[str, Any]) -> np.ndarray:
        delta = np.array(
            [
                float(params.get("delta_x", params.get("transport_dx", 0.0))),
                float(params.get("delta_y", params.get("transport_dy", 0.0))),
                float(params.get("delta_z", params.get("transport_dz", 0.0))),
            ],
            dtype=np.float64,
        )
        return np.asarray(start_torso, dtype=np.float64).reshape(3) + delta

    @staticmethod
    def _to_world(model: mujoco.MjModel, data: mujoco.MjData, torso_frame: str, target_torso: np.ndarray) -> np.ndarray:
        frame_pos, frame_xmat = torso_frame_pose(model, data, torso_frame)
        return local_to_world(frame_pos, frame_xmat, target_torso)


def load_skill(path: str | Path | None = None) -> R1ProTorsoFrameTransportObjectSkill:
    return R1ProTorsoFrameTransportObjectSkill()
