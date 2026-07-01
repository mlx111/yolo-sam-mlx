from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from skills.torso_frame.continuous_grasp_executor import (
    R1ProTorsoFrameContinuousGraspExecutor,
    TorsoFrameContinuousGraspResult,
)


class R1ProTorsoFrameGraspSkill:
    """Grasp an object or explicit target using coordinates in a torso reference frame."""

    def __init__(self, torso_frame: str = "torso_link4"):
        self.torso_frame = str(torso_frame)
        self.executor = R1ProTorsoFrameContinuousGraspExecutor()

    def execute_recovery_action(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        params: dict[str, Any],
    ) -> TorsoFrameContinuousGraspResult:
        torso_frame = str(params.get("torso_frame", self.torso_frame))
        use_object_target = bool(params.get("use_object_target", False))
        target_torso = None
        if not use_object_target:
            target_torso = _target_torso(params)
        visual_target = _uses_visual_target(params)

        grasp_offset_torso = np.array(
            [
                float(params.get("grasp_offset_x", 0.0)),
                float(params.get("grasp_offset_y", 0.0)),
                float(params.get("grasp_offset_z", 0.0)),
            ],
            dtype=np.float64,
        )
        if visual_target and bool(params.get("enable_visual_grasp_correction", True)):
            grasp_offset_torso[2] += float(params.get("visual_grasp_offset_z", 0.0))
        for _ in range(max(0, int(params.get("settle_before_steps", 1500)))):
            mujoco.mj_step(model, data)
        return self.executor.execute(
            model,
            data,
            side=str(params.get("side", "left")),
            object_body=str(params.get("object_body", "target_cube")),
            use_object_target=use_object_target,
            target_torso=target_torso,
            grasp_offset_torso=grasp_offset_torso,
            torso_frame=torso_frame,
            pregrasp_offset_torso=_pregrasp_offset_torso(params),
            auto_grasp_table_clearance=bool(params.get("auto_grasp_table_clearance", True)),
            pad_table_clearance=float(params.get("pad_table_clearance", 0.010)),
            lift_height=float(params.get("lift_height", 0.10)),
            waypoint_count=int(params.get("waypoint_count", params.get("waypoints", 20))),
            waypoint_steps=int(params.get("waypoint_steps", 60)),
            solve_iterations=int(params.get("solve_iterations", 120)),
            fail_threshold=float(params.get("fail_threshold", 0.002)),
            orientation_weight=float(params.get("orientation_weight", 0.35)),
            lift_orientation_weight=float(params.get("lift_orientation_weight", 0.0)),
            orientation_threshold=float(params.get("orientation_threshold", 1.0)),
            max_joint_step=float(params.get("max_joint_step", 0.006)),
            topdown_mode=str(params.get("topdown_mode", "palm_down")),
            close_steps=int(params.get("close_steps", params.get("gripper_steps", 500))),
            pre_close_settle_steps=int(params.get("pre_close_settle_steps", 120)),
            close_direct_qpos=bool(params.get("close_direct_qpos", params.get("direct_gripper_qpos", False))),
            control_mode=str(params.get("control_mode", "actuator")),
            control_frame=str(params.get("control_frame", "grasp_tool")),
            max_object_displacement_before_close=float(params.get("max_object_displacement_before_close", 0.015)),
            max_approach_error_before_close=(
                None
                if params.get("max_approach_error_before_close") is None
                else float(params.get("max_approach_error_before_close"))
            ),
            attach_on_close=bool(params.get("attach_on_close", False)),
            settle_steps=int(params.get("settle_steps", 0)),
        )

    @staticmethod
    def compact_result(result: TorsoFrameContinuousGraspResult) -> dict[str, Any]:
        payload = asdict(result)
        return {
            "success": bool(payload["success"]),
            "target_torso": _list(payload["target_torso"]),
            "target_world": _list(payload["target_world"]),
            "grasp_pos_torso": _list(payload["grasp_pos_torso"]),
            "grasp_pos_world": _list(payload["grasp_pos_world"]),
            "tcp_final_torso": _list(payload["tcp_final_torso"]),
            "tcp_final_world": _list(payload["tcp_final_world"]),
            "lift_torso": float(payload["lift_torso"]),
            "lift_world": float(payload["lift_world"]),
            "stage_errors": payload.get("stage_errors", {}),
            "message": payload.get("message", ""),
        }


def _target_torso(params: dict[str, Any]) -> np.ndarray:
    if "position_input_path" in params:
        position_payload = json.loads(Path(params["position_input_path"]).read_text(encoding="utf-8"))
        key = str(params.get("position_key", "median_reference"))
        if key not in position_payload:
            raise KeyError(f"Position key not found in {params['position_input_path']}: {key}")
        return np.asarray(position_payload[key], dtype=np.float64).reshape(3)
    if "target_torso" in params:
        return np.asarray(params["target_torso"], dtype=np.float64).reshape(3)
    if "center_reference" in params:
        return np.asarray(params["center_reference"], dtype=np.float64).reshape(3)
    if "median_reference" in params:
        return np.asarray(params["median_reference"], dtype=np.float64).reshape(3)
    return np.array(
        [
            float(params["target_x"]),
            float(params["target_y"]),
            float(params["target_z"]),
        ],
        dtype=np.float64,
    )


def _uses_visual_target(params: dict[str, Any]) -> bool:
    return any(key in params for key in ("position_input_path", "center_reference", "median_reference"))


def _pregrasp_offset_torso(params: dict[str, Any]) -> np.ndarray:
    if "pregrasp_offset_torso" in params:
        return np.asarray(params["pregrasp_offset_torso"], dtype=np.float64).reshape(3)
    if any(key in params for key in ("pregrasp_offset_x", "pregrasp_offset_y", "pregrasp_offset_z")):
        return np.array(
            [
                float(params.get("pregrasp_offset_x", 0.0)),
                float(params.get("pregrasp_offset_y", 0.0)),
                float(params.get("pregrasp_offset_z", 0.0)),
            ],
            dtype=np.float64,
        )
    raise ValueError("Provide pregrasp_offset_torso or pregrasp_offset_x/pregrasp_offset_y/pregrasp_offset_z")


def _list(value: Any) -> list[float]:
    return [float(v) for v in np.asarray(value, dtype=np.float64).reshape(-1)]


def load_skill(torso_frame: str = "torso_link4") -> R1ProTorsoFrameGraspSkill:
    return R1ProTorsoFrameGraspSkill(torso_frame=torso_frame)
