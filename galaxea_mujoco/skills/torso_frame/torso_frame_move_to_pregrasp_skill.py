from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from skills.torso_frame._common import follow_line_torso, grasp_target_with_correction, local_to_world, resolve_torso_target, torso_frame_pose, current_tcp_torso
from skills.torso_frame.continuous_grasp_executor import R1ProTorsoFrameContinuousGraspExecutor

DEFAULT_PREGRASP_OFFSET_TORSO = np.array([0.0, 0.0, 0.08], dtype=np.float64)


@dataclass(frozen=True)
class TorsoFrameMoveToPregraspResult:
    success: bool
    side: str
    torso_frame: str
    target_torso: list[float]
    target_world: list[float]
    pregrasp_torso: list[float]
    pregrasp_world: list[float]
    final_tcp_torso: list[float]
    final_tcp_world: list[float]
    final_tcp_minus_pregrasp_torso: list[float]
    final_tcp_minus_pregrasp_world: list[float]
    final_tcp_pregrasp_error_norm: float
    final_error: float
    stage_errors: dict[str, float]
    stage_orientation_errors: dict[str, float]
    message: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class R1ProTorsoFrameMoveToPregraspSkill:
    def execute_recovery_action(self, model: mujoco.MjModel, data: mujoco.MjData, params: dict[str, Any]) -> TorsoFrameMoveToPregraspResult:
        side = str(params.get("side", "left"))
        torso_frame = str(params.get("torso_frame", "torso_link4"))
        target_torso = resolve_torso_target(params)
        control_frame = str(params.get("control_frame", "grasp_tool"))
        target_world = self._to_world(model, data, torso_frame, target_torso)
        trajectory_input_path = str(params.get("trajectory_input_path", self._default_trajectory_path(side, params)))
        if Path(trajectory_input_path).exists():
            pregrasp_torso = self._planned_pregrasp_torso(trajectory_input_path)
        else:
            grasp_torso = grasp_target_with_correction(params)
            pregrasp_torso = self._pregrasp_torso(grasp_torso, params)
        pregrasp_world = self._to_world(model, data, torso_frame, pregrasp_torso)
        start_torso = current_tcp_torso(model, data, side, torso_frame)
        if Path(trajectory_input_path).exists():
            ok, stage_errors, stage_orientation_errors = self._follow_planned_trajectory(
                model,
                data,
                side,
                torso_frame,
                trajectory_input_path,
                params,
            )
        else:
            ok, stage_errors, stage_orientation_errors = follow_line_torso(
                model,
                data,
                side,
                "move_to_pregrasp",
                start_torso,
                pregrasp_torso,
                torso_frame,
                params,
            )
        final_error = float(stage_errors.get("move_to_pregrasp", float("inf")))
        final_tcp_torso = current_tcp_torso(model, data, side, torso_frame)
        final_tcp_world = self._to_world(model, data, torso_frame, final_tcp_torso)
        final_tcp_minus_pregrasp_torso = final_tcp_torso - pregrasp_torso
        final_tcp_minus_pregrasp_world = final_tcp_world - pregrasp_world
        success_threshold = float(params.get("success_threshold", params.get("pregrasp_success_threshold", 0.01)))
        return TorsoFrameMoveToPregraspResult(
            success=bool(np.isfinite(final_error) and final_error <= success_threshold),
            side=side,
            torso_frame=torso_frame,
            target_torso=np.round(target_torso, 6).tolist(),
            target_world=np.round(target_world, 6).tolist(),
            pregrasp_torso=np.round(pregrasp_torso, 6).tolist(),
            pregrasp_world=np.round(pregrasp_world, 6).tolist(),
            final_tcp_torso=np.round(final_tcp_torso, 6).tolist(),
            final_tcp_world=np.round(final_tcp_world, 6).tolist(),
            final_tcp_minus_pregrasp_torso=np.round(final_tcp_minus_pregrasp_torso, 6).tolist(),
            final_tcp_minus_pregrasp_world=np.round(final_tcp_minus_pregrasp_world, 6).tolist(),
            final_tcp_pregrasp_error_norm=float(np.linalg.norm(final_tcp_minus_pregrasp_torso)),
            final_error=final_error,
            stage_errors=stage_errors,
            stage_orientation_errors=stage_orientation_errors,
            message=f"pregrasp_torso={np.round(pregrasp_torso, 6).tolist()}",
        )

    @staticmethod
    def _to_world(model: mujoco.MjModel, data: mujoco.MjData, torso_frame: str, target_torso: np.ndarray) -> np.ndarray:
        frame_pos, frame_xmat = torso_frame_pose(model, data, torso_frame)
        return local_to_world(frame_pos, frame_xmat, target_torso)

    @staticmethod
    def _pregrasp_torso(grasp_torso: np.ndarray, params: dict[str, Any]) -> np.ndarray:
        if "pregrasp_offset_torso" in params:
            offset = np.asarray(params["pregrasp_offset_torso"], dtype=np.float64).reshape(3)
        elif any(key in params for key in ("pregrasp_offset_x", "pregrasp_offset_y", "pregrasp_offset_z")):
            offset = np.array(
                [
                    float(params.get("pregrasp_offset_x", 0.0)),
                    float(params.get("pregrasp_offset_y", 0.0)),
                    float(params.get("pregrasp_offset_z", 0.0)),
                ],
                dtype=np.float64,
            )
        else:
            offset = DEFAULT_PREGRASP_OFFSET_TORSO.copy()
        return np.asarray(grasp_torso, dtype=np.float64).reshape(3) + offset

    @staticmethod
    def _planned_pregrasp_torso(trajectory_input_path: str) -> np.ndarray:
        payload = json.loads(Path(trajectory_input_path).read_text(encoding="utf-8"))
        if "pregrasp_torso" in payload:
            return np.asarray(payload["pregrasp_torso"], dtype=np.float64).reshape(3)
        waypoints = payload.get("waypoints")
        if isinstance(waypoints, list) and waypoints:
            last = waypoints[-1]
            if isinstance(last, dict) and "position_torso" in last:
                return np.asarray(last["position_torso"], dtype=np.float64).reshape(3)
        raise ValueError(f"trajectory has no pregrasp_torso or final waypoint: {trajectory_input_path}")

    @staticmethod
    def _default_trajectory_path(side: str, params: dict[str, Any]) -> Path:
        if params.get("_runtime_tmp_dir"):
            return Path(str(params["_runtime_tmp_dir"])) / "trajectories" / f"{side}_pregrasp_trajectory.json"
        return Path("output") / f"{side}_pregrasp_trajectory.json"

    @staticmethod
    def _follow_planned_trajectory(
        model: mujoco.MjModel,
        data: mujoco.MjData,
        side: str,
        torso_frame: str,
        trajectory_input_path: str,
        params: dict[str, Any],
    ) -> tuple[bool, dict[str, float], dict[str, float]]:
        payload = json.loads(Path(trajectory_input_path).read_text(encoding="utf-8"))
        waypoints = payload.get("waypoints")
        if not isinstance(waypoints, list) or not waypoints:
            raise ValueError(f"trajectory has no waypoints: {trajectory_input_path}")
        executor = R1ProTorsoFrameContinuousGraspExecutor()
        stage_errors: dict[str, float] = {}
        stage_orientation_errors: dict[str, float] = {}
        start_torso = current_tcp_torso(model, data, side, torso_frame)
        ok = True
        current = start_torso
        final_error = float("inf")
        final_orientation_error = float("inf")
        for index, waypoint in enumerate(waypoints):
            if not isinstance(waypoint, dict):
                continue
            target = np.asarray(waypoint["position_torso"], dtype=np.float64).reshape(3)
            if float(np.linalg.norm(target - current)) < float(params.get("trajectory_skip_distance", 0.003)):
                continue
            xmat_world = np.asarray(waypoint.get("xmat_world"), dtype=np.float64).reshape(3, 3)
            segment_errors: dict[str, float] = {}
            segment_orientation_errors: dict[str, float] = {}
            segment_ok = executor._follow_line_torso(
                model,
                data,
                side,
                f"move_to_pregrasp_{index:03d}",
                current,
                target,
                xmat_world,
                torso_frame,
                segment_errors,
                segment_orientation_errors,
                waypoint_count=int(params.get("trajectory_segment_waypoint_count", params.get("waypoint_count", 2))),
                waypoint_steps=int(params.get("waypoint_steps", 60)),
                solve_iterations=int(params.get("solve_iterations", 120)),
                fail_threshold=float(params.get("fail_threshold", 0.002)),
                orientation_weight=float(params.get("orientation_weight", 0.35)),
                orientation_threshold=float(params.get("orientation_threshold", 1.0)),
                max_joint_step=float(params.get("max_joint_step", 0.006)),
                control_mode=str(params.get("control_mode", "actuator")),
                settle_steps=int(params.get("settle_steps", 0)),
                step_callback=None,
                control_frame=str(params.get("control_frame", payload.get("control_frame", "grasp_tool"))),
            )
            final_error = float(segment_errors.get(f"move_to_pregrasp_{index:03d}", float("inf")))
            final_orientation_error = float(segment_orientation_errors.get(f"move_to_pregrasp_{index:03d}", float("inf")))
            ok = bool(ok and segment_ok)
            current = current_tcp_torso(model, data, side, torso_frame)
        stage_errors["move_to_pregrasp"] = final_error
        stage_orientation_errors["move_to_pregrasp"] = final_orientation_error
        return bool(ok), stage_errors, stage_orientation_errors


def load_skill(path: str | Path | None = None) -> R1ProTorsoFrameMoveToPregraspSkill:
    return R1ProTorsoFrameMoveToPregraspSkill()
