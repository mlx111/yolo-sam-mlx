from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from skills.torso_frame._common import current_tcp_torso, resolve_torso_target, topdown_xmat_world
from skills.torso_frame.trajectory_planner import TorsoFrameCartesianTrajectoryPlanner

DEFAULT_PREGRASP_OFFSET_TORSO = np.array([0.0, 0.0, 0.08], dtype=np.float64)


@dataclass(frozen=True)
class TorsoFramePlanCartesianTrajectoryResult:
    success: bool
    side: str
    torso_frame: str
    mode: str
    output_path: str
    start_torso: list[float]
    target_torso: list[float]
    pregrasp_torso: list[float]
    waypoint_count: int
    message: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class R1ProTorsoFramePlanCartesianTrajectorySkill:
    def execute_recovery_action(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        params: dict[str, Any],
    ) -> TorsoFramePlanCartesianTrajectoryResult:
        side = str(params.get("side", "left"))
        torso_frame = str(params.get("torso_frame", "torso_link4"))
        control_frame = str(params.get("control_frame", "grasp_tool"))
        mode = str(params.get("mode", params.get("trajectory_mode", "side_then_in")))
        output_path = Path(str(params.get("output_path", self._default_output_path(side, params))))

        target_torso = resolve_torso_target(params)
        pregrasp_offset_torso = self._pregrasp_offset_torso(params)
        pregrasp_torso = target_torso + pregrasp_offset_torso
        start_torso = current_tcp_torso(model, data, side, torso_frame)
        xmat_world = topdown_xmat_world(model, data, side, params)

        waypoints = TorsoFrameCartesianTrajectoryPlanner().plan(
            start_torso=start_torso,
            target_torso=target_torso,
            pregrasp_torso=pregrasp_torso,
            xmat_world=xmat_world,
            mode=mode,
            torso_frame=torso_frame,
            control_frame=control_frame,
            step_distance=float(params.get("step_distance", 0.005)),
            max_num_points=int(params.get("max_num_points", 100)),
            safe_lift=float(params.get("safe_lift", 0.0)),
            clearance_z=float(params.get("clearance_z", 0.05)),
            side_offset_y=float(params.get("side_offset_y", 0.0)),
            side_offset_x=float(params.get("side_offset_x", -0.06)),
            sequential=bool(params.get("sequential", False)),
            axis_order=str(params.get("axis_order", "xyz")),
        )

        payload = {
            "schema_version": "torso_frame_cartesian_trajectory_v1",
            "side": side,
            "torso_frame": torso_frame,
            "control_frame": control_frame,
            "mode": mode,
            "orientation_mode": str(params.get("topdown_mode", "palm_down")),
            "target_kind": str(params.get("target_kind", "pregrasp")),
            "start_torso": np.round(start_torso, 9).tolist(),
            "target_torso": np.round(target_torso, 9).tolist(),
            "pregrasp_offset_torso": np.round(pregrasp_offset_torso, 9).tolist(),
            "pregrasp_torso": np.round(pregrasp_torso, 9).tolist(),
            "waypoints": [waypoint.to_dict() for waypoint in waypoints],
            "metadata": {
                "step_distance": float(params.get("step_distance", 0.005)),
                "max_num_points": int(params.get("max_num_points", 100)),
                "safe_lift": float(params.get("safe_lift", 0.0)),
                "clearance_z": float(params.get("clearance_z", 0.05)),
                "side_offset_y": float(params.get("side_offset_y", 0.0)),
                "side_offset_x": float(params.get("side_offset_x", -0.06)),
                "sequential": bool(params.get("sequential", False)),
                "axis_order": str(params.get("axis_order", "xyz")),
            },
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

        return TorsoFramePlanCartesianTrajectoryResult(
            success=True,
            side=side,
            torso_frame=torso_frame,
            mode=mode,
            output_path=str(output_path),
            start_torso=np.round(start_torso, 6).tolist(),
            target_torso=np.round(target_torso, 6).tolist(),
            pregrasp_torso=np.round(pregrasp_torso, 6).tolist(),
            waypoint_count=len(waypoints),
            message=f"planned {len(waypoints)} waypoints: {mode}",
        )

    @staticmethod
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
        return DEFAULT_PREGRASP_OFFSET_TORSO.copy()

    @staticmethod
    def _default_output_path(side: str, params: dict[str, Any]) -> Path:
        if params.get("_runtime_tmp_dir"):
            return Path(str(params["_runtime_tmp_dir"])) / "trajectories" / f"{side}_pregrasp_trajectory.json"
        return Path("output") / f"{side}_pregrasp_trajectory.json"


def load_skill(path: str | Path | None = None) -> R1ProTorsoFramePlanCartesianTrajectorySkill:
    return R1ProTorsoFramePlanCartesianTrajectorySkill()
