from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

import numpy as np


TrajectoryMode = Literal["straight", "top_then_down", "side_then_in"]


@dataclass(frozen=True)
class TorsoTrajectoryPose:
    position_torso: list[float]
    xmat_world: list[list[float]]
    torso_frame: str = "torso_link4"
    control_frame: str = "grasp_tool"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class TorsoFrameCartesianTrajectoryPlanner:
    """Generate torso-frame Cartesian waypoints with fixed gripper orientation."""

    def plan(
        self,
        *,
        start_torso: np.ndarray,
        target_torso: np.ndarray,
        pregrasp_torso: np.ndarray,
        xmat_world: np.ndarray,
        mode: TrajectoryMode,
        torso_frame: str,
        control_frame: str,
        step_distance: float = 0.005,
        max_num_points: int = 100,
        safe_lift: float = 0.08,
        clearance_z: float = 0.05,
        side_offset_y: float = 0.0,
        side_offset_x: float = -0.06,
        sequential: bool = False,
        axis_order: str = "xyz",
    ) -> list[TorsoTrajectoryPose]:
        anchors = self._anchors(
            np.asarray(start_torso, dtype=np.float64).reshape(3),
            np.asarray(target_torso, dtype=np.float64).reshape(3),
            np.asarray(pregrasp_torso, dtype=np.float64).reshape(3),
            mode=str(mode),
            safe_lift=float(safe_lift),
            clearance_z=float(clearance_z),
            side_offset_y=float(side_offset_y),
            side_offset_x=float(side_offset_x),
        )
        points = self._interpolate_anchors(
            anchors,
            step_distance=max(float(step_distance), 1e-4),
            max_num_points=max(int(max_num_points), 2),
            sequential=bool(sequential),
            axis_order=str(axis_order),
        )
        xmat = np.asarray(xmat_world, dtype=np.float64).reshape(3, 3)
        return [
            TorsoTrajectoryPose(
                position_torso=np.round(point, 9).tolist(),
                xmat_world=np.round(xmat, 9).tolist(),
                torso_frame=torso_frame,
                control_frame=control_frame,
            )
            for point in points
        ]

    @staticmethod
    def _anchors(
        start: np.ndarray,
        target: np.ndarray,
        pregrasp: np.ndarray,
        *,
        mode: str,
        safe_lift: float,
        clearance_z: float,
        side_offset_y: float,
        side_offset_x: float,
    ) -> list[np.ndarray]:
        if mode == "straight":
            return [start, pregrasp]
        if mode == "top_then_down":
            safe_z = max(float(start[2]), float(pregrasp[2]), float(target[2]) + clearance_z) + safe_lift
            high_start = np.array([start[0], start[1], safe_z], dtype=np.float64)
            high_target = np.array([pregrasp[0], pregrasp[1], safe_z], dtype=np.float64)
            return [start, high_start, high_target, pregrasp]
        if mode == "side_then_in":
            x_entry_pregrasp = pregrasp + np.array([side_offset_x, side_offset_y, 0.0], dtype=np.float64)
            return [start, x_entry_pregrasp, pregrasp]
        raise ValueError(f"Unsupported torso-frame trajectory mode: {mode!r}")

    def _interpolate_anchors(
        self,
        anchors: list[np.ndarray],
        *,
        step_distance: float,
        max_num_points: int,
        sequential: bool,
        axis_order: str,
    ) -> list[np.ndarray]:
        if len(anchors) < 2:
            return anchors
        out: list[np.ndarray] = [anchors[0].copy()]
        remaining = max_num_points - 1
        distances = [float(np.linalg.norm(anchors[i + 1] - anchors[i])) for i in range(len(anchors) - 1)]
        total_distance = max(float(sum(distances)), 1e-9)
        for i, distance in enumerate(distances):
            start = anchors[i]
            end = anchors[i + 1]
            if i == len(distances) - 1:
                budget = remaining + 1
            else:
                budget = min(max(2, int(round(max_num_points * distance / total_distance))), remaining + 1)
            segment = self._interpolate_segment(
                start,
                end,
                point_count=max(2, min(max(int(np.ceil(distance / step_distance)) + 1, 2), budget)),
                sequential=sequential,
                axis_order=axis_order,
            )
            out.extend(segment[1:])
            remaining = max_num_points - len(out)
            if remaining <= 0:
                break
        if not np.allclose(out[-1], anchors[-1]):
            out[-1] = anchors[-1].copy()
        return out

    def _interpolate_segment(
        self,
        start: np.ndarray,
        end: np.ndarray,
        *,
        point_count: int,
        sequential: bool,
        axis_order: str,
    ) -> list[np.ndarray]:
        point_count = max(int(point_count), 2)
        if not sequential:
            return [
                start + self._smoothstep5(i / (point_count - 1)) * (end - start)
                for i in range(point_count)
            ]
        axis_indices = self._axis_indices(axis_order)
        points: list[np.ndarray] = []
        for i in range(point_count):
            s = i / (point_count - 1)
            point = start.copy()
            for segment_index, axis_index in enumerate(axis_indices):
                t0 = segment_index / 3.0
                t1 = (segment_index + 1) / 3.0
                if s <= t0:
                    value = start[axis_index]
                elif s >= t1:
                    value = end[axis_index]
                else:
                    value = self._fifth_order((s - t0) / (t1 - t0), start[axis_index], end[axis_index])
                point[axis_index] = value
            points.append(point)
        return points

    @staticmethod
    def _axis_indices(axis_order: str) -> list[int]:
        mapping = {"x": 0, "y": 1, "z": 2}
        order = axis_order.lower()
        if len(order) != 3 or set(order) != {"x", "y", "z"}:
            raise ValueError(f"axis_order must contain x/y/z exactly once, got {axis_order!r}")
        return [mapping[item] for item in order]

    @staticmethod
    def _smoothstep5(s: float) -> float:
        s = float(np.clip(s, 0.0, 1.0))
        return 10.0 * s**3 - 15.0 * s**4 + 6.0 * s**5

    def _fifth_order(self, s: float, s0: float, s1: float) -> float:
        return float(s0 + self._smoothstep5(s) * (s1 - s0))
