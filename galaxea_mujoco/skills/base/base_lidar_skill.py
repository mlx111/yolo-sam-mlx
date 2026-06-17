from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np


@dataclass(frozen=True)
class BaseLidarScan:
    site_name: str
    origin: np.ndarray
    directions: np.ndarray
    ranges: np.ndarray
    hit_geom_ids: np.ndarray
    min_range: float
    max_range: float
    horizontal_fov_deg: float


class R1ProBaseLidarSkill:
    """Simulated chassis lidar using MuJoCo ray casting from the base lidar site."""

    def __init__(
        self,
        site_name: str = "base_lidar_site",
        *,
        ray_count: int = 181,
        horizontal_fov_deg: float = 360.0,
        min_range: float = 0.1,
        max_range: float = 5.0,
    ):
        self.site_name = site_name
        self.ray_count = int(ray_count)
        self.horizontal_fov_deg = float(horizontal_fov_deg)
        self.min_range = float(min_range)
        self.max_range = float(max_range)

    def scan(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        *,
        ray_count: int | None = None,
        horizontal_fov_deg: float | None = None,
        min_range: float | None = None,
        max_range: float | None = None,
        exclude_sensor_body: bool = True,
    ) -> BaseLidarScan:
        site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, self.site_name)
        if site_id < 0:
            raise ValueError(f"MuJoCo site not found: {self.site_name}")

        count = self.ray_count if ray_count is None else int(ray_count)
        fov = self.horizontal_fov_deg if horizontal_fov_deg is None else float(horizontal_fov_deg)
        near = self.min_range if min_range is None else float(min_range)
        far = self.max_range if max_range is None else float(max_range)
        if count <= 0:
            raise ValueError("ray_count must be positive")
        if near < 0.0 or far <= near:
            raise ValueError("expected 0 <= min_range < max_range")

        mujoco.mj_forward(model, data)
        origin = data.site_xpos[site_id].copy()
        site_xmat = data.site_xmat[site_id].reshape(3, 3)
        bodyexclude = int(model.site_bodyid[site_id]) if exclude_sensor_body else -1
        forward = site_xmat[:, 0]
        left = site_xmat[:, 1]

        endpoint = not np.isclose(abs(fov), 360.0)
        angles = np.linspace(-0.5 * np.deg2rad(fov), 0.5 * np.deg2rad(fov), count, endpoint=endpoint)
        directions = []
        ranges = []
        hit_geom_ids = []
        geom_group = np.ones(6, dtype=np.uint8)
        for angle in angles:
            direction = np.cos(angle) * forward + np.sin(angle) * left
            direction[2] = 0.0
            norm = np.linalg.norm(direction)
            if norm <= 1e-9:
                direction = np.array([1.0, 0.0, 0.0], dtype=np.float64)
            else:
                direction = direction / norm
            geom_id = np.array([-1], dtype=np.int32)
            distance = float(mujoco.mj_ray(model, data, origin, direction, geom_group, 1, bodyexclude, geom_id))
            if distance < near or distance > far:
                distance = far
                geom_id[0] = -1
            directions.append(direction)
            ranges.append(distance)
            hit_geom_ids.append(int(geom_id[0]))

        return BaseLidarScan(
            site_name=self.site_name,
            origin=origin,
            directions=np.asarray(directions, dtype=np.float64),
            ranges=np.asarray(ranges, dtype=np.float64),
            hit_geom_ids=np.asarray(hit_geom_ids, dtype=np.int32),
            min_range=near,
            max_range=far,
            horizontal_fov_deg=fov,
        )

    def execute_recovery_action(self, model: mujoco.MjModel, data: mujoco.MjData, params: dict) -> BaseLidarScan:
        return self.scan(
            model,
            data,
            ray_count=params.get("ray_count"),
            horizontal_fov_deg=params.get("horizontal_fov_deg"),
            min_range=params.get("min_range"),
            max_range=params.get("max_range"),
            exclude_sensor_body=bool(params.get("exclude_sensor_body", True)),
        )


def load_skill(site_name: str = "base_lidar_site") -> R1ProBaseLidarSkill:
    return R1ProBaseLidarSkill(site_name=site_name)
