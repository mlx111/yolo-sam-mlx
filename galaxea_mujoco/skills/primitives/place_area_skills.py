from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import mujoco
import numpy as np


@dataclass(frozen=True)
class PlaceAreaResult:
    name: str
    success: bool
    occupied: bool = False
    selected_site: str | None = None
    occupied_objects: tuple[str, ...] = field(default_factory=tuple)
    message: str = ""


class BasePlaceAreaSkill:
    def __init__(self, config_path: str | Path | None = None):
        self.config = json.loads(Path(config_path).read_text()) if config_path is not None else {}
        self.name = self.config.get("name", self.__class__.__name__)


def _site_pos(model: mujoco.MjModel, data: mujoco.MjData, site_name: str) -> np.ndarray:
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
    if site_id < 0:
        raise ValueError(f"MuJoCo site not found: {site_name}")
    return data.site_xpos[site_id].copy()


def _split_names(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return [str(item) for item in value]


def _body_pos(model: mujoco.MjModel, data: mujoco.MjData, body_name: str) -> np.ndarray:
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if body_id < 0:
        raise ValueError(f"MuJoCo body not found: {body_name}")
    return data.xpos[body_id].copy()


def _is_occupied(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    site_name: str,
    candidate_bodies: list[str],
    *,
    radius: float,
    z_tolerance: float,
    exclude_bodies: set[str],
) -> tuple[bool, tuple[str, ...]]:
    center = _site_pos(model, data, site_name)
    occupied: list[str] = []
    for body_name in candidate_bodies:
        if body_name in exclude_bodies:
            continue
        pos = _body_pos(model, data, body_name)
        xy_error = float(np.linalg.norm(pos[:2] - center[:2]))
        z_error = abs(float(pos[2] - center[2]))
        if xy_error <= radius and z_error <= z_tolerance:
            occupied.append(body_name)
    return bool(occupied), tuple(occupied)


class DetectPlaceOccupancySkill(BasePlaceAreaSkill):
    def execute_recovery_action(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        params: dict,
        *,
        step_callback: Callable[[], None] | None = None,
    ) -> PlaceAreaResult:
        del step_callback
        mujoco.mj_forward(model, data)
        site_name = str(params.get("place_site", self.config.get("place_site", "place_zone_site")))
        candidate_bodies = _split_names(params.get("candidate_bodies", self.config.get("candidate_bodies", [])))
        exclude_bodies = set(_split_names(params.get("exclude_bodies", self.config.get("exclude_bodies", []))))
        radius = float(params.get("occupancy_radius", self.config.get("occupancy_radius", 0.12)))
        z_tolerance = float(params.get("z_tolerance", self.config.get("z_tolerance", 0.12)))
        occupied, bodies = _is_occupied(
            model,
            data,
            site_name,
            candidate_bodies,
            radius=radius,
            z_tolerance=z_tolerance,
            exclude_bodies=exclude_bodies,
        )
        return PlaceAreaResult(
            self.name,
            success=not occupied,
            occupied=occupied,
            selected_site=site_name,
            occupied_objects=bodies,
            message=f"place_site={site_name}, occupied={occupied}, objects={len(bodies)}",
        )


class ChooseAlternatePlaceSkill(BasePlaceAreaSkill):
    def execute_recovery_action(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        params: dict,
        *,
        step_callback: Callable[[], None] | None = None,
    ) -> PlaceAreaResult:
        del step_callback
        mujoco.mj_forward(model, data)
        place_sites = _split_names(params.get("place_sites", self.config.get("place_sites", [])))
        if not place_sites:
            primary = str(params.get("place_site", self.config.get("place_site", "place_zone_site")))
            alternate = str(params.get("alternate_place_site", self.config.get("alternate_place_site", "alternate_place_zone_site")))
            place_sites = [primary, alternate]
        candidate_bodies = _split_names(params.get("candidate_bodies", self.config.get("candidate_bodies", [])))
        exclude_bodies = set(_split_names(params.get("exclude_bodies", self.config.get("exclude_bodies", []))))
        radius = float(params.get("occupancy_radius", self.config.get("occupancy_radius", 0.12)))
        z_tolerance = float(params.get("z_tolerance", self.config.get("z_tolerance", 0.12)))
        occupied_summary: list[str] = []
        for site_name in place_sites:
            occupied, bodies = _is_occupied(
                model,
                data,
                site_name,
                candidate_bodies,
                radius=radius,
                z_tolerance=z_tolerance,
                exclude_bodies=exclude_bodies,
            )
            if not occupied:
                return PlaceAreaResult(
                    self.name,
                    success=True,
                    occupied=False,
                    selected_site=site_name,
                    occupied_objects=tuple(occupied_summary),
                    message=f"selected_place_site={site_name}",
                )
            occupied_summary.extend(f"{site_name}:{body}" for body in bodies)
        return PlaceAreaResult(
            self.name,
            success=False,
            occupied=True,
            selected_site=None,
            occupied_objects=tuple(occupied_summary),
            message="no_free_place_site",
        )
