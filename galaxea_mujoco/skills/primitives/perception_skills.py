from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import mujoco
import numpy as np


@dataclass(frozen=True)
class PerceptionObject:
    name: str
    position: np.ndarray
    body_name: str | None = None
    geom_name: str | None = None
    site_name: str | None = None
    orientation: np.ndarray | None = None
    size: np.ndarray | None = None
    label: str | None = None
    confidence: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_grasp_params(self) -> dict[str, float | str]:
        params: dict[str, float | str] = {
            "object_x": float(self.position[0]),
            "object_y": float(self.position[1]),
            "object_z": float(self.position[2]),
        }
        if self.body_name is not None:
            params["object_body"] = self.body_name
        if self.geom_name is not None:
            params["object_geom"] = self.geom_name
        if self.site_name is not None:
            params["object_site"] = self.site_name
        return params


@dataclass(frozen=True)
class PerceptionSkillResult:
    name: str
    success: bool
    objects: tuple[PerceptionObject, ...] = ()
    selected_object: PerceptionObject | None = None
    message: str = ""

    def as_grasp_params(self) -> dict[str, float | str]:
        if self.selected_object is None:
            raise ValueError("No selected object is available")
        return self.selected_object.as_grasp_params()


class BasePerceptionSkill:
    def __init__(self, config_path: str | Path | None = None):
        self.config = json.loads(Path(config_path).read_text()) if config_path is not None else {}
        self.name = self.config.get("name", self.__class__.__name__)


def _name(model: mujoco.MjModel, obj_type: mujoco.mjtObj, obj_id: int) -> str:
    return mujoco.mj_id2name(model, obj_type, obj_id) or ""


def _body_object(model: mujoco.MjModel, data: mujoco.MjData, body_name: str, *, confidence: float = 1.0) -> PerceptionObject:
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if body_id < 0:
        raise ValueError(f"MuJoCo body not found: {body_name}")
    return PerceptionObject(
        name=body_name,
        body_name=body_name,
        position=data.xpos[body_id].copy(),
        orientation=data.xmat[body_id].reshape(3, 3).copy(),
        label=_label_from_name(body_name),
        confidence=confidence,
        metadata={"source": "body", "body_id": int(body_id)},
    )


def _geom_object(model: mujoco.MjModel, data: mujoco.MjData, geom_name: str, *, confidence: float = 1.0) -> PerceptionObject:
    geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
    if geom_id < 0:
        raise ValueError(f"MuJoCo geom not found: {geom_name}")
    body_id = int(model.geom_bodyid[geom_id])
    body_name = _name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or None
    return PerceptionObject(
        name=geom_name,
        body_name=body_name,
        geom_name=geom_name,
        position=data.geom_xpos[geom_id].copy(),
        orientation=data.geom_xmat[geom_id].reshape(3, 3).copy(),
        size=model.geom_size[geom_id].copy(),
        label=_label_from_name(geom_name),
        confidence=confidence,
        metadata={"source": "geom", "geom_id": int(geom_id), "body_id": body_id},
    )


def _site_object(model: mujoco.MjModel, data: mujoco.MjData, site_name: str, *, confidence: float = 1.0) -> PerceptionObject:
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
    if site_id < 0:
        raise ValueError(f"MuJoCo site not found: {site_name}")
    return PerceptionObject(
        name=site_name,
        site_name=site_name,
        position=data.site_xpos[site_id].copy(),
        orientation=data.site_xmat[site_id].reshape(3, 3).copy(),
        size=model.site_size[site_id].copy(),
        label=_label_from_name(site_name),
        confidence=confidence,
        metadata={"source": "site", "site_id": int(site_id)},
    )


def _label_from_name(name: str) -> str:
    label = name
    for suffix in ("_geom", "_body", "_site", "_freejoint"):
        if label.endswith(suffix):
            label = label[: -len(suffix)]
    return label


def _tcp_position(model: mujoco.MjModel, data: mujoco.MjData, side: str) -> np.ndarray:
    site_name = f"{side}_hand_tcp"
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
    if site_id < 0:
        raise ValueError(f"MuJoCo site not found: {site_name}")
    return data.site_xpos[site_id].copy()


def _body_has_freejoint(model: mujoco.MjModel, body_id: int) -> bool:
    for joint_id in range(model.njnt):
        if int(model.jnt_bodyid[joint_id]) == body_id and model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_FREE:
            return True
    return False


def _within_bounds(position: np.ndarray, bounds: Any) -> bool:
    if bounds is None:
        return True
    arr = np.asarray(bounds, dtype=np.float64)
    if arr.shape != (3, 2):
        raise ValueError("workspace_bounds must be [[xmin,xmax], [ymin,ymax], [zmin,zmax]]")
    return bool(np.all(position >= arr[:, 0]) and np.all(position <= arr[:, 1]))


def _split_names(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return [str(item) for item in value]


def _object_from_params(model: mujoco.MjModel, data: mujoco.MjData, params: dict) -> PerceptionObject:
    if "object_body" in params:
        return _body_object(model, data, str(params["object_body"]))
    if "object_geom" in params:
        return _geom_object(model, data, str(params["object_geom"]))
    if "object_site" in params:
        return _site_object(model, data, str(params["object_site"]))
    if "target_name" in params:
        target = str(params["target_name"])
        for builder, key in (
            (_body_object, mujoco.mjtObj.mjOBJ_BODY),
            (_geom_object, mujoco.mjtObj.mjOBJ_GEOM),
            (_site_object, mujoco.mjtObj.mjOBJ_SITE),
        ):
            if mujoco.mj_name2id(model, key, target) >= 0:
                return builder(model, data, target)
    raise ValueError("Provide object_body/object_geom/object_site or target_name")


def _candidate_objects(model: mujoco.MjModel, data: mujoco.MjData, params: dict) -> list[PerceptionObject]:
    objects: list[PerceptionObject] = []
    for body_name in _split_names(params.get("object_bodies")):
        objects.append(_body_object(model, data, body_name))
    for geom_name in _split_names(params.get("object_geoms")):
        objects.append(_geom_object(model, data, geom_name))
    for site_name in _split_names(params.get("object_sites")):
        objects.append(_site_object(model, data, site_name))
    if objects:
        return objects

    name_prefix = params.get("name_prefix")
    exclude_prefixes = tuple(_split_names(params.get("exclude_prefix", self_default_excludes())))
    workspace_bounds = params.get("workspace_bounds")
    movable_only = bool(params.get("movable_only", True))

    for body_id in range(1, model.nbody):
        body_name = _name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
        if not body_name:
            continue
        if name_prefix is not None and not body_name.startswith(str(name_prefix)):
            continue
        if exclude_prefixes and body_name.startswith(exclude_prefixes):
            continue
        if movable_only and not _body_has_freejoint(model, body_id):
            continue
        obj = _body_object(model, data, body_name)
        if _within_bounds(obj.position, workspace_bounds):
            objects.append(obj)
    return objects


def self_default_excludes() -> list[str]:
    return [
        "left_",
        "right_",
        "torso_",
        "steer_",
        "wheel_",
        "base_",
        "grasp_table",
        "floor",
        "world",
    ]


def _select_by_label(objects: list[PerceptionObject], target_label: str | None, target_name: str | None) -> PerceptionObject | None:
    if target_name:
        for obj in objects:
            names = (obj.name, obj.body_name, obj.geom_name, obj.site_name)
            if target_name in [name for name in names if name is not None]:
                return obj
        for obj in objects:
            if target_name in obj.name:
                return obj

    if target_label:
        target = target_label.lower()
        exact = [obj for obj in objects if (obj.label or "").lower() == target]
        if exact:
            return max(exact, key=lambda obj: obj.confidence)
        fuzzy = [obj for obj in objects if target in obj.name.lower() or target in (obj.label or "").lower()]
        if fuzzy:
            return max(fuzzy, key=lambda obj: obj.confidence)
    return None


class DetectObjectPoseSkill(BasePerceptionSkill):
    def execute_recovery_action(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        params: dict,
        *,
        step_callback: Callable[[], None] | None = None,
    ) -> PerceptionSkillResult:
        del step_callback
        mujoco.mj_forward(model, data)
        obj = _object_from_params(model, data, params)
        return PerceptionSkillResult(self.name, True, (obj,), obj, f"detected {obj.name}")


class RedetectTargetPoseSkill(BasePerceptionSkill):
    def execute_recovery_action(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        params: dict,
        *,
        step_callback: Callable[[], None] | None = None,
    ) -> PerceptionSkillResult:
        del step_callback
        mujoco.mj_forward(model, data)
        try:
            obj = _object_from_params(model, data, params)
            return PerceptionSkillResult(self.name, True, (obj,), obj, f"redetected {obj.name}")
        except ValueError:
            candidates = _candidate_objects(model, data, params)
            selected = _select_by_label(candidates, params.get("target_label"), params.get("target_name"))
            success = selected is not None
            return PerceptionSkillResult(
                self.name,
                success,
                tuple(candidates),
                selected,
                f"searched {len(candidates)} candidates",
            )


class DetectMultipleObjectsSkill(BasePerceptionSkill):
    def execute_recovery_action(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        params: dict,
        *,
        step_callback: Callable[[], None] | None = None,
    ) -> PerceptionSkillResult:
        del step_callback
        mujoco.mj_forward(model, data)
        objects = _candidate_objects(model, data, params)
        return PerceptionSkillResult(
            self.name,
            bool(objects),
            tuple(objects),
            objects[0] if objects else None,
            f"detected {len(objects)} object(s)",
        )


class ClassifyTargetObjectSkill(BasePerceptionSkill):
    def execute_recovery_action(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        params: dict,
        *,
        step_callback: Callable[[], None] | None = None,
    ) -> PerceptionSkillResult:
        del step_callback
        mujoco.mj_forward(model, data)
        objects = list(params.get("objects", ()))
        if not objects:
            objects = _candidate_objects(model, data, params)
        objects = [_coerce_object(obj) for obj in objects]
        selected = _select_by_label(objects, params.get("target_label"), params.get("target_name"))
        if selected is None and "target_position" in params and objects:
            target_pos = np.asarray(params["target_position"], dtype=np.float64)
            selected = min(objects, key=lambda obj: float(np.linalg.norm(obj.position - target_pos)))
        if selected is None and objects:
            selected = max(objects, key=lambda obj: obj.confidence)
        return PerceptionSkillResult(
            self.name,
            selected is not None,
            tuple(objects),
            selected,
            f"selected {selected.name}" if selected is not None else "no matching object",
        )


class SelectCorrectObjectSkill(BasePerceptionSkill):
    def execute_recovery_action(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        params: dict,
        *,
        step_callback: Callable[[], None] | None = None,
    ) -> PerceptionSkillResult:
        del step_callback
        mujoco.mj_forward(model, data)
        objects = list(params.get("objects", ()))
        if not objects:
            objects = _candidate_objects(model, data, params)
        objects = [_coerce_object(obj) for obj in objects]
        selected = _select_by_label(objects, params.get("target_label"), params.get("target_name"))
        if selected is None and "target_position" in params and objects:
            target_pos = np.asarray(params["target_position"], dtype=np.float64)
            selected = min(objects, key=lambda obj: float(np.linalg.norm(obj.position - target_pos)))
        require_unique = bool(params.get("require_unique", self.config.get("require_unique", False)))
        if require_unique and selected is not None:
            same_label = [
                obj
                for obj in objects
                if obj is not selected
                and selected.label is not None
                and obj.label is not None
                and obj.label.lower() == selected.label.lower()
            ]
            if same_label and "target_position" not in params and not params.get("target_name"):
                return PerceptionSkillResult(
                    self.name,
                    False,
                    tuple(objects),
                    None,
                    f"ambiguous_candidates={len(same_label) + 1}",
                )
        return PerceptionSkillResult(
            self.name,
            selected is not None,
            tuple(objects),
            selected,
            f"selected={selected.name}" if selected is not None else "no_correct_object",
        )


class VerifyGraspedObjectSkill(BasePerceptionSkill):
    def execute_recovery_action(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        params: dict,
        *,
        step_callback: Callable[[], None] | None = None,
    ) -> PerceptionSkillResult:
        del step_callback
        mujoco.mj_forward(model, data)
        side = str(params.get("side", "left"))
        tcp_pos = _tcp_position(model, data, side)
        max_distance = float(params.get("max_grasp_distance", self.config.get("max_grasp_distance", 0.08)))
        expected_name = params.get("expected_object_body", params.get("object_body"))
        expected_label = params.get("expected_label", params.get("target_label"))
        candidates = _candidate_objects(model, data, params)
        if not candidates and expected_name is not None:
            candidates = [_body_object(model, data, str(expected_name))]
        near = [obj for obj in candidates if float(np.linalg.norm(obj.position - tcp_pos)) <= max_distance]
        selected = _select_by_label(near, expected_label, expected_name)
        if selected is None and near:
            selected = min(near, key=lambda obj: float(np.linalg.norm(obj.position - tcp_pos)))
        success = selected is not None
        message = f"near_tcp={len(near)}, max_distance={max_distance:.6f}"
        if selected is not None:
            message += f", selected={selected.name}"
        return PerceptionSkillResult(self.name, success, tuple(candidates), selected, message)


class MultiViewRedetectSkill(BasePerceptionSkill):
    def execute_recovery_action(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        params: dict,
        *,
        step_callback: Callable[[], None] | None = None,
    ) -> PerceptionSkillResult:
        # First implementation uses MuJoCo state as a deterministic multi-view backend.
        # The API accepts view_offsets so a camera-based backend can replace this later.
        del step_callback
        mujoco.mj_forward(model, data)
        candidates = _candidate_objects(model, data, params)
        if not candidates:
            try:
                obj = _object_from_params(model, data, params)
                candidates = [obj]
            except ValueError:
                candidates = []
        selected = _select_by_label(candidates, params.get("target_label"), params.get("target_name"))
        if selected is None and candidates:
            selected = max(candidates, key=lambda obj: obj.confidence)
        return PerceptionSkillResult(
            self.name,
            selected is not None,
            tuple(candidates),
            selected,
            f"multi_view_candidates={len(candidates)}",
        )


def _coerce_object(value: Any) -> PerceptionObject:
    if isinstance(value, PerceptionObject):
        return value
    if isinstance(value, dict):
        position = np.asarray(value["position"], dtype=np.float64)
        return PerceptionObject(
            name=str(value.get("name", value.get("body_name", value.get("geom_name", "object")))),
            body_name=value.get("body_name"),
            geom_name=value.get("geom_name"),
            site_name=value.get("site_name"),
            position=position,
            orientation=np.asarray(value["orientation"], dtype=np.float64) if "orientation" in value else None,
            size=np.asarray(value["size"], dtype=np.float64) if "size" in value else None,
            label=value.get("label"),
            confidence=float(value.get("confidence", 1.0)),
            metadata=dict(value.get("metadata", {})),
        )
    raise TypeError(f"Unsupported object value: {type(value).__name__}")
