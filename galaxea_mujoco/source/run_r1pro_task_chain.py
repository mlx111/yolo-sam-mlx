"""Run R1Pro task chains and optionally write experience-memory entries."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("MUJOCO_GL", "osmesa")
_DELAY_CONTEXTS: dict[int, "_DelayedControlContext"] = {}
_TRACE_RECORDERS: dict[int, "TrajectoryTraceRecorder"] = {}
_VIEWER_CONTEXTS: dict[int, "_PassiveViewerContext"] = {}

import mujoco
import numpy as np
from PIL import Image

from memory import MemoryLibrary, TaskChainResult, make_memory_entry
from experience_adapters import R1ProMujocoAdapter
from experience_core import ExperienceLibrary, apply_calibration_to_position
from source.control_profiles import DEFAULT_PHYSICAL_CONTROL_PROFILE_ID, get_control_profile
from source.grasp_defaults import DEFAULT_GRASP_OFFSET_Z, DEFAULT_PREGRASP_DISTANCE
from skills.base.arm_ik_skill import ARM_JOINTS, R1ProArmIKSkill
from skills.base.base_motion_skill import load_skill as load_base_motion
from skills.primitives.approach_object_skill import load_skill as load_approach
from skills.primitives.choose_alternate_place_skill import load_skill as load_choose_place
from skills.primitives.detect_multiple_objects_skill import load_skill as load_detect_multiple
from skills.primitives.detect_place_occupancy_skill import load_skill as load_detect_occupancy
from skills.primitives.left_gripper_close_skill import load_skill as load_left_close
from skills.primitives.left_vertical_lift_skill import load_skill as load_left_lift
from skills.primitives.move_to_pregrasp_skill import load_skill as load_pregrasp
from skills.primitives.open_gripper_release_skill import load_skill as load_release
from skills.primitives.place_object_skill import load_skill as load_place
from skills.primitives.pre_grasp_safe_posture_skill import load_skill as load_pregrasp_safe_posture
from skills.primitives.adjust_torso_for_reach_skill import load_skill as load_adjust_torso_for_reach
from skills.primitives.recover_from_joint_limit_skill import load_skill as load_recover_from_joint_limit
from skills.primitives.reposition_base_for_reach_skill import load_skill as load_reposition_base_for_reach
from skills.primitives.retry_lift_after_grasp_check_skill import load_skill as load_retry_lift_after_grasp_check
from skills.primitives.retry_pregrasp_with_safer_offset_skill import load_skill as load_retry_pregrasp_with_safer_offset
from skills.primitives.slow_cartesian_approach_skill import load_skill as load_slow_cartesian_approach
from skills.primitives.right_gripper_close_skill import load_skill as load_right_close
from skills.primitives.right_vertical_lift_skill import load_skill as load_right_lift
from skills.primitives.select_correct_object_skill import load_skill as load_select_correct
from skills.primitives.safe_transport_pose_skill import load_skill as load_safe_transport_pose
from skills.primitives.torso_set_height_skill import load_skill as load_torso_height
from skills.primitives.torso_turn_to_target_skill import load_skill as load_torso_turn
from skills.primitives.verify_grasp_skill import load_skill as load_verify_grasp
from skills.primitives.verify_place_zone_skill import load_skill as load_verify_place


@dataclass(frozen=True)
class JointPosturePrepareResult:
    success: bool
    final_error: float
    message: str = ""


def _load_gripper_close(side: str):
    if side == "left":
        return load_left_close()
    if side == "right":
        return load_right_close()
    raise ValueError(f"Unsupported grasp side: {side!r}")


def _load_vertical_lift(side: str):
    if side == "left":
        return load_left_lift()
    if side == "right":
        return load_right_lift()
    raise ValueError(f"Unsupported grasp side: {side!r}")


def _side_skill_name(side: str, suffix: str) -> str:
    if side not in {"left", "right"}:
        raise ValueError(f"Unsupported grasp side: {side!r}")
    return f"{side}_{suffix}"


def _round_list(values: Any, digits: int = 6) -> list[float]:
    return [round(float(x), digits) for x in np.asarray(values, dtype=np.float64).reshape(-1)]


def _body_pos(model: mujoco.MjModel, data: mujoco.MjData, body_name: str) -> np.ndarray:
    mujoco.mj_forward(model, data)
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if body_id < 0:
        raise ValueError(f"MuJoCo body not found: {body_name}")
    return data.xpos[body_id].copy()


def _first_existing_body(model: mujoco.MjModel, names: list[str], *, required: bool = True) -> str:
    for name in names:
        if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name) >= 0:
            return name
    if required:
        raise ValueError(f"None of the MuJoCo bodies exist: {names}")
    return ""


def _first_existing_site(model: mujoco.MjModel, names: list[str], *, required: bool = True) -> str:
    for name in names:
        if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name) >= 0:
            return name
    if required:
        raise ValueError(f"None of the MuJoCo sites exist: {names}")
    return ""


def _site_pos(model: mujoco.MjModel, data: mujoco.MjData, site_name: str) -> np.ndarray:
    mujoco.mj_forward(model, data)
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
    if site_id < 0:
        raise ValueError(f"MuJoCo site not found: {site_name}")
    return data.site_xpos[site_id].copy()


def _set_freejoint_body_pose(model: mujoco.MjModel, data: mujoco.MjData, body_name: str, pos: list[float]) -> None:
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if body_id < 0:
        raise ValueError(f"MuJoCo body not found: {body_name}")
    for joint_id in range(model.njnt):
        if int(model.jnt_bodyid[joint_id]) != body_id:
            continue
        if model.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_FREE:
            continue
        qpos_id = int(model.jnt_qposadr[joint_id])
        dof_id = int(model.jnt_dofadr[joint_id])
        data.qpos[qpos_id : qpos_id + 3] = np.asarray(pos, dtype=np.float64)
        data.qpos[qpos_id + 3 : qpos_id + 7] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        data.qvel[dof_id : dof_id + 6] = 0.0
        mujoco.mj_forward(model, data)
        return
    raise ValueError(f"Body {body_name!r} does not have a free joint")


def _calibration_id(calibration: dict[str, Any] | None) -> str:
    return str((calibration or {}).get("calibration_id") or "")


def _object_pose_bias(calibration: dict[str, Any] | None) -> list[float]:
    bias = list((calibration or {}).get("object_pose_bias") or [])
    out: list[float] = []
    for item in bias[:3]:
        try:
            out.append(float(item))
        except (TypeError, ValueError):
            out.append(0.0)
    while len(out) < 3:
        out.append(0.0)
    return out


def _apply_object_pose_calibration(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    body_name: str,
    calibration: dict[str, Any] | None,
) -> dict[str, Any]:
    nominal = _body_pos(model, data, body_name)
    calibration_id = _calibration_id(calibration)
    bias = _object_pose_bias(calibration)
    applied = bool(calibration_id and any(abs(value) > 1e-9 for value in bias))
    if applied:
        calibrated = apply_calibration_to_position(nominal, calibration)
        _set_freejoint_body_pose(model, data, body_name, calibrated)
    else:
        calibrated = _round_list(nominal)
    return {
        "applied": applied,
        "calibration_id": calibration_id,
        "object_pose_bias": _round_list(bias),
        "nominal_object_start": _round_list(nominal),
        "calibrated_object_start": _round_list(calibrated),
    }


def _pose_position(pose: Any) -> list[float] | None:
    if not isinstance(pose, dict):
        return None
    position = pose.get("position") or pose.get("start_position") or pose.get("xyz")
    if hasattr(position, "tolist"):
        position = position.tolist()
    if not isinstance(position, (list, tuple)) or len(position) < 3:
        return None
    try:
        return [float(position[0]), float(position[1]), float(position[2])]
    except (TypeError, ValueError):
        return None


def _state_named_pose(state: dict[str, Any] | None, names: list[str]) -> tuple[str, list[float]] | None:
    if not isinstance(state, dict):
        return None
    groups = [state.get("object_poses"), state.get("obstacle_poses")]
    for group in groups:
        if not isinstance(group, dict):
            continue
        for name in names:
            position = _pose_position(group.get(name))
            if position is not None:
                return name, position
    return None


def _apply_sandbox_initial_state(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    state: dict[str, Any] | None,
    *,
    object_body_map: dict[str, list[str]],
) -> dict[str, Any]:
    if not isinstance(state, dict) or not state:
        return {"applied": False, "reason": "no_sandbox_initial_state"}

    applied_objects: dict[str, Any] = {}
    skipped_objects: dict[str, str] = {}
    for body_name, aliases in object_body_map.items():
        found = _state_named_pose(state, [body_name, *aliases])
        if found is None:
            skipped_objects[body_name] = "pose_not_found"
            continue
        source_name, position = found
        try:
            _set_freejoint_body_pose(model, data, body_name, position)
            applied_objects[body_name] = {"source_name": source_name, "position": _round_list(position)}
        except ValueError as exc:
            skipped_objects[body_name] = str(exc)

    qpos_applied = False
    qvel_applied = False
    qpos = state.get("robot_qpos")
    qvel = state.get("robot_qvel")
    if isinstance(qpos, list) and len(qpos) == model.nq:
        data.qpos[:] = np.asarray(qpos, dtype=np.float64)
        qpos_applied = True
    if isinstance(qvel, list) and len(qvel) == model.nv:
        data.qvel[:] = np.asarray(qvel, dtype=np.float64)
        qvel_applied = True
    mujoco.mj_forward(model, data)
    return {
        "applied": bool(applied_objects or qpos_applied or qvel_applied),
        "schema_version": str(state.get("schema_version") or ""),
        "source_episode_id": str(state.get("source_episode_id") or ""),
        "confidence": float(state.get("confidence") or 0.0),
        "missing_fields": list(state.get("missing_fields") or []),
        "applied_objects": applied_objects,
        "skipped_objects": skipped_objects,
        "qpos_applied": qpos_applied,
        "qvel_applied": qvel_applied,
        "robot_qpos_count": len(qpos) if isinstance(qpos, list) else 0,
        "robot_qvel_count": len(qvel) if isinstance(qvel, list) else 0,
    }


def _perturbation_value(state: dict[str, Any] | None, key: str, default: float = 0.0) -> float:
    if not isinstance(state, dict):
        return float(default)
    perturbation = state.get("perturbation") if isinstance(state.get("perturbation"), dict) else {}
    dynamics_profile = state.get("dynamics_profile") if isinstance(state.get("dynamics_profile"), dict) else {}
    try:
        return float(perturbation.get(key, dynamics_profile.get(key, state.get(key, default))))
    except (TypeError, ValueError):
        return float(default)


def _perturbation_int_value(state: dict[str, Any] | None, key: str, default: int = 0) -> int:
    if not isinstance(state, dict):
        return int(default)
    perturbation = state.get("perturbation") if isinstance(state.get("perturbation"), dict) else {}
    dynamics_profile = state.get("dynamics_profile") if isinstance(state.get("dynamics_profile"), dict) else {}
    try:
        return int(round(float(perturbation.get(key, dynamics_profile.get(key, state.get(key, default))))))
    except (TypeError, ValueError):
        return int(default)


class _DelayedControlContext:
    """Delay MuJoCo control application without changing individual skills."""

    def __init__(self, data: mujoco.MjData, delay_steps: int) -> None:
        self.data = data
        self.delay_steps = max(0, int(delay_steps))
        self._original_mj_step = None
        self._queue: deque[np.ndarray] = deque(maxlen=max(self.delay_steps, 1))
        self.report: dict[str, Any] = {
            "actuation_delay_steps": self.delay_steps,
            "actuation_delay_applied": False,
            "delayed_step_count": 0,
            "control_dimension": int(data.ctrl.size),
        }

    def __enter__(self) -> "_DelayedControlContext":
        if self.delay_steps <= 0 or self.data.ctrl.size == 0:
            return self
        self._queue = deque([self.data.ctrl.copy() for _ in range(self.delay_steps)], maxlen=self.delay_steps)
        self._original_mj_step = mujoco.mj_step

        def delayed_mj_step(model: mujoco.MjModel, data: mujoco.MjData, *args: Any, **kwargs: Any) -> Any:
            if data is not self.data:
                return self._original_mj_step(model, data, *args, **kwargs)
            current_ctrl = data.ctrl.copy()
            delayed_ctrl = self._queue.popleft()
            self._queue.append(current_ctrl)
            data.ctrl[:] = delayed_ctrl
            try:
                return self._original_mj_step(model, data, *args, **kwargs)
            finally:
                data.ctrl[:] = current_ctrl
                self.report["actuation_delay_applied"] = True
                self.report["delayed_step_count"] = int(self.report["delayed_step_count"]) + 1

        mujoco.mj_step = delayed_mj_step
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if self._original_mj_step is not None:
            mujoco.mj_step = self._original_mj_step


def _apply_runtime_dynamics_profile(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    body_name: str,
    sandbox_initial_state: dict[str, Any] | None,
) -> dict[str, Any]:
    friction_scale = _perturbation_value(sandbox_initial_state, "friction_scale", 1.0)
    mass_scale = _perturbation_value(sandbox_initial_state, "mass_scale", 1.0)
    controller_gain_scale = _perturbation_value(sandbox_initial_state, "controller_gain_scale", 1.0)
    contact_solref_time_scale = _perturbation_value(sandbox_initial_state, "contact_solref_time_scale", 1.0)
    contact_solimp_margin_scale = _perturbation_value(sandbox_initial_state, "contact_solimp_margin_scale", 1.0)
    actuation_delay_steps = _perturbation_int_value(sandbox_initial_state, "actuation_delay_steps", 0)
    gripper_closure_bias = _perturbation_value(sandbox_initial_state, "gripper_closure_bias", 0.0)
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    applied_geoms: list[dict[str, Any]] = []
    target_geom_ids: list[int] = []
    if body_id >= 0:
        target_geom_ids = [
            geom_id
            for geom_id in range(model.ngeom)
            if int(model.geom_bodyid[geom_id]) == body_id
        ]
    if abs(friction_scale - 1.0) > 1e-9:
        if target_geom_ids:
            for geom_id in target_geom_ids:
                before = model.geom_friction[geom_id].copy()
                model.geom_friction[geom_id] = before * friction_scale
                applied_geoms.append({
                    "geom_id": int(geom_id),
                    "geom_name": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or "",
                    "friction_before": _round_list(before),
                    "friction_after": _round_list(model.geom_friction[geom_id]),
                })
            mujoco.mj_forward(model, data)
    contact_solver_geoms: list[dict[str, Any]] = []
    if target_geom_ids and (
        abs(contact_solref_time_scale - 1.0) > 1e-9
        or abs(contact_solimp_margin_scale - 1.0) > 1e-9
    ):
        for geom_id in target_geom_ids:
            solref_before = model.geom_solref[geom_id].copy()
            solimp_before = model.geom_solimp[geom_id].copy()
            model.geom_solref[geom_id, 0] = max(1e-5, float(solref_before[0]) * contact_solref_time_scale)
            model.geom_solimp[geom_id, 2] = max(1e-6, float(solimp_before[2]) * contact_solimp_margin_scale)
            contact_solver_geoms.append({
                "geom_id": int(geom_id),
                "geom_name": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or "",
                "solref_before": _round_list(solref_before),
                "solref_after": _round_list(model.geom_solref[geom_id]),
                "solimp_before": _round_list(solimp_before),
                "solimp_after": _round_list(model.geom_solimp[geom_id]),
            })
        mujoco.mj_forward(model, data)
    mass_report: dict[str, Any] = {
        "mass_scale": round(float(mass_scale), 6),
        "mass_applied": False,
        "body_name": body_name,
        "body_mass_before": None,
        "body_mass_after": None,
    }
    if body_id >= 0 and abs(mass_scale - 1.0) > 1e-9:
        before_mass = float(model.body_mass[body_id])
        before_inertia = model.body_inertia[body_id].copy()
        model.body_mass[body_id] = before_mass * mass_scale
        model.body_inertia[body_id] = before_inertia * mass_scale
        mass_report.update({
            "mass_applied": True,
            "body_mass_before": round(before_mass, 6),
            "body_mass_after": round(float(model.body_mass[body_id]), 6),
            "body_inertia_before": _round_list(before_inertia),
            "body_inertia_after": _round_list(model.body_inertia[body_id]),
        })
        mujoco.mj_forward(model, data)
    controller_report: dict[str, Any] = {
        "controller_gain_scale": round(float(controller_gain_scale), 6),
        "controller_gain_applied": False,
        "actuator_count": int(model.nu),
        "scaled_actuator_count": 0,
        "sample_actuators": [],
    }
    if abs(controller_gain_scale - 1.0) > 1e-9:
        sample_actuators: list[dict[str, Any]] = []
        scaled_count = 0
        for actuator_id in range(model.nu):
            gain_before = model.actuator_gainprm[actuator_id].copy()
            bias_before = model.actuator_biasprm[actuator_id].copy()
            force_before = model.actuator_forcerange[actuator_id].copy()
            model.actuator_gainprm[actuator_id, 0] *= controller_gain_scale
            model.actuator_biasprm[actuator_id, 1] *= controller_gain_scale
            model.actuator_biasprm[actuator_id, 2] *= controller_gain_scale
            model.actuator_forcerange[actuator_id] *= controller_gain_scale
            scaled_count += 1
            if len(sample_actuators) < 6:
                sample_actuators.append({
                    "actuator_id": int(actuator_id),
                    "actuator_name": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_id) or "",
                    "gain_before": _round_list(gain_before[:3]),
                    "gain_after": _round_list(model.actuator_gainprm[actuator_id, :3]),
                    "bias_before": _round_list(bias_before[:3]),
                    "bias_after": _round_list(model.actuator_biasprm[actuator_id, :3]),
                    "forcerange_before": _round_list(force_before),
                    "forcerange_after": _round_list(model.actuator_forcerange[actuator_id]),
                })
        controller_report.update({
            "controller_gain_applied": scaled_count > 0,
            "scaled_actuator_count": scaled_count,
            "sample_actuators": sample_actuators,
        })
        mujoco.mj_forward(model, data)
    return {
        "friction_scale": round(float(friction_scale), 6),
        "friction_applied": bool(applied_geoms),
        "friction_geoms": applied_geoms,
        "contact_solref_time_scale": round(float(contact_solref_time_scale), 6),
        "contact_solimp_margin_scale": round(float(contact_solimp_margin_scale), 6),
        "contact_solver_applied": bool(contact_solver_geoms),
        "contact_solver_geoms": contact_solver_geoms,
        **mass_report,
        **controller_report,
        "actuation_delay_steps": int(max(0, actuation_delay_steps)),
        "actuation_delay_applied": False,
        "delayed_step_count": 0,
        "gripper_closure_bias": round(float(gripper_closure_bias), 6),
    }


def _register_actuation_delay_context(data: mujoco.MjData, dynamics_profile: dict[str, Any]) -> _DelayedControlContext | None:
    delay_steps = int(dynamics_profile.get("actuation_delay_steps") or 0)
    if delay_steps <= 0:
        return None
    context = _DelayedControlContext(data, delay_steps)
    _DELAY_CONTEXTS[id(data)] = context
    return context


def _unregister_actuation_delay_context(data: mujoco.MjData) -> None:
    _DELAY_CONTEXTS.pop(id(data), None)


def _merge_actuation_delay_report(dynamics_profile: dict[str, Any], context: _DelayedControlContext | None) -> dict[str, Any]:
    if context is None:
        return dynamics_profile
    updated = dict(dynamics_profile)
    updated.update(context.report)
    return updated


def _uses_actuation_delay(dynamics_profile: dict[str, Any]) -> bool:
    return int(dynamics_profile.get("actuation_delay_steps") or 0) > 0


def _normalize_control_mode(control_mode: str) -> str:
    if control_mode in {"ideal", "direct", "direct_position", "position"}:
        return "ideal"
    if control_mode in {"physical", "actuator"}:
        return "physical"
    raise ValueError(f"Unsupported control_mode: {control_mode}")


def _uses_physical_control(control_mode: str, dynamics_profile: dict[str, Any]) -> bool:
    return _normalize_control_mode(control_mode) == "physical" or _uses_actuation_delay(dynamics_profile)


def _control_execution_report(control_mode: str, dynamics_profile: dict[str, Any], control_profile: dict[str, Any] | None = None) -> dict[str, Any]:
    normalized = _normalize_control_mode(control_mode)
    physical = _uses_physical_control(control_mode, dynamics_profile)
    profile = control_profile if physical and isinstance(control_profile, dict) else {}
    arm_profile = profile.get("arm") if isinstance(profile.get("arm"), dict) else {}
    return {
        "requested_control_mode": control_mode,
        "normalized_control_mode": normalized,
        "execution_type": "physical_actuator" if physical else "direct_position",
        "direct_qpos_default": not physical,
        "actuator_control_enabled": physical,
        "physical_forced_by_delay": normalized != "physical" and _uses_actuation_delay(dynamics_profile),
        "control_profile_id": str(profile.get("profile_id") or ""),
        "arm_control_mode": str(arm_profile.get("control_mode") or ""),
        "velocity_limit_applied": bool(arm_profile.get("velocity_limit")),
    }


def _profile_from_dynamics(dynamics_profile: dict[str, Any]) -> dict[str, Any]:
    profile_id = str(dynamics_profile.get("control_profile_id") or DEFAULT_PHYSICAL_CONTROL_PROFILE_ID)
    return get_control_profile(profile_id)


def _physical_profile(section: str, control_profile: dict[str, Any] | None) -> dict[str, Any]:
    profile_root = control_profile if isinstance(control_profile, dict) else get_control_profile()
    profile = profile_root.get(section)
    return dict(profile) if isinstance(profile, dict) else {}


def _control_overrides(control_mode: str, dynamics_profile: dict[str, Any], control_profile: dict[str, Any] | None = None) -> dict[str, Any]:
    if not _uses_physical_control(control_mode, dynamics_profile):
        return {}
    profile = _physical_profile("arm", control_profile or _profile_from_dynamics(dynamics_profile))
    return profile


def _gripper_overrides(control_mode: str, dynamics_profile: dict[str, Any], control_profile: dict[str, Any] | None = None) -> dict[str, Any]:
    return _physical_profile("gripper", control_profile or _profile_from_dynamics(dynamics_profile)) if _uses_physical_control(control_mode, dynamics_profile) else {}


def _pregrasp_safe_posture_params(control_mode: str, dynamics_profile: dict[str, Any], control_profile: dict[str, Any] | None = None) -> dict[str, Any] | None:
    if not _uses_physical_control(control_mode, dynamics_profile):
        return None
    arm_profile = _physical_profile("arm", control_profile or _profile_from_dynamics(dynamics_profile))
    if not bool(arm_profile.get("pregrasp_safe_posture", False)):
        return None
    return {
        "posture": str(arm_profile.get("pregrasp_safe_posture_name") or "left_pregrasp_seed"),
        "steps": int(arm_profile.get("pregrasp_safe_posture_steps", arm_profile.get("steps", 1800))),
        "settle_steps": int(arm_profile.get("pregrasp_safe_posture_settle_steps", arm_profile.get("settle_steps", 800))),
        "max_joint_step": min(float(arm_profile.get("max_joint_step", 0.006)), 0.004),
        "fail_threshold": 0.08,
        # This is a sandbox seed reset before evaluating physical arm tracking,
        # not evidence that the real robot can jump to this posture.
        "direct_qpos": bool(arm_profile.get("pregrasp_safe_posture_direct_qpos", False)),
    }


def _run_pregrasp_safe_posture_if_needed(
    skill_trace: list[dict[str, Any]],
    model: mujoco.MjModel,
    data: mujoco.MjData,
    control_mode: str,
    dynamics_profile: dict[str, Any],
    control_profile: dict[str, Any] | None = None,
) -> Any | None:
    params = _pregrasp_safe_posture_params(control_mode, dynamics_profile, control_profile)
    if params is None:
        return None
    return _run_traced(
        skill_trace,
        "pre_grasp_safe_posture",
        model,
        data,
        load_pregrasp_safe_posture().execute_recovery_action,
        model,
        data,
        params,
        posture=params["posture"],
    )


def _base_align_target_for_object(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    object_body: str,
    *,
    side: str = "left",
) -> np.ndarray:
    object_pos = _body_pos(model, data, object_body)
    current = np.zeros(3, dtype=np.float64)
    for index, name in enumerate(("base_x", "base_y", "base_yaw")):
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id >= 0:
            current[index] = float(data.qpos[model.jnt_qposadr[joint_id]])
    lateral_offset = 0.08 if side == "left" else -0.08
    target = current.copy()
    target[0] = float(np.clip(object_pos[0] - 0.08, current[0] - 0.18, current[0] + 0.18))
    target[1] = float(np.clip(object_pos[1] - lateral_offset, current[1] - 0.18, current[1] + 0.18))
    target[2] = 0.0
    return target


def _joint_qpos(model: mujoco.MjModel, data: mujoco.MjData, joint_name: str) -> float:
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    if joint_id < 0:
        return 0.0
    return float(data.qpos[model.jnt_qposadr[joint_id]])


def _set_joint_qpos_and_ctrl(model: mujoco.MjModel, data: mujoco.MjData, joint_name: str, value: float) -> None:
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    if joint_id < 0:
        return
    qpos_id = int(model.jnt_qposadr[joint_id])
    data.qpos[qpos_id] = float(value)
    dof_id = int(model.jnt_dofadr[joint_id])
    if 0 <= dof_id < data.qvel.size:
        data.qvel[dof_id] = 0.0
    actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{joint_name}_pos")
    if actuator_id >= 0:
        low, high = model.actuator_ctrlrange[actuator_id]
        data.ctrl[actuator_id] = np.clip(value, low, high)


def _score_whole_body_pregrasp_candidate(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    object_body: str,
    *,
    side: str,
    base_x: float,
    base_y: float,
    torso_yaw: float,
    current_base_x: float,
    current_base_y: float,
    current_torso_yaw: float,
    pregrasp_distance: float,
    grasp_offset_z: float,
    posture_gain: float,
) -> dict[str, Any]:
    trial = mujoco.MjData(model)
    trial.qpos[:] = data.qpos
    trial.qvel[:] = 0.0
    trial.ctrl[:] = data.ctrl
    _set_joint_qpos_and_ctrl(model, trial, "base_x", base_x)
    _set_joint_qpos_and_ctrl(model, trial, "base_y", base_y)
    _set_joint_qpos_and_ctrl(model, trial, "torso_joint4", torso_yaw)
    mujoco.mj_forward(model, trial)
    object_pos = _body_pos(model, trial, object_body)
    target = object_pos + np.array([0.0, 0.0, grasp_offset_z + pregrasp_distance], dtype=np.float64)
    ik_skill = R1ProArmIKSkill()
    q_seed = ik_skill.sync_q_from_mujoco(model, trial)
    q_target, _ik_pos, ik_error = ik_skill.solve_position_ik(
        side,  # type: ignore[arg-type]
        q_seed,
        ik_skill.world_to_pinocchio_position(model, trial, target),
        joint_names=ARM_JOINTS[side],  # type: ignore[index]
        posture_reference=q_seed,
        posture_gain=posture_gain,
    )
    margin = ik_skill.joint_limit_margin_from_pin_q(side, q_target, joint_names=ARM_JOINTS[side])  # type: ignore[arg-type,index]
    motion_cost = abs(base_x - current_base_x) * 0.4 + abs(base_y - current_base_y) * 0.4 + abs(torso_yaw - current_torso_yaw) * 0.2
    margin_penalty = max(0.0, 0.03 - margin) * 5.0
    score = float(ik_error) + margin_penalty + motion_cost
    return {
        "base_x": round(float(base_x), 6),
        "base_y": round(float(base_y), 6),
        "torso_yaw": round(float(torso_yaw), 6),
        "target": _round_list(target),
        "ik_error": round(float(ik_error), 6),
        "joint_limit_margin": round(float(margin), 6),
        "motion_cost": round(float(motion_cost), 6),
        "score": round(float(score), 6),
    }


def _run_whole_body_prepare_grasp_if_needed(
    skill_trace: list[dict[str, Any]],
    model: mujoco.MjModel,
    data: mujoco.MjData,
    control_mode: str,
    dynamics_profile: dict[str, Any],
    object_body: str,
    grasp_params: dict[str, Any],
    control_profile: dict[str, Any] | None = None,
    *,
    side: str = "left",
) -> dict[str, Any] | None:
    if not _uses_physical_control(control_mode, dynamics_profile):
        return None
    arm_profile = _physical_profile("arm", control_profile or _profile_from_dynamics(dynamics_profile))
    if not bool(arm_profile.get("whole_body_prepare_grasp", False)):
        return None
    current_base_x = _joint_qpos(model, data, "base_x")
    current_base_y = _joint_qpos(model, data, "base_y")
    current_torso_yaw = _joint_qpos(model, data, "torso_joint4")
    object_pos = _body_pos(model, data, object_body)
    max_base_delta = float(arm_profile.get("whole_body_prepare_max_base_delta", 0.12))
    max_torso_yaw = float(arm_profile.get("whole_body_prepare_max_torso_yaw", 0.30))
    desired_base_x = float(np.clip(object_pos[0] - 0.22, current_base_x - max_base_delta, current_base_x + max_base_delta))
    desired_base_y = float(np.clip(object_pos[1] - 0.08, current_base_y - max_base_delta, current_base_y + max_base_delta))
    base_x_candidates = [
        current_base_x,
        desired_base_x,
        float(np.clip(desired_base_x - 0.04, current_base_x - max_base_delta, current_base_x + max_base_delta)),
        float(np.clip(desired_base_x + 0.04, current_base_x - max_base_delta, current_base_x + max_base_delta)),
    ]
    base_candidates = [
        current_base_y,
        desired_base_y,
        float(np.clip(desired_base_y - 0.04, current_base_y - max_base_delta, current_base_y + max_base_delta)),
        float(np.clip(desired_base_y + 0.04, current_base_y - max_base_delta, current_base_y + max_base_delta)),
    ]
    torso_candidates = [
        0.0,
        float(np.clip(np.arctan2(object_pos[1] - desired_base_y, object_pos[0]), -max_torso_yaw, max_torso_yaw)),
        -max_torso_yaw,
        max_torso_yaw,
    ]
    scored = []
    for base_x in sorted(set(round(x, 6) for x in base_x_candidates)):
        for base_y in sorted(set(round(x, 6) for x in base_candidates)):
            for torso_yaw in sorted(set(round(x, 6) for x in torso_candidates)):
                scored.append(_score_whole_body_pregrasp_candidate(
                    model,
                    data,
                    object_body,
                    side=side,
                    base_x=float(base_x),
                    base_y=float(base_y),
                    torso_yaw=float(torso_yaw),
                    current_base_x=current_base_x,
                    current_base_y=current_base_y,
                    current_torso_yaw=current_torso_yaw,
                    pregrasp_distance=float(grasp_params.get("pregrasp_distance", DEFAULT_PREGRASP_DISTANCE)),
                    grasp_offset_z=float(grasp_params.get("grasp_offset_z", DEFAULT_GRASP_OFFSET_Z)),
                    posture_gain=float(arm_profile.get("posture_gain", 0.0)),
                ))
    scored.sort(key=lambda item: item["score"])
    selected = scored[0]
    report = {
        "enabled": True,
        "selected": selected,
        "candidate_count": len(scored),
        "candidates": scored[:8],
    }

    base = load_base_motion()
    _run_traced(
        skill_trace,
        "whole_body_prepare_base",
        model,
        data,
        lambda step_callback=None: base.move_to_pose(
            model,
            data,
            [float(selected["base_x"]), float(selected["base_y"]), 0.0],
            steps=450,
            settle_steps=100,
            max_joint_step=0.003,
            fail_threshold=0.04,
            direct_qpos=bool(arm_profile.get("whole_body_prepare_direct_qpos", False)),
            step_callback=step_callback,
        ),
        planner_report=report,
    )
    torso = load_torso_height()
    _run_traced(
        skill_trace,
        "whole_body_prepare_torso",
        model,
        data,
        torso.execute_recovery_action,
        model,
        data,
        {
            "target_qpos": [0.0, 0.0, 0.0, float(selected["torso_yaw"])],
            "steps": 450,
            "settle_steps": 100,
            "max_joint_step": 0.003,
            "fail_threshold": 0.04,
            "direct_qpos": bool(arm_profile.get("whole_body_prepare_direct_qpos", False)),
            "lock_posture": True,
        },
        planner_report=report,
    )
    if bool(arm_profile.get("whole_body_prepare_arm_seed", False)):
        ik_skill = R1ProArmIKSkill()
        object_pos_after = _body_pos(model, data, object_body)
        target = object_pos_after + np.array(
            [0.0, 0.0, float(grasp_params.get("grasp_offset_z", DEFAULT_GRASP_OFFSET_Z)) + float(grasp_params.get("pregrasp_distance", DEFAULT_PREGRASP_DISTANCE))],
            dtype=np.float64,
        )
        q_seed = ik_skill.sync_q_from_mujoco(model, data)
        q_target, _ik_pos, ik_error = ik_skill.solve_position_ik(
            side,  # type: ignore[arg-type]
            q_seed,
            ik_skill.world_to_pinocchio_position(model, data, target),
            joint_names=ARM_JOINTS[side],  # type: ignore[index]
            posture_reference=q_seed,
            posture_gain=float(arm_profile.get("posture_gain", 0.0)),
        )
        q_values = ik_skill.arm_joint_values_from_pin_q(side, q_target, joint_names=ARM_JOINTS[side])  # type: ignore[arg-type,index]
        from skills.primitives.object_manipulation_skills import _move_joints_to_posture

        def run_arm_seed(*, step_callback=None) -> JointPosturePrepareResult:
            _final, error, success = _move_joints_to_posture(
                model,
                data,
                ARM_JOINTS[side],  # type: ignore[index]
                q_values,
                {
                    "steps": 900,
                    "settle_steps": 150,
                    "max_joint_step": 0.004,
                    "fail_threshold": 0.05,
                    "direct_qpos": bool(arm_profile.get("whole_body_prepare_direct_qpos", False)),
                },
                step_callback=step_callback,
            )
            return JointPosturePrepareResult(
                success=bool(success),
                final_error=float(error),
                message=f"arm_seed_joint_error={error:.6f}",
            )

        _run_traced(
            skill_trace,
            "whole_body_prepare_arm_seed",
            model,
            data,
            run_arm_seed,
            planner_report={**report, "arm_seed_ik_error": round(float(ik_error), 6), "arm_seed_target": _round_list(target)},
        )
    return report


def _run_pregrasp_whole_body_alignment_if_needed(
    skill_trace: list[dict[str, Any]],
    model: mujoco.MjModel,
    data: mujoco.MjData,
    control_mode: str,
    dynamics_profile: dict[str, Any],
    object_body: str,
    control_profile: dict[str, Any] | None = None,
    *,
    side: str = "left",
) -> None:
    if not _uses_physical_control(control_mode, dynamics_profile):
        return
    arm_profile = _physical_profile("arm", control_profile or _profile_from_dynamics(dynamics_profile))
    if bool(arm_profile.get("pregrasp_base_align", False)):
        base = load_base_motion()
        target = _base_align_target_for_object(model, data, object_body, side=side)
        _run_traced(
            skill_trace,
            "base_align_to_target",
            model,
            data,
            lambda step_callback=None: base.move_to_pose(
                model,
                data,
                target,
                **_base_motion_kwargs(control_mode, dynamics_profile, control_profile, steps=450, settle_steps=80),
                step_callback=step_callback,
            ),
            target=target,
        )
    if bool(arm_profile.get("pregrasp_torso_align", False)):
        torso = load_torso_height()
        torso_params = _torso_motion_params(control_mode, dynamics_profile, control_profile, height_level="mid", steps=450, settle_steps=80)
        if bool(arm_profile.get("pregrasp_torso_align_direct_qpos", False)):
            torso_params["direct_qpos"] = True
        _run_traced(
            skill_trace,
            "torso_set_height",
            model,
            data,
            torso.execute_recovery_action,
            model,
            data,
            torso_params,
            height_level="mid",
        )
        turn = load_torso_turn()
        turn_params = {
            "object_body": object_body,
            **torso_params,
            "max_abs_yaw": float(arm_profile.get("pregrasp_torso_max_yaw", 0.35)),
        }
        _run_traced(
            skill_trace,
            "torso_turn_to_target",
            model,
            data,
            turn.execute_recovery_action,
            model,
            data,
            turn_params,
            object_body=object_body,
        )


def _base_motion_kwargs(control_mode: str, dynamics_profile: dict[str, Any], control_profile: dict[str, Any] | None = None, *, steps: int = 1, settle_steps: int = 0) -> dict[str, Any]:
    if not _uses_physical_control(control_mode, dynamics_profile):
        return {"steps": steps, "settle_steps": settle_steps, "direct_qpos": False}
    profile = _physical_profile("base", control_profile or _profile_from_dynamics(dynamics_profile))
    profile.update({
        "control_mode": str(profile.get("control_mode") or "joint_servo"),
        "steps": max(steps, int(profile.get("steps", steps))),
        "settle_steps": max(settle_steps, int(profile.get("settle_steps", settle_steps))),
    })
    return {
        key: profile[key]
        for key in ("steps", "settle_steps", "max_joint_step", "fail_threshold", "direct_qpos")
        if key in profile
    }


def _torso_motion_params(control_mode: str, dynamics_profile: dict[str, Any], control_profile: dict[str, Any] | None = None, *, height_level: str, steps: int = 1, settle_steps: int = 0) -> dict[str, Any]:
    params = {
        "height_level": height_level,
        "steps": steps,
        "settle_steps": settle_steps,
        "direct_qpos": False,
    }
    if _uses_physical_control(control_mode, dynamics_profile):
        profile = _physical_profile("torso", control_profile or _profile_from_dynamics(dynamics_profile))
        profile.update({
            "height_level": height_level,
            "control_mode": str(profile.get("control_mode") or "joint_servo"),
            "steps": max(steps, int(profile.get("steps", steps))),
            "settle_steps": max(settle_steps, int(profile.get("settle_steps", settle_steps))),
        })
        params = profile
    return params


def _result_error(result: Any) -> float | None:
    for name in ("final_error", "left_error"):
        value = getattr(result, name, None)
        if value is not None:
            return float(value)
    return None


def _arm_result_from_result(result: Any) -> Any:
    arm_result = getattr(result, "arm_result", None)
    return arm_result if arm_result is not None else result


def _control_trace(result: Any) -> dict[str, Any]:
    arm_result = _arm_result_from_result(result)
    mode = getattr(arm_result, "control_mode", None)
    if mode is None:
        return {}
    command = getattr(arm_result, "motor_control_like_command", None)
    command_summary: dict[str, Any] = {}
    if isinstance(command, dict):
        command_summary = {
            "joint_names": list(command.get("joint_names") or []),
            "mode": str(command.get("mode") or ""),
            "p_des_count": len(command.get("p_des") or []),
            "kp_count": len(command.get("kp") or []),
            "kd_count": len(command.get("kd") or []),
        }
    joint_names = list(command.get("joint_names") or []) if isinstance(command, dict) else []
    q_start = getattr(arm_result, "q_start", None)
    q_final = getattr(arm_result, "q_final", None)
    q_target = None
    if isinstance(command, dict) and command.get("p_des") is not None:
        q_target = command.get("p_des")
    joint_tracking: list[dict[str, Any]] = []
    if joint_names and q_final is not None and q_target is not None:
        q_final_arr = np.asarray(q_final, dtype=np.float64).reshape(-1)
        q_target_arr = np.asarray(q_target, dtype=np.float64).reshape(-1)
        q_start_arr = np.asarray(q_start, dtype=np.float64).reshape(-1) if q_start is not None else np.full_like(q_final_arr, np.nan)
        for index, name in enumerate(joint_names[: min(len(joint_names), len(q_final_arr), len(q_target_arr))]):
            joint_tracking.append({
                "joint": name,
                "q_start": round(float(q_start_arr[index]), 6) if index < len(q_start_arr) and np.isfinite(q_start_arr[index]) else None,
                "q_target": round(float(q_target_arr[index]), 6),
                "q_final": round(float(q_final_arr[index]), 6),
                "tracking_error": round(float(abs(q_final_arr[index] - q_target_arr[index])), 6),
            })
    return {
        "control_mode": str(mode),
        "velocity_limit_applied": bool(getattr(arm_result, "velocity_limit_applied", False)),
        "commanded_steps": int(getattr(arm_result, "commanded_steps", 0) or 0),
        "motor_control_like_command": command_summary,
        "precheck": getattr(arm_result, "precheck", None) or {},
        "ik_error": round(float(getattr(arm_result, "ik_error", 0.0) or 0.0), 6),
        "final_error": round(float(getattr(arm_result, "final_error", 0.0) or 0.0), 6),
        "max_joint_tracking_error": round(float(getattr(arm_result, "max_joint_tracking_error", 0.0) or 0.0), 6),
        "mean_joint_tracking_error": round(float(getattr(arm_result, "mean_joint_tracking_error", 0.0) or 0.0), 6),
        "joint_tracking": joint_tracking,
    }


def _trace(skill: str, result: Any, **extra: Any) -> dict[str, Any]:
    item: dict[str, Any] = {
        "skill": skill,
        "success": bool(getattr(result, "success", False)),
        "message": str(getattr(result, "message", "")),
    }
    error = _result_error(result)
    if error is not None:
        item["error"] = round(error, 6)
    control = _control_trace(result)
    if control:
        item["control"] = control
    for key, value in extra.items():
        if isinstance(value, np.ndarray):
            item[key] = _round_list(value)
        elif isinstance(value, (list, tuple)):
            item[key] = json.loads(json.dumps(value, default=lambda obj: obj.tolist() if isinstance(obj, np.ndarray) else obj))
        else:
            item[key] = value
    return item


def _site_pos_optional(model: mujoco.MjModel, data: mujoco.MjData, site_name: str) -> list[float] | None:
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
    if site_id < 0:
        return None
    mujoco.mj_forward(model, data)
    return _round_list(data.site_xpos[site_id])


def _joint_limit_margin(model: mujoco.MjModel, data: mujoco.MjData) -> float | None:
    margins: list[float] = []
    for joint_id in range(model.njnt):
        if not bool(model.jnt_limited[joint_id]):
            continue
        if model.jnt_type[joint_id] not in {mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE}:
            continue
        qpos_id = int(model.jnt_qposadr[joint_id])
        low, high = model.jnt_range[joint_id]
        value = float(data.qpos[qpos_id])
        margins.append(min(value - float(low), float(high) - value))
    if not margins:
        return None
    return round(float(min(margins)), 6)


def _motion_snapshot(model: mujoco.MjModel, data: mujoco.MjData) -> dict[str, Any]:
    mujoco.mj_forward(model, data)
    tcp_positions = {
        name: pos
        for name in ("left_hand_tcp", "right_hand_tcp")
        if (pos := _site_pos_optional(model, data, name)) is not None
    }
    workspace_radius = 0.0
    if tcp_positions:
        workspace_radius = max(float(np.linalg.norm(np.asarray(pos[:2], dtype=np.float64))) for pos in tcp_positions.values())
    return {
        "qpos": data.qpos.copy(),
        "qvel": data.qvel.copy(),
        "contact_count": int(data.ncon),
        "joint_limit_margin_min": _joint_limit_margin(model, data),
        "tcp_positions": tcp_positions,
        "workspace_radius": round(workspace_radius, 6),
    }


def _motion_summary(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    qpos_delta = np.abs(np.asarray(after["qpos"]) - np.asarray(before["qpos"]))
    qvel_after = np.abs(np.asarray(after["qvel"]))
    tcp_deltas: dict[str, float] = {}
    for name, before_pos in before.get("tcp_positions", {}).items():
        after_pos = after.get("tcp_positions", {}).get(name)
        if after_pos is None:
            continue
        tcp_deltas[name] = round(float(np.linalg.norm(np.asarray(after_pos) - np.asarray(before_pos))), 6)
    joint_margin = after.get("joint_limit_margin_min")
    return {
        "max_joint_delta": round(float(np.max(qpos_delta)) if qpos_delta.size else 0.0, 6),
        "max_joint_speed_proxy": round(float(np.max(qvel_after)) if qvel_after.size else 0.0, 6),
        "joint_limit_margin_min": joint_margin,
        "contact_count_before": int(before.get("contact_count", 0)),
        "contact_count_after": int(after.get("contact_count", 0)),
        "contact_count_delta": int(after.get("contact_count", 0)) - int(before.get("contact_count", 0)),
        "tcp_delta": tcp_deltas,
        "max_tcp_delta": round(max(tcp_deltas.values()) if tcp_deltas else 0.0, 6),
        "workspace_radius_after": after.get("workspace_radius", 0.0),
    }


def _run_traced(
    skill_trace: list[dict[str, Any]],
    skill: str,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    fn: Any,
    *args: Any,
    trace_recorder: TrajectoryTraceRecorder | None = None,
    **extra: Any,
) -> Any:
    before = _motion_snapshot(model, data)
    if trace_recorder is None:
        trace_recorder = _TRACE_RECORDERS.get(id(data))
    viewer_context = _VIEWER_CONTEXTS.get(id(data))
    if trace_recorder is not None:
        trace_recorder.begin_skill(skill)
        trace_recorder.capture(model, data, force=True)
    if viewer_context is not None:
        viewer_context.sync(force=True)

    def on_step() -> None:
        if trace_recorder is not None:
            trace_recorder.capture(model, data)
        if viewer_context is not None:
            viewer_context.sync()

    def run_fn() -> Any:
        if trace_recorder is None and viewer_context is None:
            return fn(*args)
        try:
            return fn(*args, step_callback=on_step)
        except TypeError as exc:
            if "step_callback" not in str(exc):
                raise
            return fn(*args)

    delay_context = _DELAY_CONTEXTS.get(id(data))
    try:
        if delay_context is not None:
            with delay_context:
                result = run_fn()
        else:
            result = run_fn()
    finally:
        if trace_recorder is not None:
            trace_recorder.capture(model, data, force=True)
            trace_recorder.end_skill()
        if viewer_context is not None:
            viewer_context.sync(force=True)
    after = _motion_snapshot(model, data)
    skill_trace.append(_trace(skill, result, motion=_motion_summary(before, after), **extra))
    return result


def _motion_critic_metrics(skill_trace: list[dict[str, Any]]) -> dict[str, Any]:
    motions = [item.get("motion") for item in skill_trace if isinstance(item.get("motion"), dict)]
    if not motions:
        return {}
    joint_margins = [
        float(item["joint_limit_margin_min"])
        for item in motions
        if item.get("joint_limit_margin_min") is not None
    ]
    return {
        "max_joint_delta": max(float(item.get("max_joint_delta", 0.0)) for item in motions),
        "max_joint_speed_proxy": max(float(item.get("max_joint_speed_proxy", 0.0)) for item in motions),
        "min_joint_limit_margin": min(joint_margins) if joint_margins else None,
        "max_contact_count_after": max(int(item.get("contact_count_after", 0)) for item in motions),
        "max_contact_count_delta": max(abs(int(item.get("contact_count_delta", 0))) for item in motions),
        "max_tcp_delta": max(float(item.get("max_tcp_delta", 0.0)) for item in motions),
        "max_workspace_radius": max(float(item.get("workspace_radius_after", 0.0)) for item in motions),
    }


def _skill_motion(skill_trace: list[dict[str, Any]], skill_names: set[str]) -> dict[str, Any]:
    for item in skill_trace:
        if str(item.get("skill") or "") in skill_names and isinstance(item.get("motion"), dict):
            return item["motion"]
    return {}


def _contact_stability_metrics(
    skill_trace: list[dict[str, Any]],
    *,
    object_start: np.ndarray,
    object_after_close: np.ndarray,
    object_after_lift: np.ndarray,
    object_final: np.ndarray,
    attach_mode: str,
) -> dict[str, Any]:
    close_motion = _skill_motion(skill_trace, {"left_gripper_close", "right_gripper_close"})
    lift_motion = _skill_motion(skill_trace, {"left_vertical_lift", "right_vertical_lift"})
    transport_motion = _skill_motion(skill_trace, {"safe_transport_pose"})
    close_contact = int(close_motion.get("contact_count_after", 0)) if close_motion else 0
    lift_contact = int(lift_motion.get("contact_count_after", 0)) if lift_motion else 0
    transport_contact = int(transport_motion.get("contact_count_after", lift_contact)) if transport_motion else lift_contact
    contact_samples = [value for value in (close_contact, lift_contact, transport_contact) if value >= 0]
    active_contact_samples = sum(1 for value in contact_samples if value > 0)
    contact_during_lift_ratio = active_contact_samples / max(len(contact_samples), 1)

    lift_slip = float(np.linalg.norm((object_after_lift - object_after_close)[:2]))
    total_slip = float(np.linalg.norm((object_final - object_after_close)[:2]))
    vertical_lift = float(object_after_lift[2] - object_start[2])
    force_proxy = max(
        abs(int(close_motion.get("contact_count_delta", 0))) if close_motion else 0,
        abs(int(lift_motion.get("contact_count_delta", 0))) if lift_motion else 0,
        abs(int(transport_motion.get("contact_count_delta", 0))) if transport_motion else 0,
    )
    normalized_slip = min(lift_slip / 0.05, 1.0)
    normalized_force = min(force_proxy / 20.0, 1.0)
    contact_after_close = close_contact > 0 or "hand" in attach_mode
    contact_lost_step = -1
    if close_contact > 0 and lift_contact <= 0:
        contact_lost_step = 1
    elif lift_contact > 0 and transport_contact <= 0:
        contact_lost_step = 2
    stability = (
        0.40 * contact_during_lift_ratio
        + 0.30 * (1.0 - normalized_slip)
        + 0.20 * (1.0 if contact_after_close else 0.0)
        + 0.10 * (1.0 - normalized_force)
    )
    return {
        "source": "mujoco_proxy",
        "attach_mode": attach_mode,
        "contact_after_close": bool(contact_after_close),
        "contact_count_after_close": close_contact,
        "contact_count_after_lift": lift_contact,
        "contact_count_after_transport": transport_contact,
        "contact_during_lift_ratio": round(float(contact_during_lift_ratio), 6),
        "contact_lost_step": contact_lost_step,
        "object_slip_distance": round(total_slip, 6),
        "object_lift_slip_distance": round(lift_slip, 6),
        "object_vertical_lift": round(vertical_lift, 6),
        "wrist_force_proxy": round(float(force_proxy), 6),
        "grasp_stability_score": round(max(0.0, min(stability, 1.0)), 6),
    }


def _failure_diagnosis_metrics(
    *,
    skill_trace: list[dict[str, Any]],
    failure_reason: str,
    success: bool,
    task_success: bool,
    motion_critic: dict[str, Any],
    contact_stability: dict[str, Any],
) -> dict[str, Any]:
    failed_items = [
        item for item in skill_trace
        if not bool(item.get("success", False)) and str(item.get("skill") or "") != "detect_place_occupancy"
    ]
    failed_skill = str(failed_items[0].get("skill") or "") if failed_items else ""
    failed_control = failed_items[0].get("control") if failed_items and isinstance(failed_items[0].get("control"), dict) else {}
    failed_motion = failed_items[0].get("motion") if failed_items and isinstance(failed_items[0].get("motion"), dict) else {}

    def as_float(value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    ik_error = as_float(failed_control.get("ik_error"))
    final_error = as_float(failed_control.get("final_error"))
    tracking_error = as_float(failed_control.get("max_joint_tracking_error"))
    joint_margin = failed_motion.get("joint_limit_margin_min", motion_critic.get("min_joint_limit_margin"))
    workspace_radius = as_float(failed_motion.get("workspace_radius_after", motion_critic.get("max_workspace_radius")))
    max_tcp_delta = as_float(failed_motion.get("max_tcp_delta", motion_critic.get("max_tcp_delta")))
    contact_lost_step = int(contact_stability.get("contact_lost_step", -1) or -1) if contact_stability else -1
    slip_distance = as_float(contact_stability.get("object_slip_distance") if contact_stability else 0.0)
    grasp_score = as_float(contact_stability.get("grasp_stability_score") if contact_stability else 1.0, 1.0)
    joint_limit_violation = joint_margin is not None and as_float(joint_margin, 1.0) < 0.0
    primary_reason = "success"
    if not success or not task_success:
        if failed_skill:
            if ik_error > 0.08:
                primary_reason = "ik_unreachable_or_bad_target"
            elif tracking_error > 0.08:
                primary_reason = "actuator_tracking_error"
            elif joint_limit_violation:
                primary_reason = "joint_limit_violation"
            elif workspace_radius > 1.2:
                primary_reason = "workspace_exceeded"
            elif contact_lost_step >= 0:
                primary_reason = "contact_lost"
            elif slip_distance > 0.05:
                primary_reason = "object_slip"
            elif grasp_score < 0.45:
                primary_reason = "weak_grasp_stability"
            else:
                primary_reason = "skill_failed"
        elif not task_success:
            primary_reason = "task_verification_failed"
        else:
            primary_reason = str(failure_reason or "unknown_failure")
    return {
        "primary_reason": primary_reason,
        "failure_reason": str(failure_reason or ""),
        "failed_skill": failed_skill,
        "failed_skill_count": len(failed_items),
        "failed_skills": [str(item.get("skill") or "") for item in failed_items],
        "ik_error": round(ik_error, 6),
        "final_error": round(final_error, 6),
        "max_joint_tracking_error": round(tracking_error, 6),
        "joint_limit_violation": bool(joint_limit_violation),
        "joint_limit_margin_min": joint_margin,
        "workspace_radius_after": round(workspace_radius, 6),
        "max_tcp_delta": round(max_tcp_delta, 6),
        "contact_lost": contact_lost_step >= 0,
        "contact_lost_step": contact_lost_step,
        "object_slip_distance": round(slip_distance, 6),
        "grasp_stability_score": round(grasp_score, 6),
    }


def _attach_failure_diagnosis(
    metrics: dict[str, Any],
    *,
    skill_trace: list[dict[str, Any]],
    failure_reason: str,
    success: bool,
    task_success: bool,
) -> dict[str, Any]:
    updated = dict(metrics)
    motion_critic = updated.get("motion_critic") if isinstance(updated.get("motion_critic"), dict) else {}
    contact_stability = updated.get("contact_stability") if isinstance(updated.get("contact_stability"), dict) else {}
    updated["failure_diagnosis"] = _failure_diagnosis_metrics(
        skill_trace=skill_trace,
        failure_reason=failure_reason,
        success=success,
        task_success=task_success,
        motion_critic=motion_critic,
        contact_stability=contact_stability,
    )
    return updated


def _base_pose_for_place(site_xyz: np.ndarray) -> np.ndarray:
    return np.array([-0.25, float(site_xyz[1]), 0.0], dtype=np.float64)


def _load_model(model_path: str) -> tuple[mujoco.MjModel, mujoco.MjData]:
    model = mujoco.MjModel.from_xml_path(str(ROOT / model_path if not Path(model_path).is_absolute() else model_path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


class _PassiveViewerContext:
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, *, sync_every: int = 5) -> None:
        self.model = model
        self.data = data
        self.sync_every = max(int(sync_every), 1)
        self._viewer: Any | None = None
        self._step_count = 0

    def __enter__(self) -> "_PassiveViewerContext":
        from mujoco import viewer as mj_viewer

        self._viewer = mj_viewer.launch_passive(self.model, self.data)
        _VIEWER_CONTEXTS[id(self.data)] = self
        self.sync(force=True)
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        _VIEWER_CONTEXTS.pop(id(self.data), None)
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None

    def sync(self, *, force: bool = False) -> None:
        if self._viewer is None:
            return
        if not self._viewer.is_running():
            return
        self._step_count += 1
        if force or self._step_count % self.sync_every == 0:
            self._viewer.sync()


def _start_viewer_if_requested(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    enabled: bool,
    sync_every: int,
) -> _PassiveViewerContext | None:
    if not enabled:
        return None
    context = _PassiveViewerContext(model, data, sync_every=sync_every)
    context.__enter__()
    return context


def _close_viewer(context: _PassiveViewerContext | None) -> None:
    if context is not None:
        context.__exit__(None, None, None)


class KeyframeRecorder:
    def __init__(self, output_dir: Path | None) -> None:
        self.output_dir = output_dir
        self.keyframes: list[dict[str, Any]] = []
        self._renderer: mujoco.Renderer | None = None

    def capture(self, model: mujoco.MjModel, data: mujoco.MjData, stage: str, *, description: str = "") -> None:
        if self.output_dir is None:
            return
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if self._renderer is None:
            self._renderer = mujoco.Renderer(model, height=480, width=640)
        mujoco.mj_forward(model, data)
        self._renderer.update_scene(data)
        image = self._renderer.render()
        path = self.output_dir / f"{stage}.png"
        Image.fromarray(image).save(path)
        self.keyframes.append({
            "stage": stage,
            "image_path": str(path.resolve()),
            "description": description or stage,
            "used_for_retrieval": True,
        })

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None


class TrajectoryTraceRecorder:
    def __init__(
        self,
        output_dir: Path | None,
        *,
        object_body: str,
        sample_every: int = 10,
        max_samples: int = 2000,
    ) -> None:
        self.output_dir = output_dir
        self.object_body = object_body
        self.sample_every = max(1, int(sample_every))
        self.max_samples = max(1, int(max_samples))
        self.step_index = 0
        self.samples: list[dict[str, Any]] = []
        self.current_skill = ""
        self.trace_path: Path | None = None
        self.summary_path: Path | None = None

    def begin_skill(self, skill: str) -> None:
        self.current_skill = skill

    def end_skill(self) -> None:
        self.current_skill = ""

    def capture(self, model: mujoco.MjModel, data: mujoco.MjData, *, force: bool = False) -> None:
        self.step_index += 1
        if self.output_dir is None:
            return
        if not force and self.step_index % self.sample_every != 0:
            return
        if len(self.samples) >= self.max_samples:
            return
        mujoco.mj_forward(model, data)
        object_pose: dict[str, Any] = {}
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, self.object_body)
        if body_id >= 0:
            object_pose = {
                "body_name": self.object_body,
                "position": _round_list(data.xpos[body_id]),
                "quaternion_wxyz": _round_list(data.xquat[body_id]),
                "linear_velocity": _round_list(data.cvel[body_id][3:6]),
                "angular_velocity": _round_list(data.cvel[body_id][0:3]),
            }
        contact_pairs: list[dict[str, Any]] = []
        for index in range(min(int(data.ncon), 12)):
            contact = data.contact[index]
            contact_pairs.append({
                "geom1": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(contact.geom1)) or "",
                "geom2": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(contact.geom2)) or "",
                "distance": round(float(contact.dist), 6),
            })
        tcp_positions = {
            name: pos
            for name in ("left_hand_tcp", "right_hand_tcp")
            if (pos := _site_pos_optional(model, data, name)) is not None
        }
        self.samples.append({
            "sample_index": len(self.samples),
            "step_index": self.step_index,
            "time": round(float(data.time), 6),
            "skill": self.current_skill,
            "qpos": _round_list(data.qpos),
            "qvel": _round_list(data.qvel),
            "ctrl": _round_list(data.ctrl),
            "ee_pose": tcp_positions,
            "object_pose": object_pose,
            "contact_count": int(data.ncon),
            "contact_pairs": contact_pairs,
            "joint_limit_margin_min": _joint_limit_margin(model, data),
        })

    def write(self, model: mujoco.MjModel, data: mujoco.MjData) -> dict[str, Any]:
        if self.output_dir is None:
            return {
                "enabled": False,
                "trace_path": "",
                "summary_path": "",
                "sample_count": 0,
            }
        self.capture(model, data, force=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.trace_path = self.output_dir / "trace.jsonl"
        self.summary_path = self.output_dir / "summary.json"
        with self.trace_path.open("w", encoding="utf-8") as handle:
            for sample in self.samples:
                handle.write(json.dumps(sample, ensure_ascii=False) + "\n")
        contact_counts = [int(sample.get("contact_count", 0)) for sample in self.samples]
        joint_margins = [
            float(sample["joint_limit_margin_min"])
            for sample in self.samples
            if sample.get("joint_limit_margin_min") is not None
        ]
        object_positions = [
            sample.get("object_pose", {}).get("position")
            for sample in self.samples
            if isinstance(sample.get("object_pose"), dict) and sample.get("object_pose", {}).get("position")
        ]
        object_path_length = 0.0
        for before, after in zip(object_positions, object_positions[1:]):
            object_path_length += float(np.linalg.norm(np.asarray(after, dtype=np.float64) - np.asarray(before, dtype=np.float64)))
        summary = {
            "enabled": True,
            "trace_path": str(self.trace_path.resolve()),
            "summary_path": str(self.summary_path.resolve()),
            "sample_count": len(self.samples),
            "step_count_observed": self.step_index,
            "sample_every": self.sample_every,
            "object_body": self.object_body,
            "max_contact_count": max(contact_counts) if contact_counts else 0,
            "min_contact_count": min(contact_counts) if contact_counts else 0,
            "min_joint_limit_margin": round(min(joint_margins), 6) if joint_margins else None,
            "object_path_length": round(object_path_length, 6),
            "first_skill": str(self.samples[0].get("skill") or "") if self.samples else "",
            "last_skill": str(self.samples[-1].get("skill") or "") if self.samples else "",
        }
        self.summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        return summary


def run_g3(
    condition_id: str,
    *,
    model_path: str,
    control_mode: str,
    candidate_id: str = "g3_default",
    grasp_side: str = "left",
    control_profile: str = DEFAULT_PHYSICAL_CONTROL_PROFILE_ID,
    keyframe_dir: Path | None = None,
    trace_dir: Path | None = None,
    sandbox_calibration: dict[str, Any] | None = None,
    sandbox_initial_state: dict[str, Any] | None = None,
    viewer: bool = False,
    viewer_sync_every: int = 5,
    stop_after_skill: str = "",
    viewer_hold_seconds: float = 0.0,
) -> TaskChainResult:
    control_mode = _normalize_control_mode(control_mode)
    if candidate_id not in {"", "g3_default", "g3_place_first", "g3_cautious_place"}:
        raise ValueError(f"Unsupported G3 candidate_id: {candidate_id}")
    candidate_id = candidate_id or "g3_default"
    if grasp_side not in {"left", "right"}:
        raise ValueError(f"Unsupported G3 grasp_side: {grasp_side}")

    model, data = _load_model(model_path)
    viewer_context = _start_viewer_if_requested(model, data, enabled=viewer, sync_every=viewer_sync_every)
    recorder = KeyframeRecorder(keyframe_dir)
    target_name = _first_existing_body(model, ["target_cube", "target_object", "object_0"])
    trace_recorder = TrajectoryTraceRecorder(trace_dir, object_body=target_name)
    object_bodies = [
        name for name in ("target_cube", "target_object", "object_0", "distractor_cylinder", "distractor_box")
        if _first_existing_body(model, [name], required=False)
    ]
    obstacle_body = _first_existing_body(model, ["place_obstacle_body", "observed_obstacle_0", "obstacle_0"], required=False)
    occupancy_bodies = [name for name in (obstacle_body, target_name) if name]
    primary_place_site = _first_existing_site(model, ["place_zone_site", "primary_place_zone_site"], required=False)
    alternate_place_site = _first_existing_site(model, ["alternate_place_zone_site"], required=False)
    primary_place_site = _first_existing_site(model, ["place_zone_site", "primary_place_zone_site"], required=False)
    alternate_place_site = _first_existing_site(model, ["alternate_place_zone_site"], required=False)
    if condition_id == "clean":
        if obstacle_body:
            _set_freejoint_body_pose(model, data, obstacle_body, [0.54, 0.32, 0.895])
    elif condition_id != "place_occupied":
        raise ValueError(f"Unsupported G3 condition: {condition_id}")
    calibration_report = _apply_object_pose_calibration(model, data, target_name, sandbox_calibration)
    object_body_map: dict[str, list[str]] = {
        target_name: [target_name, "target_object"],
    }
    if obstacle_body:
        object_body_map[obstacle_body] = [obstacle_body, "place_obstacle_body", "place_obstacle", "obstacle", "observed_obstacle_0"]
    initial_state_report = _apply_sandbox_initial_state(
        model,
        data,
        sandbox_initial_state,
        object_body_map=object_body_map,
    )
    dynamics_profile = _apply_runtime_dynamics_profile(
        model,
        data,
        body_name=target_name,
        sandbox_initial_state=sandbox_initial_state,
    )
    dynamics_profile["control_profile_id"] = control_profile
    delay_context = _register_actuation_delay_context(data, dynamics_profile)
    _TRACE_RECORDERS[id(data)] = trace_recorder
    recorder.capture(model, data, "before_task", description="G3 scene before task chain")

    skill_trace: list[dict[str, Any]] = []

    def _hold_viewer_if_requested() -> None:
        if not viewer or viewer_context is None:
            return
        hold_seconds = max(0.0, float(viewer_hold_seconds))
        if hold_seconds <= 0.0:
            return
        end_time = time.monotonic() + hold_seconds
        while time.monotonic() < end_time:
            viewer_context.sync(force=True)
            time.sleep(min(0.05, end_time - time.monotonic()))

    def _finalize_early(failure_reason: str = "") -> TaskChainResult:
        nonlocal dynamics_profile
        recorder.close()
        object_final = _body_pos(model, data, target_body)
        dynamics_profile = _merge_actuation_delay_report(dynamics_profile, delay_context)
        _unregister_actuation_delay_context(data)
        _TRACE_RECORDERS.pop(id(data), None)
        trace_summary = trace_recorder.write(model, data)
        _hold_viewer_if_requested()
        _close_viewer(viewer_context)
        return TaskChainResult(
            scenario_id="G3",
            condition_id=condition_id,
            control_mode=control_mode,
            model_path=model_path,
            success=True,
            task_success=False,
            selected_place_site="",
            target_object=target_body,
            object_start=_round_list(object_start),
            object_final=_round_list(object_final),
            skill_trace=skill_trace,
            metrics=_attach_failure_diagnosis(
                {
                "place_occupied": False,
                "object_lift": round(float(object_final[2] - object_start[2]), 6),
                "attach_mode": "single_hand",
                "candidate_id": candidate_id,
                "early_stop_skill": stop_after_skill,
                "keyframe_dir": str(keyframe_dir.resolve()) if keyframe_dir is not None else "",
                "motion_critic": _motion_critic_metrics(skill_trace),
                "contact_stability": {},
                "control_execution": _control_execution_report(control_mode, dynamics_profile, _profile_from_dynamics(dynamics_profile)),
                "sandbox_calibration": calibration_report,
                "sandbox_initial_state": initial_state_report,
                "sandbox_dynamics_profile": dynamics_profile,
                "trajectory_trace": trace_summary,
                },
                skill_trace=skill_trace,
                failure_reason=failure_reason,
                success=True,
                task_success=False,
            ),
            keyframes=recorder.keyframes,
            failure_reason=failure_reason,
        )
    detect = _run_traced(
        skill_trace,
        "detect_multiple_objects",
        model,
        data,
        load_detect_multiple().execute_recovery_action,
        model,
        data,
        {"object_bodies": object_bodies},
        object_count=0,
    )
    skill_trace[-1]["object_count"] = len(getattr(detect, "objects", []) or [])

    selected = _run_traced(
        skill_trace,
        "select_correct_object",
        model,
        data,
        load_select_correct().execute_recovery_action,
        model,
        data,
        {"objects": detect.objects, "target_name": target_name, "require_unique": True},
    )
    target_body = selected.selected_object.body_name if selected.selected_object is not None else ""
    skill_trace[-1]["selected_object"] = target_body
    if not target_body:
        raise RuntimeError("G3 target object selection failed")

    object_start = _body_pos(model, data, target_body)
    grasp_params = {
        "side": grasp_side,
        "object_body": target_body,
        "approach_dx": 0.0,
        "approach_dy": 0.0,
        "approach_dz": -1.0,
        "pregrasp_distance": DEFAULT_PREGRASP_DISTANCE,
        "grasp_offset_x": 0.0,
        "grasp_offset_y": 0.0,
        "grasp_offset_z": DEFAULT_GRASP_OFFSET_Z,
        "steps": 300,
        "settle_steps": 500,
        "fail_threshold": 0.02,
        **_control_overrides(control_mode, dynamics_profile),
    }

    selected_site = ""
    place: np.ndarray | None = None
    occupancy = None
    chosen = None
    if candidate_id == "g3_place_first":
        occupancy = _run_traced(
            skill_trace,
            "detect_place_occupancy",
            model,
            data,
            load_detect_occupancy().execute_recovery_action,
            model,
            data,
            {
                "place_site": primary_place_site or "place_zone_site",
                "candidate_bodies": occupancy_bodies,
                "exclude_bodies": [target_body],
            },
            occupied=False,
            objects=[],
        )
        skill_trace[-1]["occupied"] = bool(occupancy.occupied)
        skill_trace[-1]["objects"] = list(occupancy.occupied_objects)
        chosen = _run_traced(
            skill_trace,
            "choose_alternate_place",
            model,
            data,
            load_choose_place().execute_recovery_action,
            model,
            data,
            {
                "place_sites": [site for site in (primary_place_site, alternate_place_site) if site],
                "place_site": primary_place_site or "place_zone_site",
                "alternate_place_site": alternate_place_site or "alternate_place_zone_site",
                "candidate_bodies": occupancy_bodies,
                "exclude_bodies": [target_body],
            },
            selected_site="",
        )
        selected_site = chosen.selected_site or ""
        skill_trace[-1]["selected_site"] = selected_site
        if not selected_site:
            raise RuntimeError("G3 place selection failed")
        place = _site_pos(model, data, selected_site)

    _run_whole_body_prepare_grasp_if_needed(
        skill_trace,
        model,
        data,
        control_mode,
        dynamics_profile,
        target_body,
        grasp_params,
        _profile_from_dynamics(dynamics_profile),
        side=grasp_side,
    )
    _run_pregrasp_whole_body_alignment_if_needed(
        skill_trace,
        model,
        data,
        control_mode,
        dynamics_profile,
        target_body,
        _profile_from_dynamics(dynamics_profile),
        side=grasp_side,
    )
    _run_pregrasp_safe_posture_if_needed(skill_trace, model, data, control_mode, dynamics_profile, _profile_from_dynamics(dynamics_profile))
    pregrasp = _run_traced(skill_trace, "move_to_pregrasp", model, data, load_pregrasp().execute_recovery_action, model, data, grasp_params)
    if stop_after_skill == "move_to_pregrasp":
        return _finalize_early()
    if not bool(getattr(pregrasp, "success", False)):
        recorder.close()
        object_final = _body_pos(model, data, target_body)
        dynamics_profile = _merge_actuation_delay_report(dynamics_profile, delay_context)
        _unregister_actuation_delay_context(data)
        _TRACE_RECORDERS.pop(id(data), None)
        trace_summary = trace_recorder.write(model, data)
        _close_viewer(viewer_context)
        return TaskChainResult(
            scenario_id="G3",
            condition_id=condition_id,
            control_mode=control_mode,
            model_path=model_path,
            success=False,
            task_success=False,
            selected_place_site="",
            target_object=target_body,
            object_start=_round_list(object_start),
            object_final=_round_list(object_final),
            skill_trace=skill_trace,
            metrics=_attach_failure_diagnosis(
                {
                    "place_occupied": False,
                    "object_lift": round(float(object_final[2] - object_start[2]), 6),
                    "attach_mode": "single_hand",
                    "candidate_id": candidate_id,
                    "keyframe_dir": str(keyframe_dir.resolve()) if keyframe_dir is not None else "",
                    "motion_critic": _motion_critic_metrics(skill_trace),
                    "contact_stability": {},
                    "control_execution": _control_execution_report(control_mode, dynamics_profile, _profile_from_dynamics(dynamics_profile)),
                    "sandbox_calibration": calibration_report,
                    "sandbox_initial_state": initial_state_report,
                    "sandbox_dynamics_profile": dynamics_profile,
                    "trajectory_trace": trace_summary,
                },
                skill_trace=skill_trace,
                failure_reason="move_to_pregrasp",
                success=False,
                task_success=False,
            ),
        keyframes=recorder.keyframes,
        failure_reason="move_to_pregrasp",
    )
    approach = _run_traced(skill_trace, "approach_object", model, data, load_approach().execute_recovery_action, model, data, grasp_params)
    if stop_after_skill == "approach_object":
        return _finalize_early()
    if not bool(getattr(approach, "success", False)):
        recorder.close()
        object_final = _body_pos(model, data, target_body)
        dynamics_profile = _merge_actuation_delay_report(dynamics_profile, delay_context)
        _unregister_actuation_delay_context(data)
        _TRACE_RECORDERS.pop(id(data), None)
        trace_summary = trace_recorder.write(model, data)
        _close_viewer(viewer_context)
        return TaskChainResult(
            scenario_id="G3",
            condition_id=condition_id,
            control_mode=control_mode,
            model_path=model_path,
            success=False,
            task_success=False,
            selected_place_site="",
            target_object=target_body,
            object_start=_round_list(object_start),
            object_final=_round_list(object_final),
            skill_trace=skill_trace,
            metrics=_attach_failure_diagnosis(
                {
                    "place_occupied": False,
                    "object_lift": round(float(object_final[2] - object_start[2]), 6),
                    "attach_mode": "single_hand",
                    "candidate_id": candidate_id,
                    "keyframe_dir": str(keyframe_dir.resolve()) if keyframe_dir is not None else "",
                    "motion_critic": _motion_critic_metrics(skill_trace),
                    "contact_stability": {},
                    "control_execution": _control_execution_report(control_mode, dynamics_profile, _profile_from_dynamics(dynamics_profile)),
                    "sandbox_calibration": calibration_report,
                    "sandbox_initial_state": initial_state_report,
                    "sandbox_dynamics_profile": dynamics_profile,
                    "trajectory_trace": trace_summary,
                },
                skill_trace=skill_trace,
                failure_reason="approach_object",
                success=False,
                task_success=False,
            ),
            keyframes=recorder.keyframes,
            failure_reason="approach_object",
        )
    closed = _run_traced(
        skill_trace,
        _side_skill_name(grasp_side, "gripper_close"),
        model,
        data,
        _load_gripper_close(grasp_side).execute_recovery_action,
        model,
        data,
        {
            "object_body": target_body,
            "gripper_steps": 240,
            "closure_bias": dynamics_profile["gripper_closure_bias"],
            "attach_on_close": False,
            **_gripper_overrides(control_mode, dynamics_profile),
        },
        attached_object=target_body,
    )
    if stop_after_skill == _side_skill_name(grasp_side, "gripper_close"):
        return _finalize_early()
    recorder.capture(model, data, "after_grasp", description=f"G3 after {grasp_side} gripper close")
    object_after_close = _body_pos(model, data, target_body)
    if candidate_id == "g3_cautious_place":
        _run_traced(
            skill_trace,
            "verify_grasp_contact",
            model,
            data,
            load_verify_grasp().execute_recovery_action,
            model,
            data,
            {
                "side": grasp_side,
                "object_body": target_body,
                "max_grasp_distance": 0.08,
                "min_lift": 0.0,
            },
        )
    lifted = _run_traced(
        skill_trace,
        _side_skill_name(grasp_side, "vertical_lift"),
        model,
        data,
        _load_vertical_lift(grasp_side).execute_recovery_action,
        model,
        data,
        {
            "lift_dx": 0.0,
            "lift_dy": 0.0,
            "lift_dz": 0.18,
            "steps": 900,
            "settle_steps": 100,
            "fail_threshold": 0.03,
            "lift_tolerance": 0.03,
            **_control_overrides(control_mode, dynamics_profile),
        },
    )
    if stop_after_skill == _side_skill_name(grasp_side, "vertical_lift"):
        return _finalize_early()
    recorder.capture(model, data, "after_lift", description="G3 after vertical lift")
    object_after_lift = _body_pos(model, data, target_body)
    if candidate_id == "g3_cautious_place":
        _run_traced(
            skill_trace,
            "verify_grasp_after_lift",
            model,
            data,
            load_verify_grasp().execute_recovery_action,
            model,
            data,
            {
                "side": grasp_side,
                "object_body": target_body,
                "max_grasp_distance": 0.12,
                "min_lift": 0.03,
                "initial_object_z": object_start[2],
            },
        )

    if candidate_id != "g3_place_first":
        occupancy = _run_traced(
            skill_trace,
            "detect_place_occupancy",
            model,
            data,
            load_detect_occupancy().execute_recovery_action,
            model,
            data,
            {
                "place_site": primary_place_site or "place_zone_site",
                "candidate_bodies": occupancy_bodies,
                "exclude_bodies": [target_body],
            },
            occupied=False,
            objects=[],
        )
        skill_trace[-1]["occupied"] = bool(occupancy.occupied)
        skill_trace[-1]["objects"] = list(occupancy.occupied_objects)
        chosen = _run_traced(
            skill_trace,
            "choose_alternate_place",
            model,
            data,
            load_choose_place().execute_recovery_action,
            model,
            data,
            {
                "place_sites": [site for site in (primary_place_site, alternate_place_site) if site],
                "place_site": primary_place_site or "place_zone_site",
                "alternate_place_site": alternate_place_site or "alternate_place_zone_site",
                "candidate_bodies": occupancy_bodies,
                "exclude_bodies": [target_body],
            },
            selected_site="",
        )
        selected_site = chosen.selected_site or ""
        skill_trace[-1]["selected_site"] = selected_site
        if not selected_site:
            raise RuntimeError("G3 place selection failed")
        place = _site_pos(model, data, selected_site)
    if occupancy is None or place is None:
        raise RuntimeError("G3 place selection did not produce occupancy/place")
    recorder.capture(model, data, "before_place", description="G3 before placing object")

    placed = _run_traced(
        skill_trace,
        "place_object",
        model,
        data,
        load_place().execute_recovery_action,
        model,
        data,
        {
                "side": grasp_side,
            "place_x": float(place[0]),
            "place_y": float(place[1]),
            "place_z": float(place[2]),
            "place_offset_x": 0.0,
            "place_offset_y": 0.0,
            "place_offset_z": 0.10,
            "steps": 900,
            "settle_steps": 100,
            "fail_threshold": 0.05,
            "orientation_weight": 0.0,
            **_control_overrides(control_mode, dynamics_profile),
        },
    )
    released = _run_traced(
        skill_trace,
        "open_gripper_release",
        model,
        data,
        load_release().execute_recovery_action,
        model,
        data,
        {"side": grasp_side, "gripper_steps": 60, "settle_steps": 20, **_gripper_overrides(control_mode, dynamics_profile)},
    )
    if stop_after_skill == "open_gripper_release":
        return _finalize_early()
    verified = _run_traced(
        skill_trace,
        "verify_place_zone",
        model,
        data,
        load_verify_place().execute_recovery_action,
        model,
        data,
        {
            "side": grasp_side,
            "object_body": target_body,
            "place_x": float(place[0]),
            "place_y": float(place[1]),
            "place_z": float(place[2]),
            "max_xy_error": 0.025,
            "max_z_error": 0.08,
        },
    )
    recorder.capture(model, data, "after_place", description="G3 after place verification")
    recorder.close()

    object_final = _body_pos(model, data, target_body)
    contact_stability = _contact_stability_metrics(
        skill_trace,
        object_start=object_start,
        object_after_close=object_after_close,
        object_after_lift=object_after_lift,
        object_final=object_final,
        attach_mode="single_hand",
    )
    failed = [item["skill"] for item in skill_trace if not item["success"] and item["skill"] != "detect_place_occupancy"]
    success = not failed
    dynamics_profile = _merge_actuation_delay_report(dynamics_profile, delay_context)
    _unregister_actuation_delay_context(data)
    _TRACE_RECORDERS.pop(id(data), None)
    trace_summary = trace_recorder.write(model, data)
    try:
        return TaskChainResult(
            scenario_id="G3",
            condition_id=condition_id,
            control_mode=control_mode,
            model_path=model_path,
            success=success,
            task_success=bool(verified.success),
            selected_place_site=selected_site,
            target_object=target_body,
            object_start=_round_list(object_start),
            object_final=_round_list(object_final),
            skill_trace=skill_trace,
            metrics=_attach_failure_diagnosis(
                {
                "place_occupied": bool(occupancy.occupied),
                "object_lift": round(float(object_final[2] - object_start[2]), 6),
                "attach_mode": "single_hand",
                "candidate_id": candidate_id,
                "keyframe_dir": str(keyframe_dir.resolve()) if keyframe_dir is not None else "",
                "motion_critic": _motion_critic_metrics(skill_trace),
                "contact_stability": contact_stability,
                "control_execution": _control_execution_report(control_mode, dynamics_profile, _profile_from_dynamics(dynamics_profile)),
                "sandbox_calibration": calibration_report,
                "sandbox_initial_state": initial_state_report,
                "sandbox_dynamics_profile": dynamics_profile,
                "trajectory_trace": trace_summary,
                },
                skill_trace=skill_trace,
                failure_reason=",".join(failed),
                success=success,
                task_success=bool(verified.success),
            ),
            keyframes=recorder.keyframes,
            failure_reason=",".join(failed),
        )
    finally:
        _close_viewer(viewer_context)


def run_task_chain(
    scenario: str,
    condition: str,
    control_mode: str,
    candidate_id: str = "",
    grasp_side: str = "left",
    control_profile: str = DEFAULT_PHYSICAL_CONTROL_PROFILE_ID,
    keyframe_dir: Path | None = None,
    trace_dir: Path | None = None,
    sandbox_calibration: dict[str, Any] | None = None,
    sandbox_initial_state: dict[str, Any] | None = None,
    model_path_override: str | None = None,
    viewer: bool = False,
    viewer_sync_every: int = 5,
    stop_after_skill: str = "",
    viewer_hold_seconds: float = 0.0,
) -> TaskChainResult:
    scenario = scenario.upper()
    if scenario == "G3":
        return run_g3(
            condition,
            model_path=model_path_override or "r1pro_g3_sorting_scene.xml",
            control_mode=control_mode,
            candidate_id=candidate_id or "g3_default",
            grasp_side=grasp_side,
            control_profile=control_profile,
            keyframe_dir=keyframe_dir,
            trace_dir=trace_dir,
            sandbox_calibration=sandbox_calibration,
            sandbox_initial_state=sandbox_initial_state,
            viewer=viewer,
            viewer_sync_every=viewer_sync_every,
            stop_after_skill=stop_after_skill,
            viewer_hold_seconds=viewer_hold_seconds,
    )
    raise ValueError(f"Unsupported scenario: {scenario}")


def _run_g3_plan_chain(
    condition_id: str,
    *,
    model_path: str,
    control_mode: str,
    plan_steps: list[str | dict[str, Any]],
    grasp_side: str = "left",
    control_profile: str = DEFAULT_PHYSICAL_CONTROL_PROFILE_ID,
    keyframe_dir: Path | None = None,
    trace_dir: Path | None = None,
    sandbox_calibration: dict[str, Any] | None = None,
    sandbox_initial_state: dict[str, Any] | None = None,
) -> TaskChainResult:
    model, data = _load_model(model_path)
    if grasp_side not in {"left", "right"}:
        raise ValueError(f"Unsupported G3 grasp_side: {grasp_side}")
    recorder = KeyframeRecorder(keyframe_dir)
    target_name = _first_existing_body(model, ["target_cube", "target_object", "object_0"])
    trace_recorder = TrajectoryTraceRecorder(trace_dir, object_body=target_name)
    object_bodies = [
        name for name in ("target_cube", "target_object", "object_0", "distractor_cylinder", "distractor_box")
        if _first_existing_body(model, [name], required=False)
    ]
    obstacle_body = _first_existing_body(model, ["place_obstacle_body", "observed_obstacle_0", "obstacle_0"], required=False)
    occupancy_bodies = [name for name in (obstacle_body, target_name) if name]
    primary_place_site = _first_existing_site(model, ["place_zone_site", "primary_place_zone_site"], required=False)
    alternate_place_site = _first_existing_site(model, ["alternate_place_zone_site"], required=False)
    if condition_id == "clean":
        if obstacle_body:
            _set_freejoint_body_pose(model, data, obstacle_body, [0.54, 0.32, 0.895])
    elif condition_id != "place_occupied":
        raise ValueError(f"Unsupported G3 condition: {condition_id}")
    calibration_report = _apply_object_pose_calibration(model, data, target_name, sandbox_calibration)
    object_body_map = {
        target_name: [target_name, "target_object"],
    }
    if obstacle_body:
        object_body_map[obstacle_body] = [obstacle_body, "place_obstacle_body", "place_obstacle", "obstacle", "observed_obstacle_0"]
    initial_state_report = _apply_sandbox_initial_state(
        model,
        data,
        sandbox_initial_state,
        object_body_map=object_body_map,
    )
    dynamics_profile = _apply_runtime_dynamics_profile(model, data, body_name=target_name, sandbox_initial_state=sandbox_initial_state)
    dynamics_profile["control_profile_id"] = control_profile
    delay_context = _register_actuation_delay_context(data, dynamics_profile)
    _TRACE_RECORDERS[id(data)] = trace_recorder
    recorder.capture(model, data, "before_task", description="G3 general plan before task")

    skill_trace: list[dict[str, Any]] = []
    target_body = ""
    selected_site = ""
    place: np.ndarray | None = None
    occupancy = None
    verified = None
    object_start = _body_pos(model, data, target_name)
    object_after_close = object_start.copy()
    object_after_lift = object_start.copy()

    def finish(success_override: bool | None = None, failure_reason: str = "") -> TaskChainResult:
        nonlocal dynamics_profile
        object_final = _body_pos(model, data, target_body or target_name)
        contact_stability = _contact_stability_metrics(
            skill_trace,
            object_start=object_start,
            object_after_close=object_after_close,
            object_after_lift=object_after_lift,
            object_final=object_final,
            attach_mode="single_hand",
        )
        failed = [item["skill"] for item in skill_trace if not item["success"] and item["skill"] != "detect_place_occupancy"]
        success = not failed if success_override is None else bool(success_override)
        dynamics_profile = _merge_actuation_delay_report(dynamics_profile, delay_context)
        _unregister_actuation_delay_context(data)
        _TRACE_RECORDERS.pop(id(data), None)
        trace_summary = trace_recorder.write(model, data)
        metrics = _attach_failure_diagnosis(
            {
                "place_occupied": bool(getattr(occupancy, "occupied", False)) if occupancy is not None else False,
                "object_lift": round(float(object_final[2] - object_start[2]), 6),
                "attach_mode": "single_hand",
                "candidate_id": "llm_general_plan",
                "general_plan_executor": {"enabled": True, "plan_steps": list(plan_steps)},
                "keyframe_dir": str(keyframe_dir.resolve()) if keyframe_dir is not None else "",
                "motion_critic": _motion_critic_metrics(skill_trace),
                "contact_stability": contact_stability,
                "control_execution": _control_execution_report(control_mode, dynamics_profile, _profile_from_dynamics(dynamics_profile)),
                "sandbox_calibration": calibration_report,
                "sandbox_initial_state": initial_state_report,
                "sandbox_dynamics_profile": dynamics_profile,
                "trajectory_trace": trace_summary,
            },
            skill_trace=skill_trace,
            failure_reason=failure_reason or ",".join(failed),
            success=success,
            task_success=bool(getattr(verified, "success", False)) if verified is not None else False,
        )
        return TaskChainResult(
            scenario_id="G3",
            condition_id=condition_id,
            control_mode=control_mode,
            model_path=model_path,
            success=success,
            task_success=bool(getattr(verified, "success", False)) if verified is not None else False,
            selected_place_site=selected_site,
            target_object=target_body or target_name,
            object_start=_round_list(object_start),
            object_final=_round_list(object_final),
            skill_trace=skill_trace,
            metrics=metrics,
            keyframes=recorder.keyframes,
            failure_reason=failure_reason or ",".join(failed),
        )

    try:
        for raw_step in plan_steps:
            if isinstance(raw_step, dict):
                step = str(raw_step.get("action") or "")
                step_parameters = raw_step.get("parameters") if isinstance(raw_step.get("parameters"), dict) else {}
            else:
                step = str(raw_step)
                step_parameters = {}
            if step == "detect_multiple_objects":
                detect = _run_traced(skill_trace, step, model, data, load_detect_multiple().execute_recovery_action, model, data, {"object_bodies": object_bodies})
                skill_trace[-1]["object_count"] = len(getattr(detect, "objects", []) or [])
            elif step == "select_correct_object":
                detect_objects = getattr(locals().get("detect", None), "objects", None)
                if detect_objects is None:
                    return finish(False, "select_correct_object requires detect_multiple_objects first")
                selected = _run_traced(skill_trace, step, model, data, load_select_correct().execute_recovery_action, model, data, {"objects": detect_objects, "target_name": target_name, "require_unique": True})
                target_body = selected.selected_object.body_name if selected.selected_object is not None else ""
                skill_trace[-1]["selected_object"] = target_body
                if not target_body:
                    return finish(False, "target object selection failed")
                object_start = _body_pos(model, data, target_body)
                object_after_close = object_start.copy()
                object_after_lift = object_start.copy()
            elif step in {"move_to_pregrasp", "approach_object"}:
                if not target_body:
                    return finish(False, f"{step} requires selected target object")
                params = {
                    "side": grasp_side,
                    "object_body": target_body,
                    "approach_dx": 0.0,
                    "approach_dy": 0.0,
                    "approach_dz": -1.0,
                    "pregrasp_distance": DEFAULT_PREGRASP_DISTANCE,
                    "grasp_offset_x": 0.0,
                    "grasp_offset_y": 0.0,
                    "grasp_offset_z": DEFAULT_GRASP_OFFSET_Z,
                    "steps": 300,
                    "settle_steps": 500,
                    "fail_threshold": 0.02,
                    **_control_overrides(control_mode, dynamics_profile),
                }
                params.update(step_parameters)
                loader = load_pregrasp if step == "move_to_pregrasp" else load_approach
                if step == "move_to_pregrasp":
                    _run_pregrasp_safe_posture_if_needed(skill_trace, model, data, control_mode, dynamics_profile, _profile_from_dynamics(dynamics_profile))
                _run_traced(skill_trace, step, model, data, loader().execute_recovery_action, model, data, params)
            elif step in {
                "reposition_base_for_reach",
                "adjust_torso_for_reach",
                "retry_pregrasp_with_safer_offset",
                "slow_cartesian_approach",
                "recover_from_joint_limit",
                "retry_lift_after_grasp_check",
            }:
                if not target_body:
                    return finish(False, f"{step} requires selected target object")
                params = {
                    "side": grasp_side,
                    "object_body": target_body,
                    "initial_object_z": object_start[2],
                    "approach_dx": 0.0,
                    "approach_dy": 0.0,
                    "approach_dz": -1.0,
                    "pregrasp_distance": DEFAULT_PREGRASP_DISTANCE,
                    "grasp_offset_x": 0.0,
                    "grasp_offset_y": 0.0,
                    "grasp_offset_z": DEFAULT_GRASP_OFFSET_Z,
                    **_control_overrides(control_mode, dynamics_profile),
                }
                params.update(step_parameters)
                loader_map = {
                    "reposition_base_for_reach": load_reposition_base_for_reach,
                    "adjust_torso_for_reach": load_adjust_torso_for_reach,
                    "retry_pregrasp_with_safer_offset": load_retry_pregrasp_with_safer_offset,
                    "slow_cartesian_approach": load_slow_cartesian_approach,
                    "recover_from_joint_limit": load_recover_from_joint_limit,
                    "retry_lift_after_grasp_check": load_retry_lift_after_grasp_check,
                }
                result = _run_traced(skill_trace, step, model, data, loader_map[step]().execute_recovery_action, model, data, params)
                skill_trace[-1]["composite_substeps"] = list(getattr(result, "substeps", []) or [])
                if step == "retry_lift_after_grasp_check":
                    object_after_lift = _body_pos(model, data, target_body)
            elif step in {"left_gripper_close", "right_gripper_close"}:
                if not target_body:
                    return finish(False, f"{step} requires selected target object")
                close_step = _side_skill_name(grasp_side, "gripper_close")
                _run_traced(skill_trace, close_step, model, data, _load_gripper_close(grasp_side).execute_recovery_action, model, data, {"object_body": target_body, "gripper_steps": 60, "closure_bias": dynamics_profile["gripper_closure_bias"], **_gripper_overrides(control_mode, dynamics_profile)}, attached_object=target_body)
                object_after_close = _body_pos(model, data, target_body)
            elif step in {"verify_grasp", "verify_grasp_contact", "verify_grasp_after_lift"}:
                if not target_body:
                    return finish(False, "verify_grasp requires selected target object")
                lifted_so_far = bool(np.linalg.norm(object_after_lift - object_start) > 1e-9)
                require_lift = step == "verify_grasp_after_lift" or (step == "verify_grasp" and lifted_so_far)
                verify_params = {
                    "side": grasp_side,
                    "object_body": target_body,
                    "max_grasp_distance": 0.12 if require_lift else 0.08,
                    "min_lift": 0.03 if require_lift else 0.0,
                }
                if require_lift:
                    verify_params["initial_object_z"] = object_start[2]
                verify_params.update(step_parameters)
                trace_name = "verify_grasp_after_lift" if require_lift and step == "verify_grasp" else step
                _run_traced(skill_trace, trace_name, model, data, load_verify_grasp().execute_recovery_action, model, data, verify_params)
            elif step in {"left_vertical_lift", "right_vertical_lift"}:
                if not target_body:
                    return finish(False, f"{step} requires selected target object")
                lift_step = _side_skill_name(grasp_side, "vertical_lift")
                _run_traced(skill_trace, lift_step, model, data, _load_vertical_lift(grasp_side).execute_recovery_action, model, data, {"lift_dx": 0.0, "lift_dy": 0.0, "lift_dz": 0.18, "steps": 900, "settle_steps": 100, "fail_threshold": 0.03, "lift_tolerance": 0.03, **_control_overrides(control_mode, dynamics_profile)})
                object_after_lift = _body_pos(model, data, target_body)
            elif step == "detect_place_occupancy":
                occupancy = _run_traced(
                    skill_trace,
                    step,
                    model,
                    data,
                    load_detect_occupancy().execute_recovery_action,
                    model,
                    data,
                    {
                        "place_site": primary_place_site or "place_zone_site",
                        "candidate_bodies": occupancy_bodies,
                        "exclude_bodies": [target_body or target_name],
                    },
                )
                skill_trace[-1]["occupied"] = bool(occupancy.occupied)
                skill_trace[-1]["objects"] = list(occupancy.occupied_objects)
            elif step == "choose_alternate_place":
                chosen = _run_traced(
                    skill_trace,
                    step,
                    model,
                    data,
                    load_choose_place().execute_recovery_action,
                    model,
                    data,
                    {
                        "place_sites": [site for site in (primary_place_site, alternate_place_site) if site],
                        "place_site": primary_place_site or "place_zone_site",
                        "alternate_place_site": alternate_place_site or "alternate_place_zone_site",
                        "candidate_bodies": occupancy_bodies,
                        "exclude_bodies": [target_body or target_name],
                    },
                    selected_site="",
                )
                selected_site = chosen.selected_site or ""
                skill_trace[-1]["selected_site"] = selected_site
                if not selected_site:
                    return finish(False, "choose_alternate_place did not return a selected site")
                place = _site_pos(model, data, selected_site)
            elif step == "place_object":
                if place is None:
                    return finish(False, "place_object requires choose_alternate_place first")
                _run_traced(skill_trace, step, model, data, load_place().execute_recovery_action, model, data, {"side": grasp_side, "place_x": float(place[0]), "place_y": float(place[1]), "place_z": float(place[2]), "place_offset_x": 0.0, "place_offset_y": 0.0, "place_offset_z": 0.10, "steps": 900, "settle_steps": 100, "fail_threshold": 0.05, "orientation_weight": 0.0, **_control_overrides(control_mode, dynamics_profile)})
            elif step == "open_gripper_release":
                _run_traced(skill_trace, step, model, data, load_release().execute_recovery_action, model, data, {"side": grasp_side, "gripper_steps": 60, "settle_steps": 20, **_gripper_overrides(control_mode, dynamics_profile)})
            elif step == "verify_place_zone":
                if place is None or not target_body:
                    return finish(False, "verify_place_zone requires target object and place site")
                verified = _run_traced(skill_trace, step, model, data, load_verify_place().execute_recovery_action, model, data, {"side": grasp_side, "object_body": target_body, "place_x": float(place[0]), "place_y": float(place[1]), "place_z": float(place[2]), "max_xy_error": 0.025, "max_z_error": 0.08})
            else:
                return finish(False, f"unsupported G3 plan step: {step}")
        recorder.close()
        return finish()
    except Exception as exc:
        recorder.close()
        return finish(False, f"general G3 plan executor failed: {exc}")


def _task_chain_failure_result(
    *,
    scenario: str,
    condition: str,
    control_mode: str,
    model_path: str,
    target_object: str,
    object_start: np.ndarray,
    object_final: np.ndarray,
    selected_place_site: str,
    skill_trace: list[dict[str, Any]],
    metrics: dict[str, Any],
    keyframes: list[dict[str, Any]],
    failure_reason: str,
) -> TaskChainResult:
    return TaskChainResult(
        scenario_id=scenario,
        condition_id=condition,
        control_mode=control_mode,
        model_path=model_path,
        success=False,
        task_success=False,
        selected_place_site=selected_place_site,
        target_object=target_object,
        object_start=_round_list(object_start),
        object_final=_round_list(object_final),
        skill_trace=skill_trace,
        metrics=metrics,
        keyframes=keyframes,
        failure_reason=failure_reason,
    )


def run_task_plan_chain(
    scenario: str,
    condition: str,
    control_mode: str,
    *,
    plan_steps: list[str | dict[str, Any]],
    grasp_side: str = "left",
    control_profile: str = DEFAULT_PHYSICAL_CONTROL_PROFILE_ID,
    keyframe_dir: Path | None = None,
    trace_dir: Path | None = None,
    sandbox_calibration: dict[str, Any] | None = None,
    sandbox_initial_state: dict[str, Any] | None = None,
    model_path_override: str | None = None,
) -> TaskChainResult:
    """Execute an LLM-proposed skill sequence directly in the benchmark sandbox."""

    scenario = scenario.upper()
    control_mode = _normalize_control_mode(control_mode)
    if scenario == "G3":
        return _run_g3_plan_chain(
            condition,
            model_path=model_path_override or "r1pro_g3_sorting_scene.xml",
            control_mode=control_mode,
            plan_steps=plan_steps,
            grasp_side=grasp_side,
            control_profile=control_profile,
            keyframe_dir=keyframe_dir,
            trace_dir=trace_dir,
            sandbox_calibration=sandbox_calibration,
            sandbox_initial_state=sandbox_initial_state,
        )
    raise ValueError(f"Unsupported scenario: {scenario}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run an R1Pro task chain and save structured experience output.")
    parser.add_argument("--scenario", choices=["G3"], required=True)
    parser.add_argument("--condition", choices=["clean", "place_occupied"], required=True)
    parser.add_argument("--control-mode", choices=["ideal", "physical"], default="physical")
    parser.add_argument("--control-profile", default=DEFAULT_PHYSICAL_CONTROL_PROFILE_ID, help="Physical control profile id, e.g. real_driver_like_v1 or site_servo_safe_v1")
    parser.add_argument("--candidate-id", default="", help="Optional executable candidate id, e.g. g3_cautious_place")
    parser.add_argument("--grasp-side", choices=["left", "right"], default="left", help="Single-arm side used by G3 grasp skills")
    parser.add_argument("--save", type=Path, default=None, help="Path for task-chain result JSON")
    parser.add_argument("--experience-lib", type=Path, default=None, help="Path for cumulative memory library JSON")
    parser.add_argument("--universal-experience-lib", type=Path, default=None, help="Path for universal experience library JSON")
    parser.add_argument("--no-write-policy", action="store_true", help="append universal entries without write-time gate")
    parser.add_argument("--keyframe-dir", type=Path, default=None, help="Directory for rendered keyframe png files")
    parser.add_argument("--trace-dir", type=Path, default=None, help="Directory for trajectory trace jsonl/summary files")
    parser.add_argument("--sandbox-initial-state", type=Path, default=None, help="Optional sandbox_initial_state_v1 JSON")
    parser.add_argument("--model-path", default="", help="Optional runtime MuJoCo XML path overriding the fixed benchmark scene")
    parser.add_argument("--viewer", action="store_true", help="Open a live MuJoCo passive viewer while executing the task chain")
    parser.add_argument("--viewer-sync-every", type=int, default=5, help="Sync the viewer every N skill substeps")
    parser.add_argument("--stop-after-skill", default="", choices=["", "move_to_pregrasp", "approach_object", "left_gripper_close", "right_gripper_close", "left_vertical_lift", "right_vertical_lift", "open_gripper_release"], help="Stop after the named skill finishes")
    parser.add_argument("--viewer-hold-seconds", type=float, default=0.0, help="Keep the viewer open for N seconds after early stop")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sandbox_initial_state = {}
    if args.sandbox_initial_state is not None:
        sandbox_initial_state = json.loads(args.sandbox_initial_state.read_text(encoding="utf-8"))
    result = run_task_chain(
        args.scenario,
        args.condition,
        args.control_mode,
        candidate_id=args.candidate_id,
        grasp_side=args.grasp_side,
        control_profile=args.control_profile,
        keyframe_dir=args.keyframe_dir,
        trace_dir=args.trace_dir,
        sandbox_initial_state=sandbox_initial_state,
        model_path_override=args.model_path or None,
        viewer=args.viewer,
        viewer_sync_every=args.viewer_sync_every,
        stop_after_skill=args.stop_after_skill,
        viewer_hold_seconds=args.viewer_hold_seconds,
    )
    entry = make_memory_entry(result)
    universal_entry = R1ProMujocoAdapter().normalize_episode(result)
    payload = {
        "result": result.to_dict(),
        "memory_entry": entry.to_dict(),
        "universal_experience_entry": universal_entry.to_dict(),
    }

    if args.save is not None:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        args.save.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    if args.experience_lib is not None:
        library = MemoryLibrary.load(args.experience_lib)
        library.add(entry)
        library.save(args.experience_lib)

    if args.universal_experience_lib is not None:
        universal_library = ExperienceLibrary.load(args.universal_experience_lib)
        if args.no_write_policy:
            universal_library.add(universal_entry)
            write_policy = {"decision": "write", "reason": "write_policy_disabled", "stored_experience_id": universal_entry.experience_id}
        else:
            write_policy = universal_library.add_with_policy(universal_entry)
        universal_library.save(args.universal_experience_lib)
    else:
        write_policy = {}

    print(json.dumps({
        "success": result.success,
        "task_success": result.task_success,
        "experience_id": getattr(entry, "experience_id", getattr(entry, "memory_id", "")),
        "universal_experience_id": universal_entry.experience_id,
        "stored_universal_experience_id": write_policy.get("stored_experience_id", ""),
        "write_policy": write_policy,
    }, ensure_ascii=False))
    if not result.success:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
