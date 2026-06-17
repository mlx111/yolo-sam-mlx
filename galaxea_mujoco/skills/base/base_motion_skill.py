from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import mujoco
import numpy as np

from skills.primitives.grasp_attachment import update_attachments


BASE_JOINTS = ("base_x", "base_y", "base_yaw")


@dataclass(frozen=True)
class BaseMotionResult:
    joint_names: tuple[str, ...]
    start_qpos: np.ndarray
    target_qpos: np.ndarray
    final_qpos: np.ndarray
    final_error: float
    success: bool


@dataclass(frozen=True)
class BaseMotionSkillConfig:
    name: str
    default_steps: int
    default_settle_steps: int
    default_max_joint_step: float
    default_fail_threshold: float
    default_direct_qpos: bool


class R1ProBaseMotionSkill:
    """Planar mobile-base control for base_x, base_y, and base_yaw joints."""

    joint_names = BASE_JOINTS

    def __init__(self, config: BaseMotionSkillConfig):
        self.config = config

    @classmethod
    def from_json(cls, path: str | Path) -> "R1ProBaseMotionSkill":
        payload = json.loads(Path(path).read_text())
        defaults = payload.get("control_defaults", {})
        return cls(
            BaseMotionSkillConfig(
                name=payload["name"],
                default_steps=int(defaults.get("steps", 900)),
                default_settle_steps=int(defaults.get("settle_steps", 120)),
                default_max_joint_step=float(defaults.get("max_joint_step", 0.01)),
                default_fail_threshold=float(defaults.get("fail_threshold", 0.02)),
                default_direct_qpos=bool(defaults.get("direct_qpos", False)),
            )
        )

    def _joint_qpos_indices(self, model: mujoco.MjModel) -> list[int]:
        indices = []
        for name in self.joint_names:
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid < 0:
                raise ValueError(f"MuJoCo joint not found: {name}")
            indices.append(int(model.jnt_qposadr[jid]))
        return indices

    def _joint_qvel_indices(self, model: mujoco.MjModel) -> list[int]:
        indices = []
        for name in self.joint_names:
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid < 0:
                raise ValueError(f"MuJoCo joint not found: {name}")
            indices.append(int(model.jnt_dofadr[jid]))
        return indices

    def _actuator_ids(self, model: mujoco.MjModel) -> list[int]:
        actuator_ids = []
        for name in self.joint_names:
            aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{name}_pos")
            if aid < 0:
                raise ValueError(f"MuJoCo actuator not found: {name}_pos")
            actuator_ids.append(aid)
        return actuator_ids

    def _clip_targets(self, model: mujoco.MjModel, actuator_ids: list[int], target_qpos: np.ndarray) -> np.ndarray:
        clipped = target_qpos.copy()
        for index, aid in enumerate(actuator_ids):
            low, high = model.actuator_ctrlrange[aid]
            clipped[index] = np.clip(clipped[index], low, high)
        return clipped

    def move_to_pose(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        target_qpos: np.ndarray | list[float] | tuple[float, float, float],
        *,
        steps: int | None = None,
        settle_steps: int | None = None,
        max_joint_step: float | None = None,
        fail_threshold: float | None = None,
        direct_qpos: bool | None = None,
        step_callback: Callable[[], None] | None = None,
    ) -> BaseMotionResult:
        qpos_indices = self._joint_qpos_indices(model)
        qvel_indices = self._joint_qvel_indices(model)
        actuator_ids = self._actuator_ids(model)
        target = np.asarray(target_qpos, dtype=np.float64)
        if target.shape != (len(self.joint_names),):
            raise ValueError(f"Expected {len(self.joint_names)} base targets, got shape {target.shape}")

        start = data.qpos[qpos_indices].copy()
        target = self._clip_targets(model, actuator_ids, target)
        current = start.copy()
        total_steps = self.config.default_steps if steps is None else int(steps)
        settle = self.config.default_settle_steps if settle_steps is None else int(settle_steps)
        max_step = self.config.default_max_joint_step if max_joint_step is None else float(max_joint_step)
        threshold = self.config.default_fail_threshold if fail_threshold is None else float(fail_threshold)
        use_direct_qpos = self.config.default_direct_qpos if direct_qpos is None else bool(direct_qpos)

        for _ in range(max(total_steps, 0)):
            current = current + np.clip(target - current, -max_step, max_step)
            for aid, value in zip(actuator_ids, current):
                low, high = model.actuator_ctrlrange[aid]
                data.ctrl[aid] = np.clip(value, low, high)
            if use_direct_qpos:
                data.qpos[qpos_indices] = current
                data.qvel[qvel_indices] = 0.0
                mujoco.mj_forward(model, data)
                update_attachments(model, data)
            else:
                mujoco.mj_step(model, data)
            if step_callback is not None:
                step_callback()

        for aid, value in zip(actuator_ids, target):
            low, high = model.actuator_ctrlrange[aid]
            data.ctrl[aid] = np.clip(value, low, high)
        if use_direct_qpos:
            data.qpos[qpos_indices] = target
            data.qvel[qvel_indices] = 0.0
            mujoco.mj_forward(model, data)
            update_attachments(model, data)
        for _ in range(max(settle, 0)):
            if use_direct_qpos:
                data.qpos[qpos_indices] = target
                data.qvel[qvel_indices] = 0.0
                mujoco.mj_forward(model, data)
                update_attachments(model, data)
            else:
                mujoco.mj_step(model, data)
            if step_callback is not None:
                step_callback()

        final = data.qpos[qpos_indices].copy()
        error = float(np.linalg.norm(final - target))
        return BaseMotionResult(self.joint_names, start, target, final, error, error <= threshold)

    def execute_recovery_action(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        params: dict,
        *,
        step_callback: Callable[[], None] | None = None,
    ) -> BaseMotionResult:
        if "target_qpos" in params:
            target = np.asarray(params["target_qpos"], dtype=np.float64)
        else:
            missing = [key for key in ("base_x", "base_y", "base_yaw") if key not in params]
            if missing:
                raise ValueError(f"Provide target_qpos or required parameter(s): {', '.join(missing)}")
            target = np.array([params["base_x"], params["base_y"], params["base_yaw"]], dtype=np.float64)
        return self.move_to_pose(
            model,
            data,
            target,
            steps=int(params.get("steps", self.config.default_steps)),
            settle_steps=int(params.get("settle_steps", self.config.default_settle_steps)),
            max_joint_step=float(params.get("max_joint_step", self.config.default_max_joint_step)),
            fail_threshold=float(params.get("fail_threshold", self.config.default_fail_threshold)),
            direct_qpos=bool(params.get("direct_qpos", self.config.default_direct_qpos)),
            step_callback=step_callback,
        )


def load_skill(path: str | Path | None = None) -> R1ProBaseMotionSkill:
    config_path = Path(path) if path is not None else Path(__file__).with_suffix(".json")
    return R1ProBaseMotionSkill.from_json(config_path)
