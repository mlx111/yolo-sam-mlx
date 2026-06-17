from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import mujoco
import numpy as np

from skills.base.arm_ik_skill import TORSO_JOINTS, LockedJointState
from skills.primitives.grasp_attachment import update_attachments


@dataclass(frozen=True)
class TorsoMotionResult:
    joint_names: tuple[str, ...]
    start_qpos: np.ndarray
    target_qpos: np.ndarray
    final_qpos: np.ndarray
    final_error: float
    success: bool
    control_mode: str


@dataclass(frozen=True)
class TorsoMoveSkillConfig:
    name: str
    default_steps: int
    default_settle_steps: int
    default_max_joint_step: float
    default_fail_threshold: float
    default_closed_loop_gain: float
    default_direct_qpos: bool


class R1ProTorsoMoveSkill:
    """Move R1Pro torso joints with MuJoCo actuator position control."""

    joint_names = TORSO_JOINTS

    def __init__(self, config: TorsoMoveSkillConfig):
        self.config = config

    @classmethod
    def from_json(cls, path: str | Path) -> "R1ProTorsoMoveSkill":
        config_path = Path(path)
        payload = json.loads(config_path.read_text())
        control_defaults = payload.get("control_defaults", {})
        return cls(
            TorsoMoveSkillConfig(
                name=payload["name"],
                default_steps=int(control_defaults.get("steps", 900)),
                default_settle_steps=int(control_defaults.get("settle_steps", 900)),
                default_max_joint_step=float(control_defaults.get("max_joint_step", 0.004)),
                default_fail_threshold=float(control_defaults.get("fail_threshold", 0.02)),
                default_closed_loop_gain=float(control_defaults.get("closed_loop_gain", 1.0)),
                default_direct_qpos=bool(control_defaults.get("direct_qpos", False)),
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

    def _clip_command(self, model: mujoco.MjModel, actuator_ids: list[int], command: np.ndarray) -> np.ndarray:
        clipped = command.copy()
        for index, aid in enumerate(actuator_ids):
            low, high = model.actuator_ctrlrange[aid]
            clipped[index] = np.clip(clipped[index], low, high)
        return clipped

    def _capture_locked_posture(self, model: mujoco.MjModel, data: mujoco.MjData) -> dict[int, LockedJointState]:
        controlled = set(self.joint_names)
        locked: dict[int, LockedJointState] = {}
        for joint_id in range(model.njnt):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
            if not name or name in controlled:
                continue
            qpos_id = int(model.jnt_qposadr[joint_id])
            dof_id = int(model.jnt_dofadr[joint_id])
            if model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_FREE:
                qpos_size = 7
                qvel_size = 6
            elif model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_BALL:
                qpos_size = 4
                qvel_size = 3
            else:
                qpos_size = 1
                qvel_size = 1
            locked[joint_id] = LockedJointState(
                qpos_adr=qpos_id,
                qpos_size=qpos_size,
                qvel_adr=dof_id,
                qvel_size=qvel_size,
                qpos=data.qpos[qpos_id : qpos_id + qpos_size].copy(),
                qvel=data.qvel[dof_id : dof_id + qvel_size].copy(),
            )
        return locked

    def _apply_locked_posture(self, data: mujoco.MjData, locked: dict[int, LockedJointState]) -> None:
        for state in locked.values():
            data.qpos[state.qpos_adr : state.qpos_adr + state.qpos_size] = state.qpos
            data.qvel[state.qvel_adr : state.qvel_adr + state.qvel_size] = state.qvel

    def move_to_posture(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        target_qpos: np.ndarray | list[float] | tuple[float, float, float, float],
        *,
        steps: int | None = None,
        settle_steps: int | None = None,
        max_joint_step: float | None = None,
        fail_threshold: float | None = None,
        closed_loop_gain: float | None = None,
        direct_qpos: bool | None = None,
        lock_posture: bool = True,
        step_callback: Callable[[], None] | None = None,
    ) -> TorsoMotionResult:
        qpos_indices = self._joint_qpos_indices(model)
        actuator_ids = self._actuator_ids(model)
        target = np.asarray(target_qpos, dtype=np.float64)
        if target.shape != (len(self.joint_names),):
            raise ValueError(f"Expected {len(self.joint_names)} torso targets, got shape {target.shape}")

        start = data.qpos[qpos_indices].copy()
        target = self._clip_targets(model, actuator_ids, target)
        current_command = start.copy()
        total_steps = self.config.default_steps if steps is None else int(steps)
        settle = self.config.default_settle_steps if settle_steps is None else int(settle_steps)
        max_step = self.config.default_max_joint_step if max_joint_step is None else float(max_joint_step)
        threshold = self.config.default_fail_threshold if fail_threshold is None else float(fail_threshold)
        gain = self.config.default_closed_loop_gain if closed_loop_gain is None else float(closed_loop_gain)
        use_direct_qpos = self.config.default_direct_qpos if direct_qpos is None else bool(direct_qpos)
        locked = self._capture_locked_posture(model, data) if lock_posture else {}

        if use_direct_qpos:
            current_qpos = start.copy()
            for _ in range(max(total_steps, 0)):
                delta = np.clip(target - current_qpos, -max_step, max_step)
                current_qpos = current_qpos + delta
                for aid, value in zip(actuator_ids, current_qpos):
                    data.ctrl[aid] = value
                data.qpos[qpos_indices] = current_qpos
                data.qvel[:] = 0.0
                self._apply_locked_posture(data, locked)
                mujoco.mj_forward(model, data)
                update_attachments(model, data)
                if step_callback is not None:
                    step_callback()
            for aid, value in zip(actuator_ids, target):
                data.ctrl[aid] = value
            data.qpos[qpos_indices] = target
            data.qvel[:] = 0.0
            for _ in range(max(settle, 0)):
                for aid, value in zip(actuator_ids, target):
                    data.ctrl[aid] = value
                self._apply_locked_posture(data, locked)
                mujoco.mj_forward(model, data)
                update_attachments(model, data)
                if step_callback is not None:
                    step_callback()
            final = data.qpos[qpos_indices].copy()
            final_error = float(np.linalg.norm(final - target))
            return TorsoMotionResult(
                joint_names=self.joint_names,
                start_qpos=start,
                target_qpos=target,
                final_qpos=final,
                final_error=final_error,
                success=final_error <= threshold,
                control_mode="direct_qpos",
            )

        for _ in range(max(total_steps, 0)):
            measured = data.qpos[qpos_indices].copy()
            desired_command = target + gain * (target - measured)
            delta = np.clip(desired_command - current_command, -max_step, max_step)
            current_command = current_command + delta
            current_command = self._clip_command(model, actuator_ids, current_command)
            for aid, value in zip(actuator_ids, current_command):
                data.ctrl[aid] = value
            mujoco.mj_step(model, data)
            self._apply_locked_posture(data, locked)
            mujoco.mj_forward(model, data)
            update_attachments(model, data)
            if step_callback is not None:
                step_callback()

        for _ in range(max(settle, 0)):
            measured = data.qpos[qpos_indices].copy()
            desired_command = target + gain * (target - measured)
            delta = np.clip(desired_command - current_command, -max_step, max_step)
            current_command = current_command + delta
            current_command = self._clip_command(model, actuator_ids, current_command)
            for aid, value in zip(actuator_ids, current_command):
                data.ctrl[aid] = value
            mujoco.mj_step(model, data)
            self._apply_locked_posture(data, locked)
            mujoco.mj_forward(model, data)
            update_attachments(model, data)
            if step_callback is not None:
                step_callback()

        final = data.qpos[qpos_indices].copy()
        final_error = float(np.linalg.norm(final - target))
        return TorsoMotionResult(
            joint_names=self.joint_names,
            start_qpos=start,
            target_qpos=target,
            final_qpos=final,
            final_error=final_error,
            success=final_error <= threshold,
            control_mode="actuator_joint_servo",
        )

    def execute_recovery_action(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        params: dict,
    ) -> TorsoMotionResult:
        if "target_qpos" in params:
            target_qpos = params["target_qpos"]
        else:
            missing = [name for name in self.joint_names if name not in params]
            if missing:
                raise ValueError(f"Provide target_qpos or required parameter(s): {', '.join(missing)}")
            target_qpos = [
                params["torso_joint1"],
                params["torso_joint2"],
                params["torso_joint3"],
                params["torso_joint4"],
            ]
        return self.move_to_posture(
            model,
            data,
            target_qpos,
            steps=int(params.get("steps", self.config.default_steps)),
            settle_steps=int(params.get("settle_steps", self.config.default_settle_steps)),
            max_joint_step=float(params.get("max_joint_step", self.config.default_max_joint_step)),
            fail_threshold=float(params.get("fail_threshold", self.config.default_fail_threshold)),
            closed_loop_gain=float(params.get("closed_loop_gain", self.config.default_closed_loop_gain)),
            direct_qpos=bool(params.get("direct_qpos", self.config.default_direct_qpos)),
            lock_posture=bool(params.get("lock_posture", True)),
        )


def load_skill(path: str | Path | None = None) -> R1ProTorsoMoveSkill:
    config_path = Path(path) if path is not None else Path(__file__).with_suffix(".json")
    return R1ProTorsoMoveSkill.from_json(config_path)
