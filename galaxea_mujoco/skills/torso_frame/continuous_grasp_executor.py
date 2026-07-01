from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Literal

import mujoco
import numpy as np

from r1pro_control.arm import ArmSide, R1ProMuJoCoSiteServo
from skills.base.arm_ik_skill import ARM_JOINTS, R1ProArmIKSkill
from skills.primitives.grasp_attachment import attach_object_to_hand, update_attachments
from skills.primitives.object_manipulation_skills import _move_tcp, _set_side_gripper


ControlMode = Literal["direct_qpos", "site_servo", "actuator"]
ControlFrame = Literal["grasp_tool"]
TopdownMode = Literal["vertical_down", "forward_parallel", "palm_down", "x_forward", "current"]


@dataclass(frozen=True)
class TorsoFrameContinuousGraspResult:
    success: bool
    target_torso: np.ndarray
    target_world: np.ndarray
    object_start_torso: np.ndarray
    object_start_world: np.ndarray
    object_final_torso: np.ndarray
    object_final_world: np.ndarray
    grasp_pos_torso: np.ndarray
    grasp_pos_world: np.ndarray
    tcp_final_torso: np.ndarray
    tcp_final_world: np.ndarray
    lift_torso: float
    lift_world: float
    stage_errors: dict[str, float] = field(default_factory=dict)
    stage_orientation_errors: dict[str, float] = field(default_factory=dict)
    message: str = ""


class R1ProTorsoFrameContinuousGraspExecutor:
    """Continuous grasp executor whose public targets are in torso_link4 frame."""

    def __init__(self):
        self.servo = R1ProMuJoCoSiteServo()

    def execute(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        *,
        side: ArmSide = "left",
        object_body: str = "target_cube",
        use_object_target: bool = True,
        target_torso: np.ndarray | None = None,
        grasp_offset_torso: np.ndarray | None = None,
        torso_frame: str = "torso_link4",
        pregrasp_offset_torso: np.ndarray,
        auto_grasp_table_clearance: bool = True,
        pad_table_clearance: float = 0.010,
        lift_height: float = 0.10,
        waypoint_count: int = 20,
        waypoint_steps: int = 60,
        solve_iterations: int = 120,
        fail_threshold: float = 0.002,
        orientation_weight: float = 0.35,
        lift_orientation_weight: float = 0.0,
        orientation_threshold: float = 1.0,
        max_joint_step: float = 0.006,
        topdown_mode: TopdownMode = "palm_down",
        close_steps: int = 500,
        pre_close_settle_steps: int = 120,
        close_direct_qpos: bool = False,
        control_mode: ControlMode = "actuator",
        control_frame: ControlFrame = "grasp_tool",
        max_object_displacement_before_close: float = 0.015,
        max_approach_error_before_close: float | None = None,
        attach_on_close: bool = False,
        settle_steps: int = 0,
        step_callback: Callable[[], None] | None = None,
    ) -> TorsoFrameContinuousGraspResult:
        mujoco.mj_forward(model, data)
        frame_pos, frame_xmat = self._reference_frame_pose(model, data, torso_frame)
        object_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, object_body)
        if use_object_target and object_id < 0:
            raise ValueError(f"MuJoCo body not found: {object_body}")
        if not use_object_target and target_torso is None:
            raise ValueError("target_torso is required when use_object_target is False")

        object_start_world = data.xpos[object_id].copy() if object_id >= 0 else self._local_to_world_pos(frame_pos, frame_xmat, target_torso)
        object_start_torso = self._world_to_local_pos(frame_pos, frame_xmat, object_start_world)
        base_target_torso = object_start_torso if use_object_target else np.asarray(target_torso, dtype=np.float64).reshape(3)

        offset_torso = (
            np.zeros(3, dtype=np.float64)
            if grasp_offset_torso is None
            else np.asarray(grasp_offset_torso, dtype=np.float64).reshape(3)
        )
        target_torso_arr = base_target_torso.copy()
        grasp_center_torso = base_target_torso + offset_torso

        self._hold_gripper_open(model, data, side)
        tcp_start_torso = self._tcp_pos_torso(model, data, side, control_frame, torso_frame)
        target_xmat_world = self._topdown_xmat_world(model, data, side, topdown_mode, control_frame)

        grasp_world_for_clearance = self._local_to_world_pos(frame_pos, frame_xmat, grasp_center_torso)
        if auto_grasp_table_clearance and use_object_target:
            grasp_world_for_clearance = self._table_clearance_adjusted_grasp_pos(
                model,
                data,
                side,
                object_body,
                grasp_world_for_clearance,
                target_xmat_world,
                control_frame=control_frame,
                pad_table_clearance=pad_table_clearance,
            )
        grasp_pos_torso = self._world_to_local_pos(frame_pos, frame_xmat, grasp_world_for_clearance)
        pregrasp_pos_torso = grasp_pos_torso + np.asarray(pregrasp_offset_torso, dtype=np.float64).reshape(3)

        stage_errors: dict[str, float] = {}
        stage_orientation_errors: dict[str, float] = {}
        ok = True
        pregrasp_orientation_weight = (
            float(lift_orientation_weight)
            if float(lift_orientation_weight) > 0.0
            else float(orientation_weight)
        )
        ok &= self._follow_line_torso(
            model,
            data,
            side,
            "move_to_pregrasp",
            tcp_start_torso,
            pregrasp_pos_torso,
            target_xmat_world,
            torso_frame,
            stage_errors,
            stage_orientation_errors,
            waypoint_count=waypoint_count,
            waypoint_steps=waypoint_steps,
            solve_iterations=solve_iterations,
            fail_threshold=fail_threshold,
            orientation_weight=pregrasp_orientation_weight,
            orientation_threshold=orientation_threshold,
            max_joint_step=max_joint_step,
            control_mode=control_mode,
            settle_steps=settle_steps,
            step_callback=step_callback,
            control_frame=control_frame,
        )
        ok &= self._follow_line_torso(
            model,
            data,
            side,
            "approach_object",
            self._tcp_pos_torso(model, data, side, control_frame, torso_frame),
            grasp_pos_torso,
            target_xmat_world,
            torso_frame,
            stage_errors,
            stage_orientation_errors,
            waypoint_count=waypoint_count,
            waypoint_steps=waypoint_steps,
            solve_iterations=solve_iterations,
            fail_threshold=fail_threshold,
            orientation_weight=orientation_weight,
            orientation_threshold=orientation_threshold,
            max_joint_step=max_joint_step,
            control_mode=control_mode,
            settle_steps=settle_steps,
            step_callback=step_callback,
            control_frame=control_frame,
        )

        mujoco.mj_forward(model, data)
        object_after_approach_world = data.xpos[object_id].copy() if object_id >= 0 else object_start_world.copy()
        object_displacement = float(np.linalg.norm(object_after_approach_world - object_start_world))
        if max_approach_error_before_close is None:
            max_approach_error_before_close = max(float(fail_threshold) * 6.0, 0.012)
        approach_error = float(stage_errors.get("approach_object", float("inf")))
        if approach_error > max_approach_error_before_close or object_displacement > max_object_displacement_before_close:
            return self._result(
                model,
                data,
                side,
                object_id,
                object_start_world,
                target_torso_arr,
                grasp_pos_torso,
                torso_frame,
                control_frame,
                stage_errors,
                stage_orientation_errors,
                success=False,
                message=(
                    "stopped_before_close: approach_error="
                    f"{approach_error:.6f} > max_approach_error_before_close={float(max_approach_error_before_close):.6f}"
                    if approach_error > max_approach_error_before_close
                    else "stopped_before_close: object_displacement_before_close="
                    f"{object_displacement:.6f} > max={max_object_displacement_before_close:.6f}"
                ),
            )

        if pre_close_settle_steps > 0:
            arm_joint_names = ARM_JOINTS[side]
            arm_hold = np.asarray(
                [
                    float(data.qpos[model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)]])
                    for name in arm_joint_names
                ],
                dtype=np.float64,
            )
            for _ in range(pre_close_settle_steps):
                for name, target in zip(arm_joint_names, arm_hold):
                    actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{name}_pos")
                    low, high = model.actuator_ctrlrange[actuator_id]
                    data.ctrl[actuator_id] = np.clip(target, low, high)
                mujoco.mj_step(model, data)
                if step_callback is not None:
                    step_callback()

        _set_side_gripper(
            model,
            data,
            side,
            "close",
            steps=close_steps,
            direct_qpos=close_direct_qpos,
            hold_joint_names=ARM_JOINTS[side],
            step_callback=step_callback,
        )
        mujoco.mj_forward(model, data)
        object_after_close_world = data.xpos[object_id].copy() if object_id >= 0 else object_start_world.copy()
        close_displacement = float(np.linalg.norm(object_after_close_world - object_start_world))
        if close_displacement > max_object_displacement_before_close:
            return self._result(
                model,
                data,
                side,
                object_id,
                object_start_world,
                target_torso_arr,
                grasp_pos_torso,
                torso_frame,
                control_frame,
                stage_errors,
                stage_orientation_errors,
                success=False,
                message=(
                    "stopped_after_close: object_displacement="
                    f"{close_displacement:.6f} > max={max_object_displacement_before_close:.6f}"
                ),
            )

        if attach_on_close and control_mode == "direct_qpos" and object_id >= 0:
            attach_object_to_hand(model, data, side, object_body)

        lift_start_torso = self._tcp_pos_torso(model, data, side, control_frame, torso_frame)
        lift_xmat_world = self._tcp_xmat_world(model, data, side, control_frame)
        ok &= self._vertical_cartesian_lift_torso(
            model,
            data,
            side,
            f"{side}_vertical_lift",
            lift_start_torso,
            lift_xmat_world,
            torso_frame,
            stage_errors,
            stage_orientation_errors,
            lift_height=lift_height,
            segment_height=0.01,
            segment_steps=max(waypoint_steps, 80),
            fail_threshold=max(fail_threshold, 0.004),
            orientation_weight=orientation_weight,
            orientation_threshold=orientation_threshold,
            max_joint_step=min(max_joint_step, 0.003),
            step_callback=step_callback,
            control_frame=control_frame,
            hold_gripper_side=side,
        )
        update_attachments(model, data, side)

        result = self._result(
            model,
            data,
            side,
            object_id,
            object_start_world,
            target_torso_arr,
            grasp_pos_torso,
            torso_frame,
            control_frame,
            stage_errors,
            stage_orientation_errors,
            success=False,
            message="",
        )
        del ok
        success = bool(result.lift_torso >= max(0.0, float(lift_height) - 0.005))
        return TorsoFrameContinuousGraspResult(
            success=success,
            target_torso=result.target_torso,
            target_world=result.target_world,
            object_start_torso=result.object_start_torso,
            object_start_world=result.object_start_world,
            object_final_torso=result.object_final_torso,
            object_final_world=result.object_final_world,
            grasp_pos_torso=result.grasp_pos_torso,
            grasp_pos_world=result.grasp_pos_world,
            tcp_final_torso=result.tcp_final_torso,
            tcp_final_world=result.tcp_final_world,
            lift_torso=result.lift_torso,
            lift_world=result.lift_world,
            stage_errors=stage_errors,
            stage_orientation_errors=stage_orientation_errors,
            message=f"lift_torso={result.lift_torso:.6f}, lift_world={result.lift_world:.6f}",
        )

    def _follow_line_torso(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        side: ArmSide,
        stage: str,
        start_torso: np.ndarray,
        target_torso: np.ndarray,
        target_xmat_world: np.ndarray,
        torso_frame: str,
        stage_errors: dict[str, float],
        stage_orientation_errors: dict[str, float],
        *,
        waypoint_count: int,
        waypoint_steps: int,
        solve_iterations: int,
        fail_threshold: float,
        orientation_weight: float,
        orientation_threshold: float,
        max_joint_step: float,
        control_mode: ControlMode,
        settle_steps: int,
        step_callback: Callable[[], None] | None,
        control_frame: ControlFrame,
        hold_gripper_side: ArmSide | None = None,
    ) -> bool:
        ok = True
        final_error = float("inf")
        ik_skill = R1ProArmIKSkill()

        def effective_step_callback() -> None:
            if hold_gripper_side is not None:
                self._hold_gripper_closed(model, data, hold_gripper_side)
            if step_callback is not None:
                step_callback()

        for index in range(1, max(waypoint_count, 1) + 1):
            frame_pos, frame_xmat = self._reference_frame_pose(model, data, torso_frame)
            s = index / max(waypoint_count, 1)
            s = s * s * (3.0 - 2.0 * s)
            waypoint_torso = (1.0 - s) * start_torso + s * target_torso
            waypoint_world = self._local_to_world_pos(frame_pos, frame_xmat, waypoint_torso)
            if control_mode == "site_servo":
                result = self.servo.move_to_position(
                    model,
                    data,
                    side,
                    waypoint_world,
                    frame=control_frame,
                    target_xmat=target_xmat_world,
                    steps=waypoint_steps,
                    settle_steps=settle_steps,
                    solve_iterations=solve_iterations,
                    fail_threshold=fail_threshold,
                    orientation_threshold=orientation_threshold,
                    orientation_weight=orientation_weight,
                    max_joint_step=max_joint_step,
                    posture_gain=0.12,
                    runtime_damping=35.0,
                    runtime_armature=0.05,
                    step_callback=effective_step_callback,
                )
            elif control_mode == "direct_qpos":
                result = _move_tcp(
                    model,
                    data,
                    side,
                    waypoint_world,
                    {
                        "control_mode": "direct_qpos",
                        f"_locked_tcp_xmat_{side}": target_xmat_world,
                        "steps": waypoint_steps,
                        "settle_steps": settle_steps,
                        "solve_iterations": solve_iterations,
                        "fail_threshold": fail_threshold,
                        "orientation_threshold": orientation_threshold,
                        "orientation_weight": orientation_weight,
                        "max_joint_step": max_joint_step,
                        "posture_gain": 0.12,
                    },
                    step_callback=effective_step_callback,
                )
                update_attachments(model, data, side)
            elif control_mode == "actuator":
                result = ik_skill.move_to_position(
                    model,
                    data,
                    side,
                    waypoint_world,
                    target_xmat=None if float(orientation_weight) <= 0.0 else target_xmat_world,
                    steps=waypoint_steps,
                    settle_steps=settle_steps,
                    direct_qpos=False,
                    stabilize=True,
                    closed_loop=True,
                    lock_posture=True,
                    max_joint_step=max_joint_step,
                    fail_threshold=fail_threshold,
                    orientation_threshold=orientation_threshold,
                    orientation_weight=orientation_weight,
                    step_callback=effective_step_callback,
                )
            else:
                raise ValueError(f"Unsupported control_mode: {control_mode!r}")
            if hold_gripper_side is not None:
                self._hold_gripper_closed(model, data, hold_gripper_side)
            final_error = float(result.final_error)
            ok = ok and final_error <= fail_threshold
        stage_errors[stage] = final_error
        stage_orientation_errors[stage] = self._orientation_error_world(model, data, side, target_xmat_world, control_frame)
        return ok

    def _vertical_cartesian_lift_torso(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        side: ArmSide,
        stage: str,
        start_torso: np.ndarray,
        target_xmat_world: np.ndarray,
        torso_frame: str,
        stage_errors: dict[str, float],
        stage_orientation_errors: dict[str, float],
        *,
        lift_height: float,
        segment_height: float,
        segment_steps: int,
        fail_threshold: float,
        orientation_weight: float,
        orientation_threshold: float,
        max_joint_step: float,
        step_callback: Callable[[], None] | None,
        control_frame: ControlFrame,
        hold_gripper_side: ArmSide,
    ) -> bool:
        start_torso = np.asarray(start_torso, dtype=np.float64).reshape(3)
        target_xmat_world = np.asarray(target_xmat_world, dtype=np.float64).reshape(3, 3)
        lift_delta = float(lift_height)
        segment_size = max(float(segment_height), 1e-4)
        segment_count = max(1, int(np.ceil(abs(lift_delta) / segment_size)))
        ok = True
        final_error = float("inf")
        def lift_step_callback() -> None:
            self._hold_gripper_closed(model, data, hold_gripper_side)
            if step_callback is not None:
                step_callback()

        for index in range(1, segment_count + 1):
            frame_pos, frame_xmat = self._reference_frame_pose(model, data, torso_frame)
            alpha = index / segment_count
            dz = lift_delta * alpha
            target_torso = start_torso + np.array([0.0, 0.0, dz], dtype=np.float64)
            target_world = self._local_to_world_pos(frame_pos, frame_xmat, target_torso)
            result = self.servo.move_to_position(
                model,
                data,
                side,
                target_world,
                frame=control_frame,
                target_xmat=target_xmat_world,
                steps=segment_steps,
                settle_steps=0,
                solve_iterations=180,
                fail_threshold=fail_threshold,
                orientation_threshold=orientation_threshold,
                orientation_weight=max(float(orientation_weight), 0.6),
                max_joint_step=max_joint_step,
                posture_gain=0.25,
                runtime_damping=65.0,
                runtime_armature=0.08,
                step_callback=lift_step_callback,
            )
            self._hold_gripper_closed(model, data, hold_gripper_side)
            final_error = float(result.final_error)
            ok = ok and final_error <= fail_threshold
        stage_errors[stage] = final_error
        stage_orientation_errors[stage] = self._orientation_error_world(model, data, side, target_xmat_world, control_frame)
        return ok

    def _result(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        side: ArmSide,
        object_id: int,
        object_start_world: np.ndarray,
        target_torso: np.ndarray,
        grasp_pos_torso: np.ndarray,
        torso_frame: str,
        control_frame: ControlFrame,
        stage_errors: dict[str, float],
        stage_orientation_errors: dict[str, float],
        *,
        success: bool,
        message: str,
    ) -> TorsoFrameContinuousGraspResult:
        mujoco.mj_forward(model, data)
        frame_pos, frame_xmat = self._reference_frame_pose(model, data, torso_frame)
        object_final_world = data.xpos[object_id].copy() if object_id >= 0 else object_start_world.copy()
        object_start_torso = self._world_to_local_pos(frame_pos, frame_xmat, object_start_world)
        object_final_torso = self._world_to_local_pos(frame_pos, frame_xmat, object_final_world)
        tcp_final_world = self._tcp_pos_world(model, data, side, control_frame)
        tcp_final_torso = self._world_to_local_pos(frame_pos, frame_xmat, tcp_final_world)
        target_world = self._local_to_world_pos(frame_pos, frame_xmat, target_torso)
        grasp_pos_world = self._local_to_world_pos(frame_pos, frame_xmat, grasp_pos_torso)
        return TorsoFrameContinuousGraspResult(
            success=success,
            target_torso=target_torso.copy(),
            target_world=target_world,
            object_start_torso=object_start_torso,
            object_start_world=object_start_world.copy(),
            object_final_torso=object_final_torso,
            object_final_world=object_final_world,
            grasp_pos_torso=grasp_pos_torso.copy(),
            grasp_pos_world=grasp_pos_world,
            tcp_final_torso=tcp_final_torso,
            tcp_final_world=tcp_final_world,
            lift_torso=float(object_final_torso[2] - object_start_torso[2]),
            lift_world=float(object_final_world[2] - object_start_world[2]),
            stage_errors=stage_errors,
            stage_orientation_errors=stage_orientation_errors,
            message=message,
        )

    def _reference_frame_pose(self, model: mujoco.MjModel, data: mujoco.MjData, frame_name: str) -> tuple[np.ndarray, np.ndarray]:
        mujoco.mj_forward(model, data)
        name = str(frame_name or "world").strip()
        if not name or name == "world":
            return np.zeros(3, dtype=np.float64), np.eye(3, dtype=np.float64)
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if body_id >= 0:
            return data.xpos[body_id].copy(), data.xmat[body_id].reshape(3, 3).copy()
        site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
        if site_id >= 0:
            return data.site_xpos[site_id].copy(), data.site_xmat[site_id].reshape(3, 3).copy()
        raise ValueError(f"MuJoCo reference frame not found as body or site: {frame_name}")

    def _world_to_local_pos(self, frame_pos: np.ndarray, frame_xmat: np.ndarray, world_pos: np.ndarray) -> np.ndarray:
        return np.asarray(frame_xmat, dtype=np.float64).reshape(3, 3).T @ (
            np.asarray(world_pos, dtype=np.float64).reshape(3) - np.asarray(frame_pos, dtype=np.float64).reshape(3)
        )

    def _local_to_world_pos(self, frame_pos: np.ndarray, frame_xmat: np.ndarray, local_pos: np.ndarray) -> np.ndarray:
        return np.asarray(frame_pos, dtype=np.float64).reshape(3) + np.asarray(frame_xmat, dtype=np.float64).reshape(3, 3) @ np.asarray(local_pos, dtype=np.float64).reshape(3)

    def _local_to_world_xmat(self, frame_xmat: np.ndarray, local_xmat: np.ndarray) -> np.ndarray:
        return np.asarray(frame_xmat, dtype=np.float64).reshape(3, 3) @ np.asarray(local_xmat, dtype=np.float64).reshape(3, 3)

    def _site_name(self, side: ArmSide, frame: ControlFrame) -> str:
        if frame != "grasp_tool":
            raise ValueError(f"Unsupported grasp control frame: {frame!r}; use 'grasp_tool'")
        return f"{side}_grasp_tool"

    def _tcp_pos_world(self, model: mujoco.MjModel, data: mujoco.MjData, side: ArmSide, frame: ControlFrame = "grasp_tool") -> np.ndarray:
        mujoco.mj_forward(model, data)
        site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, self._site_name(side, frame))
        if site_id < 0:
            raise ValueError(f"MuJoCo site not found: {self._site_name(side, frame)}")
        return data.site_xpos[site_id].copy()

    def _tcp_xmat_world(self, model: mujoco.MjModel, data: mujoco.MjData, side: ArmSide, frame: ControlFrame = "grasp_tool") -> np.ndarray:
        mujoco.mj_forward(model, data)
        site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, self._site_name(side, frame))
        if site_id < 0:
            raise ValueError(f"MuJoCo site not found: {self._site_name(side, frame)}")
        return data.site_xmat[site_id].reshape(3, 3).copy()

    def _tcp_pos_torso(self, model: mujoco.MjModel, data: mujoco.MjData, side: ArmSide, frame: ControlFrame, torso_frame: str) -> np.ndarray:
        frame_pos, frame_xmat = self._reference_frame_pose(model, data, torso_frame)
        return self._world_to_local_pos(frame_pos, frame_xmat, self._tcp_pos_world(model, data, side, frame))

    def _orientation_error_world(self, model: mujoco.MjModel, data: mujoco.MjData, side: ArmSide, target_xmat: np.ndarray, frame: ControlFrame) -> float:
        current = self._tcp_xmat_world(model, data, side, frame)
        delta = np.asarray(target_xmat, dtype=np.float64).reshape(3, 3).T @ current
        cos_angle = max(-1.0, min(1.0, (float(np.trace(delta)) - 1.0) * 0.5))
        return float(np.arccos(cos_angle))

    def _topdown_xmat_world(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        side: ArmSide,
        mode: TopdownMode,
        frame: ControlFrame,
    ) -> np.ndarray:
        if mode == "current":
            return self._tcp_xmat_world(model, data, side, frame)
        if mode in ("vertical_down", "palm_down"):
            y_axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)
            z_axis = np.array([0.0, -1.0, 0.0], dtype=np.float64)
            x_axis = np.cross(y_axis, z_axis)
            x_axis = x_axis / np.linalg.norm(x_axis)
            return np.column_stack((x_axis, y_axis, z_axis))
        if mode == "forward_parallel":
            y_axis = np.array([-1.0, 0.0, 0.0], dtype=np.float64)
            z_axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)
            x_axis = np.cross(y_axis, z_axis)
            x_axis = x_axis / np.linalg.norm(x_axis)
            return np.column_stack((x_axis, y_axis, z_axis))
        x_axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        y_axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        z_axis = np.cross(x_axis, y_axis)
        z_axis = z_axis / np.linalg.norm(z_axis)
        x_axis = np.cross(y_axis, z_axis)
        x_axis = x_axis / np.linalg.norm(x_axis)
        return np.column_stack((x_axis, y_axis, z_axis))

    def _table_clearance_adjusted_grasp_pos(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        side: ArmSide,
        object_body: str,
        grasp_pos: np.ndarray,
        target_xmat: np.ndarray,
        *,
        control_frame: ControlFrame,
        pad_table_clearance: float,
    ) -> np.ndarray:
        half_height = self._object_half_height(model, object_body)
        if half_height <= 0.0:
            return grasp_pos
        min_pad_lowest_z = self._min_pad_lowest_z_offset_at_target_pose(
            model,
            data,
            side,
            target_xmat,
            control_frame=control_frame,
        )
        table_z = float(grasp_pos[2]) - half_height
        min_tcp_z = table_z + float(pad_table_clearance) - min_pad_lowest_z
        adjusted = np.asarray(grasp_pos, dtype=np.float64).copy()
        adjusted[2] = max(float(adjusted[2]), float(min_tcp_z))
        return adjusted

    def _object_half_height(self, model: mujoco.MjModel, object_body: str) -> float:
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, object_body)
        if body_id < 0:
            return 0.0
        for geom_id in range(model.ngeom):
            if int(model.geom_bodyid[geom_id]) != int(body_id):
                continue
            if model.geom_type[geom_id] == mujoco.mjtGeom.mjGEOM_SPHERE:
                return float(model.geom_size[geom_id][0])
            if model.geom_type[geom_id] == mujoco.mjtGeom.mjGEOM_BOX:
                return float(model.geom_size[geom_id][2])
        return 0.0

    def _min_pad_lowest_z_offset_at_target_pose(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        side: ArmSide,
        target_xmat: np.ndarray,
        *,
        control_frame: ControlFrame,
    ) -> float:
        mujoco.mj_forward(model, data)
        tcp_pos = self._tcp_pos_world(model, data, side, control_frame)
        tcp_xmat = self._tcp_xmat_world(model, data, side, control_frame)
        target_xmat_arr = np.asarray(target_xmat, dtype=np.float64).reshape(3, 3)
        z_offsets: list[float] = []
        uses_merged_long_pad = False
        for name in self._pad_geom_names(side):
            geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
            if geom_id < 0:
                continue
            if name.endswith("_pad"):
                uses_merged_long_pad = True
            rel_center = tcp_xmat.T @ (data.geom_xpos[geom_id] - tcp_pos)
            geom_axes_tcp = tcp_xmat.T @ data.geom_xmat[geom_id].reshape(3, 3)
            half_sizes = np.asarray(model.geom_size[geom_id][:3], dtype=np.float64)
            clearance_half_sizes = half_sizes.copy()
            clearance_half_sizes[1] = min(float(clearance_half_sizes[1]), 0.016)
            for sx in (-1.0, 1.0):
                for sy in (-1.0, 1.0):
                    for sz in (-1.0, 1.0):
                        corner_tcp = rel_center + geom_axes_tcp @ (clearance_half_sizes * np.array([sx, sy, sz], dtype=np.float64))
                        corner_target = target_xmat_arr @ corner_tcp
                        z_offsets.append(float(corner_target[2]))
        if not z_offsets:
            return 0.0
        lowest = min(z_offsets)
        if uses_merged_long_pad:
            lowest = min(lowest, -0.035)
        return lowest

    def _pad_geom_names(self, side: ArmSide) -> tuple[str, ...]:
        return (
            f"{side}_gripper_finger_link1_pad",
            f"{side}_gripper_finger_link2_pad",
        )

    def _hold_gripper_closed(self, model: mujoco.MjModel, data: mujoco.MjData, side: ArmSide) -> None:
        split_actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{side}_gripper_fingers_pos")
        if split_actuator_id >= 0:
            low, high = model.actuator_ctrlrange[split_actuator_id]
            data.ctrl[split_actuator_id] = np.clip(high, low, high)
            return
        for index in (1, 2):
            actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{side}_gripper_finger_joint{index}_pos")
            if actuator_id >= 0:
                low, high = model.actuator_ctrlrange[actuator_id]
                data.ctrl[actuator_id] = np.clip(high, low, high)

    def _hold_gripper_open(self, model: mujoco.MjModel, data: mujoco.MjData, side: ArmSide) -> None:
        split_actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{side}_gripper_fingers_pos")
        if split_actuator_id >= 0:
            low, high = model.actuator_ctrlrange[split_actuator_id]
            data.ctrl[split_actuator_id] = np.clip(low, low, high)
            return
        for index in (1, 2):
            actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{side}_gripper_finger_joint{index}_pos")
            if actuator_id >= 0:
                low, high = model.actuator_ctrlrange[actuator_id]
                data.ctrl[actuator_id] = np.clip(low, low, high)
