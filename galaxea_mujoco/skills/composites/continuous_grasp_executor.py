from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Literal

import mujoco
import numpy as np

from r1pro_control.arm import ArmSide, R1ProMuJoCoSiteServo
from skills.base.arm_ik_skill import R1ProArmIKSkill
from skills.primitives.grasp_attachment import attach_object_to_hand, update_attachments
from skills.base.arm_ik_skill import ARM_JOINTS
from skills.primitives.object_manipulation_skills import _move_tcp, _set_side_gripper


@dataclass(frozen=True)
class ContinuousGraspResult:
    success: bool
    object_start: np.ndarray
    object_final: np.ndarray
    tcp_final: np.ndarray
    lift: float
    stage_errors: dict[str, float] = field(default_factory=dict)
    stage_orientation_errors: dict[str, float] = field(default_factory=dict)
    message: str = ""


class R1ProContinuousGraspExecutor:
    """UR5e-style continuous grasp executor built on the MuJoCo site servo."""

    def __init__(self):
        self.servo = R1ProMuJoCoSiteServo()

    def execute(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        *,
        side: ArmSide = "left",
        object_body: str = "target_cube",
        grasp_offset: np.ndarray | None = None,
        pregrasp_distance: float = 0.12,
        auto_grasp_table_clearance: bool = True,
        pad_table_clearance: float = 0.010,
        lift_height: float = 0.30,
        waypoint_count: int = 20,
        waypoint_steps: int = 60,
        solve_iterations: int = 120,
        fail_threshold: float = 0.002,
        orientation_weight: float = 0.02,
        lift_orientation_weight: float = 0.0,
        orientation_threshold: float = 1.0,
        max_joint_step: float = 0.006,
        topdown_mode: Literal["palm_down", "x_forward", "x_side", "current"] = "palm_down",
        close_steps: int = 500,
        pre_close_settle_steps: int = 120,
        close_direct_qpos: bool = False,
        control_mode: Literal["direct_qpos", "site_servo", "actuator"] = "actuator",
        control_frame: Literal["grasp_tool"] = "grasp_tool",
        max_object_displacement_before_close: float = 0.015,
        max_approach_error_before_close: float | None = None,
        attach_on_close: bool = False,
        settle_steps: int = 0,
        step_callback: Callable[[], None] | None = None,
    ) -> ContinuousGraspResult:
        mujoco.mj_forward(model, data)
        object_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, object_body)
        if object_id < 0:
            raise ValueError(f"MuJoCo body not found: {object_body}")

        object_start = data.xpos[object_id].copy()
        self._hold_gripper_open(model, data, side)
        tcp_start = self._tcp_pos(model, data, side, control_frame)
        grasp_offset_arr = np.zeros(3, dtype=np.float64) if grasp_offset is None else np.asarray(grasp_offset, dtype=np.float64)
        grasp_center_pos = object_start + grasp_offset_arr
        target_xmat = self._topdown_xmat(model, data, side, topdown_mode, control_frame)
        grasp_pos = self._table_clearance_adjusted_grasp_pos(
            model,
            data,
            side,
            object_body,
            grasp_center_pos,
            target_xmat,
            control_frame=control_frame,
            enabled=auto_grasp_table_clearance,
            pad_table_clearance=pad_table_clearance,
        )
        pregrasp_pos = grasp_pos + np.array([0.0, 0.0, pregrasp_distance], dtype=np.float64)
        lift_pos = grasp_pos + np.array([0.0, 0.0, lift_height], dtype=np.float64)

        stage_errors: dict[str, float] = {}
        stage_orientation_errors: dict[str, float] = {}
        ok = True
        pregrasp_orientation_weight = (
            float(lift_orientation_weight)
            if float(lift_orientation_weight) > 0.0
            else float(orientation_weight)
        )
        pregrasp_ok = self._follow_line(
            model,
            data,
            side,
            "move_to_pregrasp",
            tcp_start,
            pregrasp_pos,
            target_xmat,
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
        ok &= pregrasp_ok
        approach_ok = self._follow_line(
            model,
            data,
            side,
            "approach_object",
            self._tcp_pos(model, data, side, control_frame),
            grasp_pos,
            target_xmat,
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
        ok &= approach_ok
        mujoco.mj_forward(model, data)
        object_after_approach = data.xpos[object_id].copy()
        object_displacement = float(np.linalg.norm(object_after_approach - object_start))
        if max_approach_error_before_close is None:
            max_approach_error_before_close = max(float(fail_threshold) * 6.0, 0.012)
        approach_error = float(stage_errors.get("approach_object", float("inf")))
        if approach_error > max_approach_error_before_close or object_displacement > max_object_displacement_before_close:
            mujoco.mj_forward(model, data)
            object_final = data.xpos[object_id].copy()
            tcp_final = self._tcp_pos(model, data, side, control_frame)
            lift = float(object_final[2] - object_start[2])
            reason = (
                "approach_error="
                f"{approach_error:.6f} "
                f"> max_approach_error_before_close={float(max_approach_error_before_close):.6f}"
                f", grasp_target={np.round(grasp_pos, 6).tolist()}"
                f", tcp_final={np.round(tcp_final, 6).tolist()}"
                if approach_error > max_approach_error_before_close
                else "object_displacement_before_close="
                f"{object_displacement:.6f} > max={max_object_displacement_before_close:.6f}"
            )
            return ContinuousGraspResult(
                success=False,
                object_start=object_start,
                object_final=object_final,
                tcp_final=tcp_final,
                lift=lift,
                stage_errors=stage_errors,
                stage_orientation_errors=stage_orientation_errors,
                message=f"stopped_before_close: {reason}",
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
        object_after_close = data.xpos[object_id].copy()
        close_displacement = float(np.linalg.norm(object_after_close - object_start))
        if close_displacement > max_object_displacement_before_close:
            tcp_final = self._tcp_pos(model, data, side, control_frame)
            lift = float(object_after_close[2] - object_start[2])
            return ContinuousGraspResult(
                success=False,
                object_start=object_start,
                object_final=object_after_close,
                tcp_final=tcp_final,
                lift=lift,
                stage_errors=stage_errors,
                stage_orientation_errors=stage_orientation_errors,
                message=(
                    "stopped_after_close: object_displacement="
                    f"{close_displacement:.6f} > max={max_object_displacement_before_close:.6f}"
                ),
            )

        if attach_on_close and control_mode == "direct_qpos":
            attach_object_to_hand(model, data, side, object_body)

        lift_start = self._tcp_pos(model, data, side, control_frame)
        lift_xmat = self._tcp_xmat(model, data, side, control_frame)
        ok &= self._vertical_cartesian_lift(
            model,
            data,
            side,
            f"{side}_vertical_lift",
            lift_start,
            lift_xmat,
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

        mujoco.mj_forward(model, data)
        object_final = data.xpos[object_id].copy()
        tcp_final = self._tcp_pos(model, data, side, control_frame)
        lift = float(object_final[2] - object_start[2])
        success = bool(lift > 0.03)
        return ContinuousGraspResult(
            success=success,
            object_start=object_start,
            object_final=object_final,
            tcp_final=tcp_final,
            lift=lift,
            stage_errors=stage_errors,
            stage_orientation_errors=stage_orientation_errors,
            message=f"lift={lift:.6f}",
        )

    def _follow_line(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        side: ArmSide,
        stage: str,
        start: np.ndarray,
        target: np.ndarray,
        target_xmat: np.ndarray,
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
        control_mode: Literal["direct_qpos", "site_servo", "actuator"],
        settle_steps: int,
        step_callback: Callable[[], None] | None,
        control_frame: Literal["grasp_tool"],
        hold_gripper_side: ArmSide | None = None,
    ) -> bool:
        ok = True
        final_error = float("inf")
        ik_skill = R1ProArmIKSkill()
        effective_step_callback = step_callback
        if hold_gripper_side is not None:
            def effective_step_callback() -> None:
                self._hold_gripper_closed(model, data, hold_gripper_side)
                if step_callback is not None:
                    step_callback()

        for i in range(1, max(waypoint_count, 1) + 1):
            s = i / max(waypoint_count, 1)
            s = s * s * (3.0 - 2.0 * s)
            waypoint = (1.0 - s) * start + s * target
            if control_mode == "site_servo":
                result = self.servo.move_to_position(
                    model,
                    data,
                    side,
                    waypoint,
                    frame=control_frame,
                    target_xmat=target_xmat,
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
                    waypoint,
                    {
                        "control_mode": "direct_qpos",
                        f"_locked_tcp_xmat_{side}": target_xmat,
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
                    waypoint,
                    target_xmat=target_xmat,
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
            final_error = result.final_error
            ok = result.final_error <= fail_threshold
        stage_errors[stage] = final_error
        stage_orientation_errors[stage] = self._orientation_error(model, data, side, target_xmat, control_frame)
        return ok

    def _vertical_cartesian_lift(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        side: ArmSide,
        stage: str,
        start: np.ndarray,
        target_xmat: np.ndarray,
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
        control_frame: Literal["grasp_tool"],
        hold_gripper_side: ArmSide,
    ) -> bool:
        start = np.asarray(start, dtype=np.float64).reshape(3)
        target_xmat = np.asarray(target_xmat, dtype=np.float64).reshape(3, 3)
        segment_count = max(1, int(np.ceil(float(lift_height) / max(float(segment_height), 1e-4))))
        ok = True
        final_error = float("inf")

        def lift_step_callback() -> None:
            self._hold_gripper_closed(model, data, hold_gripper_side)
            if step_callback is not None:
                step_callback()

        for index in range(1, segment_count + 1):
            dz = min(float(lift_height), index * float(segment_height))
            target = start + np.array([0.0, 0.0, dz], dtype=np.float64)
            result = self.servo.move_to_position(
                model,
                data,
                side,
                target,
                frame=control_frame,
                target_xmat=target_xmat,
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
            final_error = result.final_error
            ok = ok and result.final_error <= fail_threshold
        stage_errors[stage] = final_error
        stage_orientation_errors[stage] = self._orientation_error(model, data, side, target_xmat, control_frame)
        return ok

    def _hold_gripper_closed(self, model: mujoco.MjModel, data: mujoco.MjData, side: ArmSide) -> None:
        split_actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{side}_gripper_fingers_pos")
        if split_actuator_id >= 0:
            low, high = model.actuator_ctrlrange[split_actuator_id]
            data.ctrl[split_actuator_id] = np.clip(high, low, high)
            return
        for index in (1, 2):
            actuator_id = mujoco.mj_name2id(
                model,
                mujoco.mjtObj.mjOBJ_ACTUATOR,
                f"{side}_gripper_finger_joint{index}_pos",
            )
            if actuator_id < 0:
                continue
            low, high = model.actuator_ctrlrange[actuator_id]
            data.ctrl[actuator_id] = np.clip(high, low, high)

    def _hold_gripper_open(self, model: mujoco.MjModel, data: mujoco.MjData, side: ArmSide) -> None:
        split_actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{side}_gripper_fingers_pos")
        if split_actuator_id >= 0:
            low, high = model.actuator_ctrlrange[split_actuator_id]
            data.ctrl[split_actuator_id] = np.clip(low, low, high)
            return
        for index in (1, 2):
            actuator_id = mujoco.mj_name2id(
                model,
                mujoco.mjtObj.mjOBJ_ACTUATOR,
                f"{side}_gripper_finger_joint{index}_pos",
            )
            if actuator_id < 0:
                continue
            low, high = model.actuator_ctrlrange[actuator_id]
            data.ctrl[actuator_id] = np.clip(low, low, high)


    def _site_name(self, side: ArmSide, frame: Literal["grasp_tool"]) -> str:
        if frame != "grasp_tool":
            raise ValueError(f"Unsupported grasp control frame: {frame!r}; use 'grasp_tool'")
        return f"{side}_grasp_tool"

    def _tcp_pos(self, model: mujoco.MjModel, data: mujoco.MjData, side: ArmSide, frame: Literal["grasp_tool"] = "grasp_tool") -> np.ndarray:
        mujoco.mj_forward(model, data)
        site_name = self._site_name(side, frame)
        site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        if site_id < 0:
            raise ValueError(f"MuJoCo site not found: {site_name}")
        return data.site_xpos[site_id].copy()

    def _tcp_xmat(self, model: mujoco.MjModel, data: mujoco.MjData, side: ArmSide, frame: Literal["grasp_tool"] = "grasp_tool") -> np.ndarray:
        mujoco.mj_forward(model, data)
        site_name = self._site_name(side, frame)
        site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        if site_id < 0:
            raise ValueError(f"MuJoCo site not found: {site_name}")
        return data.site_xmat[site_id].reshape(3, 3).copy()

    def _orientation_error(self, model: mujoco.MjModel, data: mujoco.MjData, side: ArmSide, target_xmat: np.ndarray, frame: Literal["grasp_tool"] = "grasp_tool") -> float:
        current = self._tcp_xmat(model, data, side, frame)
        delta = np.asarray(target_xmat, dtype=np.float64).reshape(3, 3).T @ current
        cos_angle = max(-1.0, min(1.0, (float(np.trace(delta)) - 1.0) * 0.5))
        return float(np.arccos(cos_angle))

    def _table_clearance_adjusted_grasp_pos(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        side: ArmSide,
        object_body: str,
        grasp_pos: np.ndarray,
        target_xmat: np.ndarray,
        *,
        control_frame: Literal["grasp_tool"],
        enabled: bool,
        pad_table_clearance: float,
    ) -> np.ndarray:
        if not enabled:
            return grasp_pos
        half_height, _is_box = self._object_half_height(model, object_body)
        if half_height <= 0.0:
            return grasp_pos
        min_pad_lowest_z = self._min_pad_lowest_z_offset_at_target_pose(
            model,
            data,
            side,
            target_xmat,
            control_frame=control_frame,
        )
        table_z = grasp_pos[2] - half_height
        min_tcp_z = table_z + float(pad_table_clearance) - min_pad_lowest_z
        adjusted = grasp_pos.copy()
        adjusted[2] = max(float(adjusted[2]), float(min_tcp_z))
        return adjusted

    def _object_half_height(self, model: mujoco.MjModel, object_body: str) -> tuple[float, bool]:
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, object_body)
        if body_id < 0:
            return 0.0, False
        for geom_id in range(model.ngeom):
            if int(model.geom_bodyid[geom_id]) != int(body_id):
                continue
            if model.geom_type[geom_id] == mujoco.mjtGeom.mjGEOM_SPHERE:
                return float(model.geom_size[geom_id][0]), False
            if model.geom_type[geom_id] == mujoco.mjtGeom.mjGEOM_BOX:
                return float(model.geom_size[geom_id][2]), True
        return 0.0, False

    def _min_pad_lowest_z_offset_at_target_pose(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        side: ArmSide,
        target_xmat: np.ndarray,
        *,
        control_frame: Literal["grasp_tool"],
    ) -> float:
        mujoco.mj_forward(model, data)
        site_name = self._site_name(side, control_frame)
        tcp_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        if tcp_site_id < 0:
            raise ValueError(f"MuJoCo site not found: {site_name}")
        tcp_pos = data.site_xpos[tcp_site_id].copy()
        tcp_xmat = data.site_xmat[tcp_site_id].reshape(3, 3).copy()
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
            geom_axes_world = data.geom_xmat[geom_id].reshape(3, 3)
            geom_axes_tcp = tcp_xmat.T @ geom_axes_world
            half_sizes = np.asarray(model.geom_size[geom_id][:3], dtype=np.float64)
            # The merged pad is longer than the active gripping patch.  Table
            # clearance should protect the contact patch, not the full visual
            # finger length, otherwise the grasp target is raised too much.
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
            # The merged pad physically spans the whole finger, but the active
            # gripping envelope still needs the old two-patch clearance depth.
            # Shallower values let the pad hit the table before the box.
            lowest = min(lowest, -0.035)
        return lowest

    def _pad_geom_names(self, side: ArmSide) -> tuple[str, ...]:
        return (
            f"{side}_gripper_finger_link1_pad",
            f"{side}_gripper_finger_link2_pad",
        )

    def _topdown_xmat(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        side: ArmSide,
        mode: Literal["palm_down", "x_forward", "x_side", "current"],
        frame: Literal["grasp_tool"] = "grasp_tool",
    ) -> np.ndarray:
        if mode == "current":
            site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, self._site_name(side, frame))
            return data.site_xmat[site_id].reshape(3, 3).copy()
        if mode == "palm_down":
            y_axis = np.array([0.0, 0.0, 1.0])
            z_axis = np.array([0.0, -1.0, 0.0])
            x_axis = np.cross(y_axis, z_axis)
            x_axis = x_axis / np.linalg.norm(x_axis)
            return np.column_stack((x_axis, y_axis, z_axis))
        if mode == "x_side":
            x_axis = np.array([0.0, 1.0 if side == "left" else -1.0, 0.0])
        else:
            x_axis = np.array([1.0, 0.0, 0.0])
        y_axis = np.array([0.0, 0.0, 1.0])
        z_axis = np.cross(x_axis, y_axis)
        z_axis = z_axis / np.linalg.norm(z_axis)
        x_axis = np.cross(y_axis, z_axis)
        x_axis = x_axis / np.linalg.norm(x_axis)
        return np.column_stack((x_axis, y_axis, z_axis))
