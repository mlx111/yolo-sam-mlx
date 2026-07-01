from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from .control import FR5MuJoCoController


Q_HOME = np.array([0.0, 0.0, 0.0, -1.57079, 0.0, -0.7853], dtype=float)


@dataclass
class SkillResult:
    skill: str
    success: bool
    reason: str = "ok"
    contact: dict[str, Any] = field(default_factory=dict)
    gripper_action: float | None = None
    object_pos: Any = None
    target_pos: list[float] | None = None
    final_pos: list[float] | None = None
    pos_error: float | None = None
    target_joint: list[float] | None = None
    final_joint: list[float] | None = None
    joint_error: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "skill": self.skill,
            "success": bool(self.success),
            "reason": self.reason,
            "contact": self.contact,
            "gripper_action": self.gripper_action,
            "object_pos": self.object_pos,
            "target_pos": self.target_pos,
            "final_pos": self.final_pos,
            "pos_error": self.pos_error,
            "target_joint": self.target_joint,
            "final_joint": self.final_joint,
            "joint_error": self.joint_error,
            "extra": self.extra,
        }


def _smoothstep(x: float) -> float:
    x = float(np.clip(x, 0.0, 1.0))
    return x * x * x * (10.0 + x * (-15.0 + 6.0 * x))


def _normalize_quat(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=float).reshape(4)
    norm = np.linalg.norm(quat)
    if norm <= 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    return quat / norm


def _interp_quat(q0: np.ndarray, q1: np.ndarray, alpha: float) -> np.ndarray:
    q0 = _normalize_quat(q0)
    q1 = _normalize_quat(q1)
    if float(np.dot(q0, q1)) < 0.0:
        q1 = -q1
    return _normalize_quat((1.0 - alpha) * q0 + alpha * q1)


def _rotation_matrix_to_quat(rot: np.ndarray) -> np.ndarray:
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, np.asarray(rot, dtype=float).reshape(9))
    return _normalize_quat(quat)


def _extract_pose(target: Any) -> tuple[np.ndarray, np.ndarray | None]:
    if hasattr(target, "t"):
        pos = np.asarray(target.t, dtype=float).reshape(3)
        rot = getattr(target, "R", None)
        quat = _rotation_matrix_to_quat(rot) if rot is not None else None
        return pos, quat

    arr = np.asarray(target, dtype=float)
    if arr.shape == (3,):
        return arr.copy(), None
    if arr.shape == (4, 4):
        return arr[:3, 3].copy(), _rotation_matrix_to_quat(arr[:3, :3])

    msg = "cartesian target must be a 3D position, 4x4 transform, or SE3-like object"
    raise ValueError(msg)


class FR5MotionRuntime:
    """Movement skill runtime matching the useful UR5e motion surface."""

    def __init__(
        self,
        controller: FR5MuJoCoController | None = None,
        *,
        scene_path: str | None = None,
        realtime: bool = False,
    ) -> None:
        self.controller = controller or FR5MuJoCoController.from_scene(scene_path)
        self.model = self.controller.model
        self.data = self.controller.data
        self.realtime = realtime
        self.viewer: Any = None
        self.T_wo: Any = None
        self.T_pregrasp: Any = None
        self.metrics: dict[str, Any] = {"skill_results": []}
        self.skill_results: list[SkillResult] = []

    @classmethod
    def from_scene(cls, scene_path: str | None = None, *, realtime: bool = False) -> "FR5MotionRuntime":
        return cls(scene_path=scene_path, realtime=realtime)

    @property
    def qpos(self) -> np.ndarray:
        return self.controller.qpos

    @property
    def tcp_pos(self) -> np.ndarray:
        return self.controller.tcp_pos

    @property
    def tcp_quat(self) -> np.ndarray:
        return self.controller.tcp_quat

    def reset_home(self) -> None:
        self.controller.reset_home()
        self._sync_viewer()

    def set_viewer(self, viewer: Any | None) -> None:
        self.viewer = viewer

    def _sync_viewer(self) -> None:
        if self.viewer is not None:
            self.viewer.sync()

    def _step_sleep(self, step_start: float) -> None:
        if not self.realtime:
            return
        sleep_time = self.model.opt.timestep - (time.time() - step_start)
        if sleep_time > 0.0:
            time.sleep(sleep_time)

    def _after_step(self, step_start: float) -> None:
        self._sync_viewer()
        self._step_sleep(step_start)

    def _record(self, result: SkillResult) -> SkillResult:
        self.skill_results.append(result)
        self.metrics.setdefault("skill_results", []).append(result.to_dict())
        return result

    def _step_n(self, n: int) -> None:
        q_hold = self.qpos
        for _ in range(max(0, int(n))):
            step_start = time.time()
            self.controller.step_joint_target(q_hold)
            self._after_step(step_start)

    def _current_skill_observation(self) -> dict[str, Any]:
        return {
            "contact": self._contact_summary(),
            "gripper_action": self._gripper_opening(),
            "pinch_distance": self._pinch_distance(),
            "object_pos": None,
            "tcp_pos": self.tcp_pos.tolist(),
            "joint_positions": self.qpos.tolist(),
        }

    def camera_intrinsics(self, camera_name: str = "ee_camera", width: int = 640, height: int = 480) -> dict[str, float]:
        cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
        if cam_id < 0:
            raise KeyError(f"camera not found: {camera_name}")
        fovy = float(self.model.cam_fovy[cam_id])
        fy = 0.5 * float(height) / np.tan(np.deg2rad(fovy) * 0.5)
        return {
            "fx": float(fy),
            "fy": float(fy),
            "cx": float((int(width) - 1) * 0.5),
            "cy": float((int(height) - 1) * 0.5),
            "width": int(width),
            "height": int(height),
            "fovy": fovy,
        }

    def render_camera_rgbd(
        self,
        camera_name: str = "ee_camera",
        *,
        width: int = 640,
        height: int = 480,
        output_dir: str | Path | None = None,
        prefix: str = "fr5_ee_camera",
    ) -> dict[str, Any]:
        import cv2

        output_path = Path(output_dir) if output_dir is not None else Path(__file__).resolve().parents[2] / "output" / "camera_rgbd"
        output_path.mkdir(parents=True, exist_ok=True)
        mujoco.mj_forward(self.model, self.data)

        renderer = mujoco.Renderer(self.model, height=int(height), width=int(width))
        try:
            renderer.update_scene(self.data, camera=camera_name)
            rgb = renderer.render()
            renderer.enable_depth_rendering()
            renderer.update_scene(self.data, camera=camera_name)
            depth_m = renderer.render().astype(np.float32)
            renderer.disable_depth_rendering()
        finally:
            renderer.close()

        rgb_path = output_path / f"{prefix}_rgb.png"
        depth_npy_path = output_path / f"{prefix}_depth_m.npy"
        depth_png_path = output_path / f"{prefix}_depth_mm.png"
        cv2.imwrite(str(rgb_path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        np.save(depth_npy_path, depth_m)
        depth_mm = np.clip(depth_m * 1000.0, 0.0, np.iinfo(np.uint16).max).astype(np.uint16)
        cv2.imwrite(str(depth_png_path), depth_mm)

        cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
        if cam_id < 0:
            raise KeyError(f"camera not found: {camera_name}")
        camera_pose = {
            "name": camera_name,
            "xpos": self.data.cam_xpos[cam_id].copy().tolist(),
            "xmat": self.data.cam_xmat[cam_id].copy().reshape(3, 3).tolist(),
        }
        result = {
            "camera_name": camera_name,
            "rgb_path": str(rgb_path),
            "depth_npy_path": str(depth_npy_path),
            "depth_png_path": str(depth_png_path),
            "intrinsics": self.camera_intrinsics(camera_name, width, height),
            "camera_pose": camera_pose,
            "rgb_shape": list(rgb.shape),
            "depth_shape": list(depth_m.shape),
            "depth_min_m": float(np.nanmin(depth_m)),
            "depth_max_m": float(np.nanmax(depth_m)),
        }
        self.metrics["last_camera_rgbd"] = result
        return result

    def point_cloud_from_camera_mask(
        self,
        mask: np.ndarray,
        depth_m: np.ndarray,
        camera_name: str = "ee_camera",
        *,
        intrinsics: dict[str, float] | None = None,
        max_points: int = 20000,
        max_depth_m: float = 10.0,
    ) -> dict[str, Any]:
        cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
        if cam_id < 0:
            raise KeyError(f"camera not found: {camera_name}")
        depth = np.asarray(depth_m, dtype=np.float32)
        mask_arr = np.asarray(mask)
        if mask_arr.shape[:2] != depth.shape[:2]:
            raise ValueError(f"mask shape {mask_arr.shape[:2]} does not match depth shape {depth.shape[:2]}")
        h, w = depth.shape[:2]
        intr = intrinsics or self.camera_intrinsics(camera_name, w, h)
        valid = (mask_arr > 0) & np.isfinite(depth) & (depth > 0.0) & (depth <= float(max_depth_m))
        if not np.any(valid):
            return {"state": "fail", "info": f"No valid depth in mask within {float(max_depth_m):.3f}m", "point_cloud": None}

        v, u = np.where(valid)
        if len(u) > max_points:
            sample_idx = np.linspace(0, len(u) - 1, int(max_points), dtype=int)
            u = u[sample_idx]
            v = v[sample_idx]
        z = depth[v, u].astype(float)
        x = (u.astype(float) - float(intr["cx"])) * z / float(intr["fx"])
        y = (v.astype(float) - float(intr["cy"])) * z / float(intr["fy"])

        # MuJoCo camera looks along local -Z, with local +X right and +Y up.
        # Pixel coordinates use +Y down, so image points map to [x, -y, -z].
        points_cam = np.stack([x, -y, -z], axis=1)
        rot = self.data.cam_xmat[cam_id].reshape(3, 3)
        pos = self.data.cam_xpos[cam_id].reshape(3)
        points_world = points_cam @ rot.T + pos
        if len(points_world) == 0:
            return {"state": "fail", "info": "No points after projection", "point_cloud": None}

        p_min = points_world.min(axis=0)
        p_max = points_world.max(axis=0)
        center = (p_min + p_max) * 0.5
        return {
            "state": "success",
            "info": f"point cloud generated: {len(points_world)} points",
            "x": float(center[0]),
            "y": float(center[1]),
            "z": float(center[2]),
            "x_min": float(p_min[0]),
            "y_min": float(p_min[1]),
            "z_min": float(p_min[2]),
            "x_max": float(p_max[0]),
            "y_max": float(p_max[1]),
            "z_max": float(p_max[2]),
            "point_count": int(len(points_world)),
            "point_cloud": points_world,
        }

    def _record_skill_result(self, result: SkillResult | dict[str, Any]) -> dict[str, Any]:
        record = result.to_dict() if hasattr(result, "to_dict") else dict(result)
        results = self.metrics.setdefault("skill_results", [])
        if not results or results[-1] != record:
            results.append(record)
        return record

    def _record_basic_skill(
        self,
        skill: str,
        success: bool,
        reason: str = "ok",
        **extra: Any,
    ) -> dict[str, Any]:
        record = {
            "skill": skill,
            "success": bool(success),
            "reason": str(reason),
            "extra": extra,
        }
        self.metrics.setdefault("skill_results", []).append(record)
        return record

    def _move_joints(self, q_target: np.ndarray, duration: float = 1.0) -> SkillResult:
        return self.move_joints(q_target, duration)

    def _move_cartesian(self, target: Any, duration: float = 1.0) -> SkillResult:
        return self.move_cartesian(target, duration)

    def _gripper_open(self, duration: float = 0.5) -> SkillResult:
        return self._move_gripper(0.0, duration=duration, skill="open_gripper")

    def _gripper_close(self, duration: float = 0.5) -> SkillResult:
        return self._move_gripper(0.04, duration=duration, skill="close_gripper")

    def _move_gripper(self, opening_m: float, duration: float, skill: str) -> SkillResult:
        steps = max(1, int(float(duration) / self.model.opt.timestep))
        q_hold = self.qpos
        for _ in range(steps):
            step_start = time.time()
            self.controller.step_joint_target(q_hold, gripper_opening_m=opening_m)
            self._after_step(step_start)
        contact = self._contact_summary()
        return self._record(
            SkillResult(
                skill=skill,
                success=True,
                reason="ok",
                contact=contact,
                gripper_action=float(opening_m),
                extra={"duration": float(duration), "pinch_distance": self._pinch_distance()},
            )
        )

    def _gripper_opening(self) -> float:
        finger_ids = self.controller.ids.finger_actuator_ids
        return float(np.mean([self.data.ctrl[finger_ids[0]], self.data.ctrl[finger_ids[1]]]))

    def _pinch_distance(self) -> float | None:
        try:
            p1 = self.data.geom("finger1_collision").xpos
            p2 = self.data.geom("finger2_collision").xpos
            return float(np.linalg.norm(p1 - p2))
        except KeyError:
            return None

    def _contact_summary(self) -> dict[str, Any]:
        left_contact = False
        right_contact = False
        pairs: list[tuple[str, str]] = []
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            g1 = self.model.geom(contact.geom1).name
            g2 = self.model.geom(contact.geom2).name
            pairs.append((g1, g2))
            joined = f"{g1} {g2}"
            if "finger1" in joined:
                left_contact = True
            if "finger2" in joined:
                right_contact = True
        return {
            "left_contact": left_contact,
            "right_contact": right_contact,
            "contact_count": int(self.data.ncon),
            "pairs": pairs,
        }

    def move_joints(self, q_target: np.ndarray, duration: float = 1.0) -> SkillResult:
        q_start = self.qpos
        q_min, q_max = self.controller.joint_limits
        q_target = np.clip(np.asarray(q_target, dtype=float).reshape(6), q_min, q_max)
        steps = max(1, int(float(duration) / self.model.opt.timestep))

        for i in range(steps):
            step_start = time.time()
            alpha = _smoothstep((i + 1) / steps)
            q_des = (1.0 - alpha) * q_start + alpha * q_target
            self.controller.step_joint_target(q_des)
            self._after_step(step_start)

        settle_steps = max(1, int(0.25 / self.model.opt.timestep))
        for _ in range(settle_steps):
            step_start = time.time()
            self.controller.step_joint_target(q_target)
            self._after_step(step_start)

        final_q = self.qpos
        joint_error = float(np.linalg.norm(final_q - q_target))
        return self._record(
            SkillResult(
                skill="joint_move",
                success=joint_error <= 0.03,
                reason="ok" if joint_error <= 0.03 else "final_joint_error_exceeded",
                contact=self._contact_summary(),
                gripper_action=self._gripper_opening(),
                target_joint=q_target.tolist(),
                final_joint=final_q.tolist(),
                joint_error=joint_error,
                extra={"duration": float(duration)},
            )
        )

    def move_cartesian(self, target: Any, duration: float = 1.0, keep_orientation: bool = True) -> SkillResult:
        target_pos, target_quat = _extract_pose(target)
        start_pos = self.tcp_pos
        start_quat = self.controller.tcp_quat
        if target_quat is None and keep_orientation:
            target_quat = start_quat
        steps = max(1, int(float(duration) / self.model.opt.timestep))
        scratch = mujoco.MjData(self.model)

        ik_failed_count = 0
        for i in range(steps):
            step_start = time.time()
            alpha = _smoothstep((i + 1) / steps)
            goal_pos = (1.0 - alpha) * start_pos + alpha * target_pos
            goal_quat = _interp_quat(start_quat, target_quat, alpha) if target_quat is not None else None
            try:
                self.controller.step_tcp_target(goal_pos, goal_quat=goal_quat)
            except np.linalg.LinAlgError:
                ik_failed_count += 1
                self.controller.step_joint_target(self.qpos)
            self._after_step(step_start)

        settle_steps = max(1, int(0.75 / self.model.opt.timestep))
        for _ in range(settle_steps):
            step_start = time.time()
            try:
                # Run several IK updates against a scratch state before each physics step.
                # This improves Cartesian convergence without teleporting the real model.
                scratch.qpos[:] = self.data.qpos
                scratch.qvel[:] = self.data.qvel
                mujoco.mj_forward(self.model, scratch)
                q_des = scratch.qpos[self.controller.ids.qpos_indices].copy()
                for _ in range(4):
                    q_des = self.controller.ik.compute_q_des(
                        scratch,
                        goal_pos=target_pos,
                        goal_quat=target_quat,
                    )
                    scratch.qpos[self.controller.ids.qpos_indices] = q_des
                    mujoco.mj_forward(self.model, scratch)
                self.controller.step_joint_target(q_des)
            except np.linalg.LinAlgError:
                ik_failed_count += 1
                self.controller.step_joint_target(self.qpos)
            self._after_step(step_start)

        final_pos = self.tcp_pos
        pos_error = float(np.linalg.norm(final_pos - target_pos))
        return self._record(
            SkillResult(
                skill="cartesian_move",
                success=pos_error <= 0.015 and ik_failed_count == 0,
                reason="ok" if pos_error <= 0.015 and ik_failed_count == 0 else "final_pose_error_exceeded",
                contact=self._contact_summary(),
                gripper_action=self._gripper_opening(),
                target_pos=target_pos.tolist(),
                final_pos=final_pos.tolist(),
                pos_error=pos_error,
                extra={"duration": float(duration), "ik_failed_count": ik_failed_count},
            )
        )

    def go_home(self, duration: float = 1.0, q: np.ndarray | None = None) -> SkillResult:
        return self.move_joints(Q_HOME if q is None else q, duration=duration)

    def move_to_pregrasp(self, offset: np.ndarray, duration: float = 1.0) -> SkillResult:
        if self.T_wo is None:
            raise RuntimeError("move_to_pregrasp requires T_wo to be set first")
        grasp_pos, grasp_quat = _extract_pose(self.T_wo)
        target_pos = grasp_pos + np.asarray(offset, dtype=float).reshape(3)
        target = target_pos if grasp_quat is None else _Pose(target_pos, grasp_quat)
        result = self.move_cartesian(target, duration=duration)
        self.T_pregrasp = target
        return result

    def approach_object(self, duration: float = 1.0) -> SkillResult:
        if self.T_wo is None:
            raise RuntimeError("approach_object requires T_wo to be set first")
        return self.move_cartesian(self.T_wo, duration=duration)


@dataclass(frozen=True)
class _Pose:
    t: np.ndarray
    quat: np.ndarray

    @property
    def R(self) -> np.ndarray:
        mat = np.zeros(9)
        mujoco.mju_quat2Mat(mat, self.quat)
        return mat.reshape(3, 3)
