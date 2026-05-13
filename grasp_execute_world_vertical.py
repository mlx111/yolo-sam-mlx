from __future__ import annotations

import numpy as np
import spatialmath as sm
import mujoco

from manipulator_grasp.arm.motion_planning import *


def _camera_pose_world(model: mujoco.MjModel, data: mujoco.MjData, camera_name: str = "cam1") -> sm.SE3:
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
    if cam_id < 0:
        raise ValueError(f"Camera not found in MuJoCo model: {camera_name}")

    t_wc_raw = data.cam_xpos[cam_id].copy()
    r_wc_raw = data.cam_xmat[cam_id].reshape(3, 3).copy()
    T_wc = sm.SE3.Rt(sm.SO3(r_wc_raw), t_wc_raw) * sm.SE3.Rx(np.pi)
    return T_wc


def _grasp_pose_world(model: mujoco.MjModel, data: mujoco.MjData, gg, camera_name: str = "cam1") -> sm.SE3:
    T_wc = _camera_pose_world(model, data, camera_name)
    T_co = sm.SE3.Trans(gg.translations[0]) * sm.SE3(
        sm.SO3.TwoVectors(x=gg.rotation_matrices[0][:, 0], y=gg.rotation_matrices[0][:, 1])
    )
    return T_wc * T_co


def _run_trajectory(env, robot, action, planner, time_seconds: float):
    time_array = [0.0, time_seconds]
    planner_array = [planner]
    total_time = np.sum(time_array)
    time_step_num = round(total_time / 0.002) + 1
    times = np.linspace(0.0, total_time, time_step_num)
    time_cumsum = np.cumsum(time_array)
    for timei in times:
        for j in range(len(time_cumsum)):
            if timei == 0.0:
                break
            if timei <= time_cumsum[j]:
                planner_interpolate = planner_array[j - 1].interpolate(timei - time_cumsum[j - 1])
                if isinstance(planner_interpolate, np.ndarray):
                    joint = planner_interpolate
                    robot.move_joint(joint)
                else:
                    robot.move_cartesian(planner_interpolate)
                    joint = robot.get_joint()
                action[:6] = joint
                env.step(action)
                break


def _world_grasp_debug_geometries(T_wo: sm.SE3):
    import open3d as o3d

    geometries = []

    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.12)
    frame.transform(T_wo.A)
    geometries.append(frame)

    center = o3d.geometry.TriangleMesh.create_sphere(radius=0.015)
    center.paint_uniform_color([1.0, 0.35, 0.2])
    center.translate(T_wo.t)
    geometries.append(center)

    approach_dir = np.asarray(T_wo.R)[:, 0]
    approach_dir = approach_dir / max(float(np.linalg.norm(approach_dir)), 1e-12)
    line_points = np.vstack([
        T_wo.t,
        T_wo.t + approach_dir * 0.15,
    ])
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(line_points)
    line_set.lines = o3d.utility.Vector2iVector([[0, 1]])
    line_set.colors = o3d.utility.Vector3dVector([[0.9, 0.2, 0.2]])
    geometries.append(line_set)

    return geometries


def visualize_world_grasp_pose(T_wo: sm.SE3, *, cloud=None):
    import open3d as o3d

    geoms = _world_grasp_debug_geometries(T_wo)
    if cloud is not None:
        geoms = [cloud, *geoms]
    o3d.visualization.draw_geometries(geoms)


def execute_grasp_world_vertical(env, gg, camera_name: str = "cam1", *, visual_debug: bool = True):
    """Execute grasp using a world-frame grasp pose and world-up lifting."""
    robot = env.robot
    action = np.zeros(7)
    data = env.mj_data
    model = env.mj_model

    q0 = robot.get_joint()
    T_wo = _grasp_pose_world(model, data, gg, camera_name=camera_name)
    print("[INFO] World-frame grasp pose T_wo:", T_wo)
    print("[INFO] Grasp translation (world):", np.asarray(T_wo.t, dtype=float).tolist())
    print("[INFO] Grasp approach axis in world:", np.asarray(T_wo.R)[:, 0].tolist())
    print("[INFO] Grasp width:", float(gg.width[0]) if hasattr(gg, "width") and len(gg.width) > 0 else "n/a")

    if visual_debug:
        try:
            visualize_world_grasp_pose(T_wo)
        except Exception as exc:
            print(f"[Warning] World grasp visualization skipped: {exc}")

    # Keep the original safe pre-grasp joint posture before cartesian approach.
    time1 = 1
    q1 = np.array([0.0, 0.0, np.pi / 2 * 0, 0.0, 0.0, 0.0])
    parameter0 = JointParameter(q0, q1)
    velocity_parameter0 = QuinticVelocityParameter(time1)
    trajectory_parameter0 = TrajectoryParameter(parameter0, velocity_parameter0)
    planner1 = TrajectoryPlanner(trajectory_parameter0)
    _run_trajectory(env, robot, action, planner1, time1)

    # Approach along the grasp approach axis in world coordinates.
    time2 = 1
    robot.set_joint(q1)
    T1 = robot.get_cartesian()
    T2 = T_wo * sm.SE3(-0.1, 0.0, 0.0)
    position_parameter1 = LinePositionParameter(T1.t, T2.t)
    attitude_parameter1 = OneAttitudeParameter(sm.SO3(T1.R), sm.SO3(T2.R))
    cartesian_parameter1 = CartesianParameter(position_parameter1, attitude_parameter1)
    velocity_parameter1 = QuinticVelocityParameter(time2)
    trajectory_parameter1 = TrajectoryParameter(cartesian_parameter1, velocity_parameter1)
    planner2 = TrajectoryPlanner(trajectory_parameter1)
    _run_trajectory(env, robot, action, planner2, time2)

    # Final grasp motion.
    time3 = 1
    T3 = T_wo
    position_parameter2 = LinePositionParameter(T2.t, T3.t)
    attitude_parameter2 = OneAttitudeParameter(sm.SO3(T2.R), sm.SO3(T3.R))
    cartesian_parameter2 = CartesianParameter(position_parameter2, attitude_parameter2)
    velocity_parameter2 = QuinticVelocityParameter(time3)
    trajectory_parameter2 = TrajectoryParameter(cartesian_parameter2, velocity_parameter2)
    planner3 = TrajectoryPlanner(trajectory_parameter2)
    _run_trajectory(env, robot, action, planner3, time3)

    for _ in range(1000):
        action[-1] += 0.2
        action[-1] = np.min([action[-1], 255])
        env.step(action)

    # Lift strictly in world Z to reduce side-view frame ambiguity.
    time4 = 1
    T4 = sm.SE3.Trans(0.0, 0.0, 0.3) * T3
    position_parameter3 = LinePositionParameter(T3.t, T4.t)
    attitude_parameter3 = OneAttitudeParameter(sm.SO3(T3.R), sm.SO3(T4.R))
    cartesian_parameter3 = CartesianParameter(position_parameter3, attitude_parameter3)
    velocity_parameter3 = QuinticVelocityParameter(time4)
    trajectory_parameter3 = TrajectoryParameter(cartesian_parameter3, velocity_parameter3)
    planner4 = TrajectoryPlanner(trajectory_parameter3)
    _run_trajectory(env, robot, action, planner4, time4)

    # Move to a fixed transfer location while preserving orientation.
    time5 = 1
    T5 = sm.SE3.Trans(0.3, 0.3, T4.t[2]) * sm.SE3(sm.SO3(T4.R))
    position_parameter4 = LinePositionParameter(T4.t, T5.t)
    attitude_parameter4 = OneAttitudeParameter(sm.SO3(T4.R), sm.SO3(T5.R))
    cartesian_parameter4 = CartesianParameter(position_parameter4, attitude_parameter4)
    velocity_parameter4 = QuinticVelocityParameter(time5)
    trajectory_parameter4 = TrajectoryParameter(cartesian_parameter4, velocity_parameter4)
    planner5 = TrajectoryPlanner(trajectory_parameter4)
    _run_trajectory(env, robot, action, planner5, time5)

    time6 = 1
    T6 = sm.SE3.Trans(0.0, 0.0, -0.1) * T5
    position_parameter6 = LinePositionParameter(T5.t, T6.t)
    attitude_parameter6 = OneAttitudeParameter(sm.SO3(T5.R), sm.SO3(T6.R))
    cartesian_parameter6 = CartesianParameter(position_parameter6, attitude_parameter6)
    velocity_parameter6 = QuinticVelocityParameter(time6)
    trajectory_parameter6 = TrajectoryParameter(cartesian_parameter6, velocity_parameter6)
    planner6 = TrajectoryPlanner(trajectory_parameter6)
    _run_trajectory(env, robot, action, planner6, time6)

    for _ in range(1000):
        action[-1] -= 0.2
        action[-1] = np.max([action[-1], 0])
        env.step(action)

    time7 = 1
    T7 = sm.SE3.Trans(0.0, 0.0, 0.1) * T6
    position_parameter7 = LinePositionParameter(T6.t, T7.t)
    attitude_parameter7 = OneAttitudeParameter(sm.SO3(T6.R), sm.SO3(T7.R))
    cartesian_parameter7 = CartesianParameter(position_parameter7, attitude_parameter7)
    velocity_parameter7 = QuinticVelocityParameter(time7)
    trajectory_parameter7 = TrajectoryParameter(cartesian_parameter7, velocity_parameter7)
    planner7 = TrajectoryPlanner(trajectory_parameter7)
    _run_trajectory(env, robot, action, planner7, time7)

    time8 = 1
    q8 = robot.get_joint()
    q9 = q0
    parameter8 = JointParameter(q8, q9)
    velocity_parameter8 = QuinticVelocityParameter(time8)
    trajectory_parameter8 = TrajectoryParameter(parameter8, velocity_parameter8)
    planner8 = TrajectoryPlanner(trajectory_parameter8)
    _run_trajectory(env, robot, action, planner8, time8)
