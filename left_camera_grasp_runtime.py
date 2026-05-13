from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
import open3d as o3d
import spatialmath as sm

ROOT_DIR = Path(__file__).resolve().parent
import sys
sys.path.append(str(ROOT_DIR / 'graspnet-baseline' / 'models'))
sys.path.append(str(ROOT_DIR / 'graspnet-baseline' / 'dataset'))
sys.path.append(str(ROOT_DIR / 'graspnet-baseline' / 'utils'))
sys.path.append(str(ROOT_DIR / 'manipulator_grasp'))
sys.path.append(str(ROOT_DIR / 'anygrasp_sdk' / 'grasp_detection'))

from gsnet import AnyGrasp
from graspnetAPI import GraspGroup
from collision_detector import ModelFreeCollisionDetector

from grasp_frame_utils import filter_grasps_by_world_tilt
from camera_pose_mujoco import POINT_TRANSFORMS
from manipulator_grasp.arm.motion_planning import *  # noqa: F403


LEFT_CAMERA_NAME = 'cam1'
WORLD_VERTICAL = np.array([0.0, 0.0, 1.0], dtype=float)
CHECKPOINT = ROOT_DIR / 'anygrasp_sdk' / 'grasp_detection' / 'log' / 'checkpoint_detection.tar'


class AnyGraspConfig:
    def __init__(self):
        self.checkpoint_path = str(CHECKPOINT)
        self.max_gripper_width = 0.1
        self.gripper_height = 0.05
        self.top_down_grasp = True
        self.debug = False


def extract_points_colors(end_points):
    if 'point_clouds' not in end_points or 'cloud_colors' not in end_points:
        raise KeyError("end_points must contain 'point_clouds' and 'cloud_colors'")

    points = end_points['point_clouds']
    if hasattr(points, 'detach'):
        points = points.detach().cpu().numpy()
    else:
        points = np.asarray(points)
    points = np.asarray(points, dtype=np.float32)
    if points.ndim == 3 and points.shape[0] == 1:
        points = points[0]

    colors = end_points['cloud_colors']
    if hasattr(colors, 'detach'):
        colors = colors.detach().cpu().numpy()
    else:
        colors = np.asarray(colors)
    colors = np.asarray(colors, dtype=np.float32)

    return points, colors


def build_anygrasp_grasps(end_points, cloud_o3d, *, visual: bool = False) -> tuple[GraspGroup, AnyGrasp, np.ndarray, np.ndarray]:
    cfgs = AnyGraspConfig()
    anygrasp = AnyGrasp(cfgs)
    anygrasp.load_net()

    points, colors = extract_points_colors(end_points)
    if points.size == 0:
        return GraspGroup(), anygrasp, points, colors

    gg, cloud = anygrasp.get_grasp(
        points,
        colors,
        lims=None,
        apply_object_mask=True,
        dense_grasp=False,
        collision_detection=True,
    )
    if gg is None or len(gg) == 0:
        return GraspGroup(), anygrasp, points, colors

    gg = gg.nms().sort_by_score()
    if len(gg) == 0:
        return GraspGroup(), anygrasp, points, colors

    if visual:
        grippers = gg.to_open3d_geometry_list()
        if cloud is not None:
            o3d.visualization.draw_geometries([cloud, *grippers])
        else:
            o3d.visualization.draw_geometries(grippers)

    return gg, anygrasp, points, colors


def get_left_camera_pose_world(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    camera_name: str = LEFT_CAMERA_NAME,
) -> sm.SE3:
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
    if cam_id < 0:
        raise ValueError(f'Camera not found in MuJoCo model: {camera_name}')

    t_wc_raw = data.cam_xpos[cam_id].copy()
    r_wc_raw = data.cam_xmat[cam_id].reshape(3, 3).copy()
    T_wc_raw = sm.SE3.Rt(sm.SO3(r_wc_raw), t_wc_raw)
    left_align = sm.SE3(sm.SO3(POINT_TRANSFORMS['left']))
    return T_wc_raw * left_align


def get_left_camera_rotation_world_from_mujoco(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    camera_name: str = LEFT_CAMERA_NAME,
) -> np.ndarray:
    return np.asarray(get_left_camera_pose_world(model, data, camera_name).R, dtype=float)


def get_world_vertical_grasp_from_points(
    end_points,
    cloud_o3d,
    camera_rotation_world_from_cam: np.ndarray,
    *,
    visual: bool = False,
    angle_threshold_deg: float = 30.0,
) -> GraspGroup:
    gg, _, _, _ = build_anygrasp_grasps(end_points, cloud_o3d, visual=visual)
    if gg is None or len(gg) == 0:
        print('No Grasp detected after masking')
        return GraspGroup()

    cloud_np = np.asarray(cloud_o3d.points, dtype=np.float32)
    if cloud_np.size > 0:
        mfcdetector = ModelFreeCollisionDetector(cloud_np, voxel_size=0.005)
        collision_mask = mfcdetector.detect(gg, approach_dist=0.05, collision_thresh=0.005)
        gg = gg[~collision_mask]

    gg = gg.nms().sort_by_score()
    all_grasps = list(gg)
    if len(all_grasps) == 0:
        print('[Warning] No grasps remain after NMS/sorting.')
        return GraspGroup()

    top_grasps = filter_grasps_by_world_tilt(
        all_grasps,
        camera_rotation_world_from_cam,
        world_axis=WORLD_VERTICAL,
        min_tilt_deg=angle_threshold_deg,
        keep_top_k=20,
    )

    if len(top_grasps) == 0:
        print('[Warning] No grasps remain after side-grasp filtering.')
        return GraspGroup()

    best_grasp = top_grasps[0]
    new_gg = GraspGroup()
    new_gg.add(best_grasp)
    print('grasp score:', best_grasp.score)

    if visual:
        grippers = new_gg.to_open3d_geometry_list()
        if cloud_o3d is not None:
            o3d.visualization.draw_geometries([cloud_o3d, *grippers])
        else:
            o3d.visualization.draw_geometries(grippers)

    return new_gg


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


def _grasp_pose_world_left(model: mujoco.MjModel, data: mujoco.MjData, gg, camera_name: str = LEFT_CAMERA_NAME) -> sm.SE3:
    T_wc = get_left_camera_pose_world(model, data, camera_name)
    T_co = sm.SE3.Trans(gg.translations[0]) * sm.SE3(
        sm.SO3.TwoVectors(x=gg.rotation_matrices[0][:, 0], y=gg.rotation_matrices[0][:, 1])
    )
    return T_wc * T_co


def execute_grasp_world_vertical_left(env, gg, camera_name: str = LEFT_CAMERA_NAME, *, visual_debug: bool = True):
    robot = env.robot
    action = np.zeros(7)
    data = env.mj_data
    model = env.mj_model

    q0 = robot.get_joint()
    T_wo = _grasp_pose_world_left(model, data, gg, camera_name=camera_name)
    print('[INFO] World-frame grasp pose T_wo:', T_wo)
    print('[INFO] Grasp translation (world):', np.asarray(T_wo.t, dtype=float).tolist())
    print('[INFO] Grasp approach axis in world:', np.asarray(T_wo.R)[:, 0].tolist())
    print('[INFO] Grasp width:', float(gg.width[0]) if hasattr(gg, 'width') and len(gg.width) > 0 else 'n/a')

    if visual_debug:
        try:
            frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.12)
            frame.transform(T_wo.A)
            center = o3d.geometry.TriangleMesh.create_sphere(radius=0.015)
            center.paint_uniform_color([1.0, 0.35, 0.2])
            center.translate(T_wo.t)
            approach_dir = np.asarray(T_wo.R)[:, 0]
            approach_dir = approach_dir / max(float(np.linalg.norm(approach_dir)), 1e-12)
            line_points = np.vstack([T_wo.t, T_wo.t + approach_dir * 0.15])
            line_set = o3d.geometry.LineSet()
            line_set.points = o3d.utility.Vector3dVector(line_points)
            line_set.lines = o3d.utility.Vector2iVector([[0, 1]])
            line_set.colors = o3d.utility.Vector3dVector([[0.9, 0.2, 0.2]])
            o3d.visualization.draw_geometries([frame, center, line_set])
        except Exception as exc:
            print(f'[Warning] World grasp visualization skipped: {exc}')

    time1 = 1
    q1 = np.array([0.0, 0.0, np.pi / 2 * 0, 0.0, 0.0, 0.0])
    parameter0 = JointParameter(q0, q1)
    velocity_parameter0 = QuinticVelocityParameter(time1)
    trajectory_parameter0 = TrajectoryParameter(parameter0, velocity_parameter0)
    planner1 = TrajectoryPlanner(trajectory_parameter0)
    _run_trajectory(env, robot, action, planner1, time1)

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

    time4 = 1
    T4 = sm.SE3.Trans(0.0, 0.0, 0.3) * T3
    position_parameter3 = LinePositionParameter(T3.t, T4.t)
    attitude_parameter3 = OneAttitudeParameter(sm.SO3(T3.R), sm.SO3(T4.R))
    cartesian_parameter3 = CartesianParameter(position_parameter3, attitude_parameter3)
    velocity_parameter3 = QuinticVelocityParameter(time4)
    trajectory_parameter3 = TrajectoryParameter(cartesian_parameter3, velocity_parameter3)
    planner4 = TrajectoryPlanner(trajectory_parameter3)
    _run_trajectory(env, robot, action, planner4, time4)

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
