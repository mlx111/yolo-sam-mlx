import argparse
from dataclasses import dataclass

import mujoco
import numpy as np

from skills.base.base_motion_skill import load_skill as load_base_motion
from skills.base.gripper_skill import load_skill as load_gripper
from skills.base.left_arm_move_skill import load_skill as load_left_arm
from skills.base.right_arm_move_skill import load_skill as load_right_arm
from skills.base.torso_move_skill import load_skill as load_torso
from skills.primitives.left_retreat_hand_skill import load_skill as load_left_retreat
from skills.primitives.right_retreat_hand_skill import load_skill as load_right_retreat
from skills.primitives.safe_transport_pose_skill import load_skill as load_safe_transport
from skills.primitives.torso_set_height_skill import load_skill as load_torso_height


@dataclass
class Check:
    name: str
    passed: bool
    value: float
    threshold: float
    detail: str


def parse_args():
    parser = argparse.ArgumentParser(description="Strict robot-only control smoke test.")
    parser.add_argument("--model", default="model.xml")
    return parser.parse_args()


def joint_qpos(model, data, joint_name):
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    if joint_id < 0:
        raise ValueError(f"MuJoCo joint not found: {joint_name}")
    return float(data.qpos[model.jnt_qposadr[joint_id]])


def set_joint_qpos(model, data, joint_name, value):
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    if joint_id < 0:
        raise ValueError(f"MuJoCo joint not found: {joint_name}")
    data.qpos[model.jnt_qposadr[joint_id]] = value
    data.qvel[model.jnt_dofadr[joint_id]] = 0.0
    actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{joint_name}_pos")
    if actuator_id >= 0:
        low, high = model.actuator_ctrlrange[actuator_id]
        data.ctrl[actuator_id] = np.clip(value, low, high)


def tcp_pos(model, data, side):
    mujoco.mj_forward(model, data)
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"{side}_hand_tcp")
    if site_id < 0:
        raise ValueError(f"MuJoCo site not found: {side}_hand_tcp")
    return data.site_xpos[site_id].copy()


def check(name, value, threshold, detail):
    return Check(name, bool(value <= threshold), float(value), float(threshold), detail)


def print_check(result):
    status = "PASS" if result.passed else "FAIL"
    print(f"{status} {result.name}: value={result.value:.9f}, threshold={result.threshold:.9f} | {result.detail}")


def initialize_robot(model, data):
    for joint_name in ("base_x", "base_y", "base_yaw"):
        set_joint_qpos(model, data, joint_name, 0.0)
    for joint_name in ("torso_joint1", "torso_joint2", "torso_joint3", "torso_joint4"):
        set_joint_qpos(model, data, joint_name, 0.0)
    for prefix in ("left_arm", "right_arm"):
        for idx in range(1, 8):
            set_joint_qpos(model, data, f"{prefix}_joint{idx}", 0.0)
    for side in ("left", "right"):
        for idx in (1, 2):
            set_joint_qpos(model, data, f"{side}_gripper_finger_joint{idx}", 0.0)
    mujoco.mj_forward(model, data)


def main():
    args = parse_args()
    model = mujoco.MjModel.from_xml_path(args.model)
    data = mujoco.MjData(model)
    initialize_robot(model, data)

    results = []

    base = load_base_motion()
    base_target = np.array([0.12, -0.08, 0.18], dtype=np.float64)
    base_result = base.move_to_pose(model, data, base_target, steps=1, settle_steps=0, direct_qpos=True)
    base_error = float(np.linalg.norm(base_result.final_qpos - base_target))
    results.append(check("base_move_direct_qpos", base_error, 1e-9, f"final={np.round(base_result.final_qpos, 9).tolist()}"))

    torso = load_torso()
    torso_target = np.array([0.05, -0.04, 0.03, 0.02], dtype=np.float64)
    torso_result = torso.move_to_posture(model, data, torso_target, steps=1, settle_steps=0, direct_qpos=True)
    results.append(check("torso_move_direct_qpos", torso_result.final_error, 1e-9, f"final={np.round(torso_result.final_qpos, 9).tolist()}"))

    torso_height = load_torso_height()
    height_result = torso_height.execute_recovery_action(
        model,
        data,
        {"height_level": "mid", "steps": 1, "settle_steps": 0, "direct_qpos": True},
    )
    results.append(check("torso_set_height_mid", height_result.final_error, 1e-9, f"final={np.round(height_result.final_qpos, 9).tolist()}"))

    gripper = load_gripper()
    gripper.apply(model, data, "open", direct_qpos=True)
    gripper_open = max(abs(joint_qpos(model, data, "left_gripper_finger_joint1")), abs(joint_qpos(model, data, "right_gripper_finger_joint1")))
    results.append(check("gripper_open_direct", gripper_open, 1e-9, f"max_open_qpos={gripper_open:.9f}"))
    gripper.apply(model, data, "close", direct_qpos=True)
    close_error = max(
        abs(joint_qpos(model, data, "left_gripper_finger_joint1") - 0.025),
        abs(joint_qpos(model, data, "right_gripper_finger_joint1") - 0.025),
    )
    results.append(check("gripper_close_direct", close_error, 1e-9, f"max_close_error={close_error:.9f}"))

    left_arm = load_left_arm()
    left_start = tcp_pos(model, data, "left")
    left_target = left_start + np.array([0.01, -0.005, 0.008], dtype=np.float64)
    left_direct_result = left_arm.move_to_position(
        model,
        data,
        left_target,
        steps=1,
        settle_steps=0,
        direct_qpos=True,
        fail_threshold=0.03,
    )
    results.append(
        check(
            "left_arm_tcp_move_direct_qpos",
            left_direct_result.final_error,
            0.03,
            f"target={np.round(left_target, 6).tolist()}",
        )
    )
    left_result = left_arm.move_to_position(model, data, left_target, steps=2500, settle_steps=1500, fail_threshold=0.03)
    results.append(check("left_arm_tcp_move", left_result.final_error, 0.03, f"target={np.round(left_target, 6).tolist()}"))

    right_arm = load_right_arm()
    right_start = tcp_pos(model, data, "right")
    right_target = right_start + np.array([0.01, 0.005, 0.008], dtype=np.float64)
    right_direct_result = right_arm.move_to_position(
        model,
        data,
        right_target,
        steps=1,
        settle_steps=0,
        direct_qpos=True,
        fail_threshold=0.03,
    )
    results.append(
        check(
            "right_arm_tcp_move_direct_qpos",
            right_direct_result.final_error,
            0.03,
            f"target={np.round(right_target, 6).tolist()}",
        )
    )
    right_result = right_arm.move_to_position(model, data, right_target, steps=2500, settle_steps=1500, fail_threshold=0.03)
    results.append(check("right_arm_tcp_move", right_result.final_error, 0.03, f"target={np.round(right_target, 6).tolist()}"))

    left_retreat = load_left_retreat()
    left_before_retreat = tcp_pos(model, data, "left")
    left_retreat_result = left_retreat.execute_recovery_action(
        model,
        data,
        {"retreat_dx": 0.0, "retreat_dy": 0.0, "retreat_dz": 0.015, "servo_steps": 1200, "settle_steps": 400, "fail_threshold": 0.04},
    )
    left_retreat_delta = tcp_pos(model, data, "left") - left_before_retreat
    left_retreat_error = abs(float(left_retreat_delta[2]) - 0.015)
    results.append(check("left_retreat_z_delta", left_retreat_error, 0.015, f"delta={np.round(left_retreat_delta, 6).tolist()}, result_error={left_retreat_result.final_error}"))

    right_retreat = load_right_retreat()
    right_before_retreat = tcp_pos(model, data, "right")
    right_retreat_result = right_retreat.execute_recovery_action(
        model,
        data,
        {"retreat_dx": 0.0, "retreat_dy": 0.0, "retreat_dz": 0.015, "servo_steps": 1200, "settle_steps": 400, "fail_threshold": 0.04},
    )
    right_retreat_delta = tcp_pos(model, data, "right") - right_before_retreat
    right_retreat_error = abs(float(right_retreat_delta[2]) - 0.015)
    results.append(check("right_retreat_z_delta", right_retreat_error, 0.015, f"delta={np.round(right_retreat_delta, 6).tolist()}, result_error={right_retreat_result.final_error}"))

    safe = load_safe_transport()
    safe_result = safe.execute_recovery_action(model, data, {"posture": "home", "steps": 1, "settle_steps": 0, "direct_qpos": True})
    results.append(check("safe_transport_home_direct", safe_result.final_error, 1e-9, f"final_error={safe_result.final_error:.9f}"))

    for result in results:
        print_check(result)

    failed = [result for result in results if not result.passed]
    print(f"strict_control_summary: passed={len(results) - len(failed)} failed={len(failed)} total={len(results)}")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
