import argparse

import mujoco
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Smoke test planar mobile base joints in the mobile R1Pro MuJoCo model.")
    parser.add_argument("--model", default="r1pro_grasp_scene_mobile.xml")
    parser.add_argument("--base-x", type=float, default=0.2)
    parser.add_argument("--base-y", type=float, default=-0.1)
    parser.add_argument("--base-yaw", type=float, default=0.25)
    parser.add_argument("--steps", type=int, default=1000)
    return parser.parse_args()


def body_pos(model, data, name):
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    if body_id < 0:
        raise ValueError(f"MuJoCo body not found: {name}")
    return data.xpos[body_id].copy()


def set_ctrl(model, data, actuator_name, value):
    actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name)
    if actuator_id < 0:
        raise ValueError(f"MuJoCo actuator not found: {actuator_name}")
    low, high = model.actuator_ctrlrange[actuator_id]
    data.ctrl[actuator_id] = np.clip(value, low, high)


def joint_qpos(model, data, joint_name):
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    if joint_id < 0:
        raise ValueError(f"MuJoCo joint not found: {joint_name}")
    return float(data.qpos[model.jnt_qposadr[joint_id]])


def main():
    args = parse_args()
    model = mujoco.MjModel.from_xml_path(args.model)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    tracked_bodies = ("mobile_base_yaw", "steer_motor_link1", "torso_link1", "target_cube")
    start = {name: body_pos(model, data, name) for name in tracked_bodies}

    set_ctrl(model, data, "base_x_pos", args.base_x)
    set_ctrl(model, data, "base_y_pos", args.base_y)
    set_ctrl(model, data, "base_yaw_pos", args.base_yaw)
    for _ in range(max(args.steps, 0)):
        mujoco.mj_step(model, data)
    mujoco.mj_forward(model, data)

    end = {name: body_pos(model, data, name) for name in tracked_bodies}
    print("base joints:", {
        "base_x": round(joint_qpos(model, data, "base_x"), 6),
        "base_y": round(joint_qpos(model, data, "base_y"), 6),
        "base_yaw": round(joint_qpos(model, data, "base_yaw"), 6),
    })
    for name in tracked_bodies:
        print(
            f"{name}: start={np.round(start[name], 6).tolist()} "
            f"end={np.round(end[name], 6).tolist()} "
            f"delta={np.round(end[name] - start[name], 6).tolist()}"
        )


if __name__ == "__main__":
    main()
