import argparse
import math
import time

import mujoco
from mujoco import viewer as mj_viewer


def parse_args():
    parser = argparse.ArgumentParser(description="View R1Pro gripper open/close motion in MuJoCo.")
    parser.add_argument("--model", default="r1pro_grasp_scene.xml")
    parser.add_argument("--side", choices=["left", "right", "both"], default="left")
    parser.add_argument("--open", type=float, default=0.0)
    parser.add_argument("--close", type=float, default=0.025)
    parser.add_argument("--period", type=float, default=4.0)
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--direct-qpos", action="store_true", help="Also write qpos for clearer visual debugging.")
    return parser.parse_args()


def actuator_id(model, name):
    actuator = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
    if actuator < 0:
        raise ValueError(f"Actuator not found: {name}")
    return actuator


def joint_id(model, name):
    joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    if joint < 0:
        raise ValueError(f"Joint not found: {name}")
    return joint


def side_joint_names(side):
    return [f"{side}_gripper_finger_joint1", f"{side}_gripper_finger_joint2"]


def controlled_joints(side):
    sides = ("left", "right") if side == "both" else (side,)
    names = []
    for current_side in sides:
        names.extend(side_joint_names(current_side))
    return names


def set_gripper(model, data, joint_names, value, direct_qpos):
    for joint_name in joint_names:
        actuator = actuator_id(model, f"{joint_name}_pos")
        data.ctrl[actuator] = value
        if direct_qpos:
            joint = joint_id(model, joint_name)
            data.qpos[model.jnt_qposadr[joint]] = value


def print_state(model, data, joint_names, value):
    states = []
    for joint_name in joint_names:
        joint = joint_id(model, joint_name)
        states.append(f"{joint_name}={data.qpos[model.jnt_qposadr[joint]]:.4f}")
    print(f"command={value:.4f} " + " ".join(states), flush=True)


def main():
    args = parse_args()
    model = mujoco.MjModel.from_xml_path(args.model)
    data = mujoco.MjData(model)
    joint_names = controlled_joints(args.side)

    mujoco.mj_forward(model, data)
    with mj_viewer.launch_passive(model, data) as v:
        start = time.time()
        last_print_bucket = -1
        while v.is_running():
            elapsed = (time.time() - start) * args.speed
            phase = 0.5 * (1.0 - math.cos(2.0 * math.pi * elapsed / max(args.period, 1e-6)))
            value = (1.0 - phase) * args.open + phase * args.close
            set_gripper(model, data, joint_names, value, args.direct_qpos)
            mujoco.mj_step(model, data)
            v.sync()

            bucket = int(elapsed * 2.0)
            if bucket != last_print_bucket:
                print_state(model, data, joint_names, value)
                last_print_bucket = bucket
            time.sleep(model.opt.timestep)


if __name__ == "__main__":
    main()
