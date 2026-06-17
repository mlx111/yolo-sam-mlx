import argparse
import time

import mujoco


def parse_args():
    parser = argparse.ArgumentParser(description="View the G2 drawer/cabinet scene.")
    parser.add_argument("--model", default="r1pro_drawer_scene.xml")
    parser.add_argument("--open", type=float, default=0.18)
    parser.add_argument("--base-x", type=float, default=-0.45)
    parser.add_argument("--base-y", type=float, default=0.0)
    parser.add_argument("--base-yaw", type=float, default=0.0)
    parser.add_argument("--settle-steps", type=int, default=300)
    return parser.parse_args()


def set_actuator(model, data, name, value):
    actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
    if actuator_id < 0:
        return
    low, high = model.actuator_ctrlrange[actuator_id]
    data.ctrl[actuator_id] = min(max(value, low), high)


def set_joint_qpos(model, data, joint_name, value):
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    if joint_id < 0:
        return
    qpos_id = model.jnt_qposadr[joint_id]
    dof_id = model.jnt_dofadr[joint_id]
    data.qpos[qpos_id] = value
    data.qvel[dof_id] = 0.0
    set_actuator(model, data, f"{joint_name}_pos", value)


def hold_position_actuators(model, data):
    for actuator_id in range(model.nu):
        if model.actuator_trntype[actuator_id] != mujoco.mjtTrn.mjTRN_JOINT:
            continue
        joint_id = int(model.actuator_trnid[actuator_id, 0])
        if joint_id < 0 or model.jnt_type[joint_id] not in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
            continue
        value = float(data.qpos[model.jnt_qposadr[joint_id]])
        if model.actuator_ctrllimited[actuator_id]:
            low, high = model.actuator_ctrlrange[actuator_id]
            value = min(max(value, low), high)
        data.ctrl[actuator_id] = value


def main():
    args = parse_args()
    model = mujoco.MjModel.from_xml_path(args.model)
    data = mujoco.MjData(model)

    set_joint_qpos(model, data, "base_x", args.base_x)
    set_joint_qpos(model, data, "base_y", args.base_y)
    set_joint_qpos(model, data, "base_yaw", args.base_yaw)
    set_joint_qpos(model, data, "drawer_slide_joint", args.open)
    hold_position_actuators(model, data)
    mujoco.mj_forward(model, data)

    from mujoco import viewer as mj_viewer

    with mj_viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            mujoco.mj_forward(model, data)
            viewer.sync()
            time.sleep(model.opt.timestep)


if __name__ == "__main__":
    main()
