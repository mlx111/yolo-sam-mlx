import argparse

import mujoco
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Smoke test the G2 drawer/cabinet MuJoCo scene.")
    parser.add_argument("--model", default="r1pro_drawer_scene.xml")
    parser.add_argument("--open", type=float, default=0.25)
    parser.add_argument("--steps", type=int, default=800)
    return parser.parse_args()


def obj_id(model, obj_type, name):
    object_id = mujoco.mj_name2id(model, obj_type, name)
    if object_id < 0:
        raise ValueError(f"MuJoCo object not found: {name}")
    return object_id


def main():
    args = parse_args()
    model = mujoco.MjModel.from_xml_path(args.model)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    required_bodies = ("cabinet_body", "drawer_body", "inside_target_object")
    required_joints = ("drawer_slide_joint", "base_x", "base_y", "base_yaw")
    required_actuators = ("drawer_slide_joint_pos", "base_x_pos", "base_y_pos", "base_yaw_pos")
    required_sites = ("handle_site", "drawer_inside_site", "drawer_open_site", "inside_target_site")
    required_geoms = ("drawer_handle_collision_geom", "drawer_front_edge_geom", "inside_target_collision_geom")

    print("nq/nv/nu:", model.nq, model.nv, model.nu)
    for name in required_bodies:
        print("body", name, obj_id(model, mujoco.mjtObj.mjOBJ_BODY, name))
    for name in required_joints:
        print("joint", name, obj_id(model, mujoco.mjtObj.mjOBJ_JOINT, name))
    for name in required_actuators:
        print("actuator", name, obj_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name))
    for name in required_sites:
        print("site", name, obj_id(model, mujoco.mjtObj.mjOBJ_SITE, name))
    for name in required_geoms:
        print("geom", name, obj_id(model, mujoco.mjtObj.mjOBJ_GEOM, name))

    drawer_body_id = obj_id(model, mujoco.mjtObj.mjOBJ_BODY, "drawer_body")
    drawer_joint_id = obj_id(model, mujoco.mjtObj.mjOBJ_JOINT, "drawer_slide_joint")
    drawer_actuator_id = obj_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "drawer_slide_joint_pos")
    start = data.xpos[drawer_body_id].copy()
    low, high = model.actuator_ctrlrange[drawer_actuator_id]
    data.ctrl[drawer_actuator_id] = np.clip(args.open, low, high)
    for _ in range(max(args.steps, 0)):
        mujoco.mj_step(model, data)
    mujoco.mj_forward(model, data)

    end = data.xpos[drawer_body_id].copy()
    qpos = float(data.qpos[model.jnt_qposadr[drawer_joint_id]])
    print("drawer_slide_qpos:", round(qpos, 6))
    print("drawer_delta:", np.round(end - start, 6).tolist())
    for name in required_sites:
        site_id = obj_id(model, mujoco.mjtObj.mjOBJ_SITE, name)
        print(name, np.round(data.site_xpos[site_id], 6).tolist())


if __name__ == "__main__":
    main()
