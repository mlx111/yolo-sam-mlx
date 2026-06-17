import argparse

import mujoco
import numpy as np

from skills.base.wrist_force_skill import load_skill as load_wrist_force
from skills.primitives.left_gripper_close_skill import load_skill as load_left_close


def parse_args():
    parser = argparse.ArgumentParser(description="Smoke test wrist external force estimation from MuJoCo contacts.")
    parser.add_argument("--model", default="r1pro_grasp_scene.xml")
    parser.add_argument("--side", choices=["left", "right"], default="left")
    return parser.parse_args()


def set_freejoint_body_pose(model, data, body_name, pos):
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if body_id < 0:
        raise ValueError(f"MuJoCo body not found: {body_name}")
    for joint_id in range(model.njnt):
        if int(model.jnt_bodyid[joint_id]) == body_id and model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_FREE:
            qpos_adr = int(model.jnt_qposadr[joint_id])
            qvel_adr = int(model.jnt_dofadr[joint_id])
            data.qpos[qpos_adr : qpos_adr + 3] = np.asarray(pos, dtype=np.float64)
            data.qpos[qpos_adr + 3 : qpos_adr + 7] = np.array([1.0, 0.0, 0.0, 0.0])
            data.qvel[qvel_adr : qvel_adr + 6] = 0.0
            return
    raise ValueError(f"Freejoint not found for body: {body_name}")


def main():
    args = parse_args()
    model = mujoco.MjModel.from_xml_path(args.model)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    skill = load_wrist_force()
    initial = skill.read(model, data, args.side)
    print(
        "wrist_force_initial:",
        {
            "side": initial.side,
            "force_norm": round(initial.force_norm, 6),
            "torque_norm": round(initial.torque_norm, 6),
            "contact_count": initial.contact_count,
        },
    )

    if args.side == "left":
        pad_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "left_gripper_finger_link1_pad1")
        if pad_id < 0:
            raise ValueError("left_gripper_finger_link1_pad1 geom not found")
        set_freejoint_body_pose(model, data, "target_cube", data.geom_xpos[pad_id].copy())
        mujoco.mj_forward(model, data)
        loaded = skill.read(model, data, "left")
        print(
            "wrist_force_after_contact:",
            {
                "side": loaded.side,
                "force_world": np.round(loaded.force_world, 6).tolist(),
                "torque_world": np.round(loaded.torque_world, 6).tolist(),
                "force_norm": round(loaded.force_norm, 6),
                "torque_norm": round(loaded.torque_norm, 6),
                "contact_count": loaded.contact_count,
                "contacts": [(item.geom1, item.geom2) for item in loaded.contacts[:5]],
            },
        )


if __name__ == "__main__":
    main()
