import argparse
import time

import mujoco
import numpy as np
import pinocchio as pin


ARM_JOINTS = {
    "left": [f"left_arm_joint{i}" for i in range(1, 8)],
    "right": [f"right_arm_joint{i}" for i in range(1, 8)],
}


def mj_id(model, obj_type, name):
    idx = mujoco.mj_name2id(model, obj_type, name)
    if idx < 0:
        raise ValueError(f"MuJoCo object not found: {name}")
    return idx


def pin_joint_q_index(model, name):
    if not model.existJointName(name):
        raise ValueError(f"Pinocchio joint not found: {name}")
    jid = model.getJointId(name)
    if model.nqs[jid] != 1 or model.nvs[jid] != 1:
        raise ValueError(f"Only 1-DoF joints are supported here: {name}")
    return model.idx_qs[jid], model.idx_vs[jid]


def sync_pin_q_from_mujoco(pin_model, mj_model, mj_data):
    q = pin.neutral(pin_model)
    for name in pin_model.names[1:]:
        if not pin_model.existJointName(name):
            continue
        jid_pin = pin_model.getJointId(name)
        if pin_model.nqs[jid_pin] != 1:
            continue
        jid_mj = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid_mj < 0:
            continue
        q[pin_model.idx_qs[jid_pin]] = mj_data.qpos[mj_model.jnt_qposadr[jid_mj]]
    return q


def solve_position_ik(
    pin_model,
    frame_name,
    controlled_joints,
    q0,
    target_pos,
    iterations=200,
    damping=1e-4,
    step_scale=0.5,
    tolerance=1e-4,
):
    data = pin_model.createData()
    q = q0.copy()
    frame_id = pin_model.getFrameId(frame_name)
    controlled_qidx = []
    controlled_vidx = []
    lower = []
    upper = []

    for name in controlled_joints:
        qidx, vidx = pin_joint_q_index(pin_model, name)
        jid = pin_model.getJointId(name)
        controlled_qidx.append(qidx)
        controlled_vidx.append(vidx)
        lower.append(pin_model.lowerPositionLimit[qidx])
        upper.append(pin_model.upperPositionLimit[qidx])

    lower = np.asarray(lower)
    upper = np.asarray(upper)
    controlled_vidx = np.asarray(controlled_vidx)

    last_error = None
    for _ in range(iterations):
        pin.forwardKinematics(pin_model, data, q)
        pin.updateFramePlacements(pin_model, data)
        current_pos = data.oMf[frame_id].translation.copy()
        error = target_pos - current_pos
        last_error = float(np.linalg.norm(error))
        if last_error < tolerance:
            break

        jac = pin.computeFrameJacobian(
            pin_model,
            data,
            q,
            frame_id,
            pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
        )[:3, controlled_vidx]
        lhs = jac @ jac.T + damping * np.eye(3)
        dq = jac.T @ np.linalg.solve(lhs, error)
        q_arm = q[controlled_qidx] + step_scale * dq
        q[controlled_qidx] = np.clip(q_arm, lower, upper)

    pin.forwardKinematics(pin_model, data, q)
    pin.updateFramePlacements(pin_model, data)
    final_pos = data.oMf[frame_id].translation.copy()
    return q, final_pos, last_error


def apply_arm_targets(mj_model, mj_data, side, q_pin, pin_model):
    for name in ARM_JOINTS[side]:
        qidx, _ = pin_joint_q_index(pin_model, name)
        aid = mj_id(mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{name}_pos")
        lo, hi = mj_model.actuator_ctrlrange[aid]
        mj_data.ctrl[aid] = np.clip(q_pin[qidx], lo, hi)


def set_arm_qpos_direct(mj_model, mj_data, side, q_pin, pin_model):
    for name in ARM_JOINTS[side]:
        qidx, _ = pin_joint_q_index(pin_model, name)
        jid = mj_id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, name)
        mj_data.qpos[mj_model.jnt_qposadr[jid]] = q_pin[qidx]


def make_arm_qpos_trajectory(pin_model, side, q_start, q_target, steps):
    q_steps = []
    arm_qidx = [pin_joint_q_index(pin_model, name)[0] for name in ARM_JOINTS[side]]
    for i in range(max(steps, 1)):
        alpha = 1.0 if steps <= 1 else i / (steps - 1)
        # Smoothstep avoids a hard start/stop in the viewer.
        alpha = alpha * alpha * (3.0 - 2.0 * alpha)
        q = q_start.copy()
        q[arm_qidx] = (1.0 - alpha) * q_start[arm_qidx] + alpha * q_target[arm_qidx]
        q_steps.append(q)
    return q_steps


def main():
    parser = argparse.ArgumentParser(description="Test Pinocchio IK targets in MuJoCo.")
    parser.add_argument("--model", default="model.xml")
    parser.add_argument("--urdf", default="urdf/r1_pro_with_gripper.urdf")
    parser.add_argument("--side", choices=["left", "right"], default="left")
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--direct-qpos", action="store_true", help="Teleport joints for static inspection.")
    parser.add_argument("--dx", type=float, default=0.04)
    parser.add_argument("--dy", type=float, default=0.0)
    parser.add_argument("--dz", type=float, default=0.04)
    parser.add_argument("--target-x", type=float, default=None, help="Absolute world target x for the hand tcp.")
    parser.add_argument("--target-y", type=float, default=None, help="Absolute world target y for the hand tcp.")
    parser.add_argument("--target-z", type=float, default=None, help="Absolute world target z for the hand tcp.")
    parser.add_argument("--steps", type=int, default=240)
    parser.add_argument("--hold-seconds", type=float, default=2.0)
    args = parser.parse_args()

    mj_model = mujoco.MjModel.from_xml_path(args.model)
    mj_data = mujoco.MjData(mj_model)
    pin_model = pin.buildModelFromUrdf(args.urdf)

    q0 = sync_pin_q_from_mujoco(pin_model, mj_model, mj_data)
    frame_name = f"{args.side}_hand_tcp"
    frame_id = pin_model.getFrameId(frame_name)
    pin_data = pin_model.createData()
    pin.forwardKinematics(pin_model, pin_data, q0)
    pin.updateFramePlacements(pin_model, pin_data)
    start_pos = pin_data.oMf[frame_id].translation.copy()
    absolute_values = [args.target_x, args.target_y, args.target_z]
    if any(value is not None for value in absolute_values):
        if not all(value is not None for value in absolute_values):
            raise ValueError("Use --target-x, --target-y, and --target-z together for an absolute target.")
        target_pos = np.array(absolute_values, dtype=float)
    else:
        target_pos = start_pos + np.array([args.dx, args.dy, args.dz])

    q_target, final_pos, ik_error = solve_position_ik(
        pin_model,
        frame_name,
        ARM_JOINTS[args.side],
        q0,
        target_pos,
    )

    print("side:", args.side)
    print("frame:", frame_name)
    print("start_pos:", np.round(start_pos, 6))
    print("target_pos:", np.round(target_pos, 6))
    print("ik_final_pos:", np.round(final_pos, 6))
    print("ik_error:", ik_error)
    for name in ARM_JOINTS[args.side]:
        qidx, _ = pin_joint_q_index(pin_model, name)
        print(f"{name}: {q0[qidx]: .6f} -> {q_target[qidx]: .6f}")

    trajectory = make_arm_qpos_trajectory(pin_model, args.side, q0, q_target, args.steps)

    if args.direct_qpos:
        set_arm_qpos_direct(mj_model, mj_data, args.side, q_target, pin_model)
        mujoco.mj_forward(mj_model, mj_data)
    else:
        for _ in range(args.steps):
            apply_arm_targets(mj_model, mj_data, args.side, q_target, pin_model)
            mujoco.mj_step(mj_model, mj_data)

    site_id = mj_id(mj_model, mujoco.mjtObj.mjOBJ_SITE, frame_name)
    print("mujoco_site_pos:", np.round(mj_data.site_xpos[site_id], 6))

    if args.viewer:
        from mujoco import viewer as mj_viewer

        with mj_viewer.launch_passive(mj_model, mj_data) as viewer:
            step_index = 0
            hold_until = None
            while viewer.is_running():
                if args.direct_qpos:
                    if step_index < len(trajectory):
                        set_arm_qpos_direct(mj_model, mj_data, args.side, trajectory[step_index], pin_model)
                        step_index += 1
                        if step_index == len(trajectory):
                            hold_until = time.time() + args.hold_seconds
                    elif hold_until is not None and time.time() > hold_until:
                        step_index = 0
                        hold_until = None
                    mujoco.mj_forward(mj_model, mj_data)
                else:
                    apply_arm_targets(mj_model, mj_data, args.side, q_target, pin_model)
                    mujoco.mj_step(mj_model, mj_data)
                viewer.sync()
                time.sleep(mj_model.opt.timestep)


if __name__ == "__main__":
    main()
