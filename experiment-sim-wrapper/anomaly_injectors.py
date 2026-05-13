"""Anomaly injection functions for MuJoCo simulation experiments."""

import numpy as np
import mujoco


def inject_grasp_miss(model, data, apple_body_id, apple_initial_pos, apple_initial_quat):
    """模拟抓空异常：提起后将物体位置重置回桌面（抓取前的位置）。

    在 gripper-close + vertical-grasp 完成后调用此函数。
    效果：机械臂提起时手里是空的，物体还在桌上。
    """
    # 找到 apple 的 free joint qpos 地址
    apple_qpos_adr = None
    for j in range(model.njnt):
        if model.jnt_bodyid[j] == apple_body_id:
            apple_qpos_adr = model.jnt_qposadr[j]
            break

    if apple_qpos_adr is None:
        raise ValueError("apple body not found")

    # 将 apple 位置重置回抓取前的初始位置
    data.qpos[apple_qpos_adr:apple_qpos_adr+3] = apple_initial_pos
    data.qpos[apple_qpos_adr+3:apple_qpos_adr+7] = apple_initial_quat

    # 速度清零
    # apple free joint 在 qvel 中有 6 个 DOF
    # 需要找到对应的 qvel 地址
    for j in range(model.njnt):
        if model.jnt_bodyid[j] == apple_body_id:
            # free joint 的 DOF 地址
            if hasattr(model, 'jnt_dofadr'):
                dof_adr = model.jnt_dofadr[j]
                data.qvel[dof_adr:dof_adr+6] = 0.0
            break

    mujoco.mj_forward(model, data)

    return {
        "type": "grasp_miss",
        "object": "apple",
        "original_pos": apple_initial_pos.tolist(),
        "original_quat": apple_initial_quat.tolist(),
    }


def _get_object_qpos_adr(model, body_name):
    """Get the qpos address for a body with a free joint."""
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if body_id < 0:
        return None, None
    for j in range(model.njnt):
        if model.jnt_bodyid[j] == body_id:
            return model.jnt_qposadr[j], body_id
    return None, None


def inject_object_displaced(model, data, body_name, dx=0.05, dy=0.03, dz=0.0):
    """模拟物体被推走：给物体位置加一个偏移."""
    qpos_adr, body_id = _get_object_qpos_adr(model, body_name)
    if qpos_adr is None:
        raise ValueError(f"body {body_name} not found")

    data.qpos[qpos_adr:qpos_adr+3] += np.array([dx, dy, dz])
    mujoco.mj_forward(model, data)

    return {
        "type": "object_displaced",
        "object": body_name,
        "offset": [dx, dy, dz],
    }


def inject_slip(model, data, apple_body_id, apple_initial_pos, apple_initial_quat):
    """模拟提起中途滑落：苹果被夹起后从夹爪中滑脱，落回桌面。

    在提起轨迹执行到一半时调用此函数。
    效果：机械臂继续上升但手里已经空了，苹果回到桌面。
    """
    apple_qpos_adr = None
    for j in range(model.njnt):
        if model.jnt_bodyid[j] == apple_body_id:
            apple_qpos_adr = model.jnt_qposadr[j]
            break

    if apple_qpos_adr is None:
        raise ValueError("apple body not found")

    data.qpos[apple_qpos_adr:apple_qpos_adr + 3] = apple_initial_pos
    data.qpos[apple_qpos_adr + 3:apple_qpos_adr + 7] = apple_initial_quat

    for j in range(model.njnt):
        if model.jnt_bodyid[j] == apple_body_id:
            if hasattr(model, 'jnt_dofadr'):
                dof_adr = model.jnt_dofadr[j]
                data.qvel[dof_adr:dof_adr + 6] = 0.0
            break

    mujoco.mj_forward(model, data)

    return {
        "type": "slip",
        "object": "apple",
        "reset_pos": apple_initial_pos.tolist(),
        "reset_quat": apple_initial_quat.tolist(),
    }


def inject_collision(model, data, body_name, vx=1.2, vy=0.9, vz=2.5):
    """模拟碰撞推飞：末端碰到物体，给物体一个瞬时速度，物体被弹飞。

    在接近轨迹中途调用，物体靠物理模拟自己滚到新位置。
    """
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if body_id < 0:
        raise ValueError(f"body {body_name} not found")
    for j in range(model.njnt):
        if model.jnt_bodyid[j] == body_id:
            if hasattr(model, 'jnt_dofadr'):
                dof_adr = model.jnt_dofadr[j]
                data.qvel[dof_adr:dof_adr + 3] = np.array([vx, vy, vz], dtype=np.float64)
            break
    mujoco.mj_forward(model, data)

    return {
        "type": "collision",
        "object": body_name,
        "velocity": [vx, vy, vz],
    }


def inject_gripper_fail(model, data):
    """模拟夹爪故障：强制夹爪不闭合。

    在 gripper-action(close) 之后调用，维持 finger joint 在张开位置。
    """
    for joint_name in ['right_driver_joint', 'left_driver_joint']:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if jid >= 0:
            adr = model.jnt_qposadr[jid]
            data.qpos[adr] = 0.0  # 张开位置

    mujoco.mj_forward(model, data)

    return {
        "type": "gripper_fail",
        "detail": "fingers forced open",
    }


def setup_incipient_slip(model, data, apple_body_id):
    """初始化渐发式滑移状态。

    在提起轨迹开始前调用，记录苹果当前 Z 高度和关节地址。

    返回一个 state dict，供 apply_incipient_slip_step() 每步使用。
    """
    apple_qpos_adr = None
    apple_dof_adr = None
    for j in range(model.njnt):
        if model.jnt_bodyid[j] == apple_body_id:
            apple_qpos_adr = model.jnt_qposadr[j]
            if hasattr(model, 'jnt_dofadr'):
                apple_dof_adr = model.jnt_dofadr[j]
            break

    if apple_qpos_adr is None:
        raise ValueError("apple body not found")

    initial_z = float(data.qpos[apple_qpos_adr + 2])

    return {
        "apple_qpos_adr": apple_qpos_adr,
        "apple_dof_adr": apple_dof_adr,
        "initial_z": initial_z,
        "table_z": initial_z,  # 桌面参考高度
    }


def apply_incipient_slip_step(model, data, apple_body_id, step, total_steps, slip_state):
    """在提起轨迹的每一步施加渐发式滑移。

    滑移曲线（三段式）：
      - Phase 1 (前 20%):  无滑移，MuJoCo 接触力自然保持苹果在夹爪中
      - Phase 2 (20%-70%): 每步施加递增的向下位移 (二次方曲线)
      - Phase 3 (70%+):    苹果完全脱出，放回桌面高度并清零速度

    位移是相对于苹果**当前** z 坐标的 (而非初始高度)，且以桌面高度为下界。

    Parameters
    ----------
    step : int — 当前步数 (0-based)
    total_steps : int — 提起轨迹总步数
    slip_state : dict — setup_incipient_slip() 的返回值

    Returns
    -------
    float — 本步施加的向下位移量 (m)
    """
    frac = step / max(total_steps, 1)
    table_z = slip_state["table_z"]
    adr = slip_state["apple_qpos_adr"]

    if frac < 0.2:
        # Phase 1: 不干预，让 MuJoCo 接触力自然提起苹果
        return 0.0

    elif frac < 0.7:
        # Phase 2: 加速度向下漂移
        phase_frac = (frac - 0.2) / 0.5
        # 累计漂移量 (二次方)，最大 ~0.20m，足够从提起高度落回桌面
        total_drift = 0.20 * phase_frac ** 2
        # 上一帧的累计漂移
        prev_frac = max(0, (step - 1) / max(total_steps, 1))
        if prev_frac < 0.2:
            prev_drift = 0.0
        elif prev_frac < 0.7:
            prev_drift = 0.20 * ((prev_frac - 0.2) / 0.5) ** 2
        else:
            prev_drift = 0.20
        step_drift = total_drift - prev_drift

        current_z = float(data.qpos[adr + 2])
        new_z = max(current_z - step_drift, table_z)
        data.qpos[adr + 2] = new_z
        return step_drift

    else:
        # Phase 3: 落回桌面
        data.qpos[adr + 2] = table_z
        if slip_state.get("apple_dof_adr") is not None:
            dof_adr = slip_state["apple_dof_adr"]
            data.qvel[dof_adr:dof_adr + 6] = 0.0
        return 0.0
