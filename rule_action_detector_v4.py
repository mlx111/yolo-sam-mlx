from __future__ import annotations

from typing import Any, Dict, Optional


THRESHOLDS = {
    "move_pregrasp_success_pos": 0.02,
    "move_pregrasp_uncertain_pos": 0.04,
    "move_pregrasp_success_rot": 10.0,
    "move_pregrasp_uncertain_rot": 20.0,
    "move_pregrasp_obj_soft": 0.005,
    "move_pregrasp_obj_hard": 0.015,
    "move_grasp_success_pos": 0.012,
    "move_grasp_uncertain_pos": 0.025,
    "move_grasp_success_rot": 8.0,
    "move_grasp_uncertain_rot": 15.0,
    "move_grasp_obj_soft": 0.015,
    "move_grasp_obj_hard": 0.02,
    "grasp_rel_success": 0.01,
    "grasp_rel_uncertain": 0.02,
    "lift_success": 0.08,
    "lift_hard_failure": 0.03,
    "lift_end_success": 0.05,
    "lift_rel_success": 0.015,
    "lift_rel_uncertain": 0.03,
    "place_xy_success": 0.03,
    "place_xy_uncertain": 0.06,
    "home_joint_success": 0.05,
    "open_signal_max": 30.0,
    "close_signal_min": 150.0,
}


def _result(
    rule_status: str,
    reason: str,
    *,
    message: Optional[str] = None,
    metrics: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    return {
        "rule_status": rule_status,
        "status": "success" if rule_status == "SUCCESS" else "failure",
        "message": message or reason,
        "reason": reason,
        "metrics": metrics or {},
    }


def _get(metrics: Dict[str, Any], key: str) -> Optional[float]:
    value = metrics.get(key)
    if value is None:
        return None
    return float(value)


def detect_action(action_name: str, snapshot: Dict[str, Any]) -> Dict[str, Any]:
    metrics = dict(snapshot.get("metrics", {}))
    if action_name == "移动到预抓取位置":
        return _detect_move_pregrasp(metrics)
    if action_name == "移动到抓取位置":
        return _detect_move_grasp(metrics)
    if action_name == "夹爪闭合":
        return _detect_gripper_close(metrics)
    if action_name == "夹爪开启":
        return _detect_gripper_open(metrics)
    if action_name == "提升物体":
        return _detect_vertical_grasp(metrics)
    if action_name == "移动到预放置位置":
        return _detect_place_move(metrics)
    if action_name == "回到初始位置":
        return _detect_execute_init(metrics)
    return _result("UNCERTAIN", f"未定义动作[{action_name}]的规则检测器", metrics=metrics)


def _detect_move_pregrasp(metrics: Dict[str, Any]) -> Dict[str, Any]:
    pos_error = _get(metrics, "expected_pos_error")
    rot_error = _get(metrics, "expected_rot_error_deg")
    obj_motion = _get(metrics, "target_motion")
    if pos_error is None:
        return _result("UNCERTAIN", "缺少预抓取位姿误差", metrics=metrics)
    if (
        pos_error > THRESHOLDS["move_pregrasp_uncertain_pos"]
        or (rot_error is not None and rot_error > THRESHOLDS["move_pregrasp_uncertain_rot"])
        or (obj_motion is not None and obj_motion > THRESHOLDS["move_pregrasp_obj_hard"])
    ):
        return _result("FAILURE_HARD", "预抓取位姿偏差过大或目标被明显撞动", metrics=metrics)
    if (
        pos_error <= THRESHOLDS["move_pregrasp_success_pos"]
        and (rot_error is None or rot_error <= THRESHOLDS["move_pregrasp_success_rot"])
        and (obj_motion is None or obj_motion <= THRESHOLDS["move_pregrasp_obj_soft"])
    ):
        return _result("SUCCESS", "预抓取位姿满足要求", metrics=metrics)
    return _result("UNCERTAIN", "预抓取接近阈值边界，需要复核", metrics=metrics)


def _detect_move_grasp(metrics: Dict[str, Any]) -> Dict[str, Any]:
    pos_error = _get(metrics, "expected_pos_error")
    rot_error = _get(metrics, "expected_rot_error_deg")
    obj_motion = _get(metrics, "target_motion")
    if pos_error is None:
        return _result("UNCERTAIN", "缺少抓取位姿误差", metrics=metrics)
    if (
        pos_error > THRESHOLDS["move_grasp_uncertain_pos"]
        or (rot_error is not None and rot_error > THRESHOLDS["move_grasp_uncertain_rot"])
        or (obj_motion is not None and obj_motion > THRESHOLDS["move_grasp_obj_hard"])
    ):
        return _result("FAILURE_HARD", "抓取位姿偏差过大或目标被明显撞偏", metrics=metrics)
    if (
        pos_error <= THRESHOLDS["move_grasp_success_pos"]
        and (rot_error is None or rot_error <= THRESHOLDS["move_grasp_success_rot"])
        and (obj_motion is None or obj_motion <= THRESHOLDS["move_grasp_obj_soft"])
    ):
        return _result("SUCCESS", "抓取位姿已准确到位", metrics=metrics)
    # 对 move-grasp，位姿到位是主判据；轻微接触或小幅物体扰动不直接判失败。
    if (
        pos_error <= 0.003
        and (rot_error is None or rot_error <= 3.0)
        and (obj_motion is None or obj_motion <= THRESHOLDS["move_grasp_obj_hard"])
    ):
        return _result("SUCCESS", "抓取位姿已精确到位，允许轻微接触", metrics=metrics)
    return _result("UNCERTAIN", "抓取位姿接近阈值边界，需要复核", metrics=metrics)


def _detect_gripper_close(metrics: Dict[str, Any]) -> Dict[str, Any]:
    left_contact = bool(metrics.get("left_contact"))
    right_contact = bool(metrics.get("right_contact"))
    rel_error = _get(metrics, "post_relative_error")
    gripper_signal = _get(metrics, "gripper_signal")
    has_any_contact = left_contact or right_contact
    has_dual_contact = left_contact and right_contact

    if gripper_signal is not None and gripper_signal < THRESHOLDS["close_signal_min"] and not has_any_contact:
        return _result("FAILURE_HARD", "夹爪未充分闭合且未接触目标", metrics=metrics)

    # 闭合阶段只做宽松预筛查，真正是否抓稳交给 vertical-grasp 再判断。
    if has_dual_contact:
        return _result("SUCCESS", "夹爪已闭合并形成双侧接触", metrics=metrics)

    if has_any_contact:
        return _result("SUCCESS", "夹爪已闭合并接触到目标", metrics=metrics)

    if gripper_signal is not None and gripper_signal >= THRESHOLDS["close_signal_min"]:
        return _result("UNCERTAIN", "夹爪已闭合，但当前未检测到稳定接触", metrics=metrics)

    if rel_error is not None and rel_error <= THRESHOLDS["grasp_rel_uncertain"]:
        return _result("UNCERTAIN", "夹爪闭合证据不足，需要复核", metrics=metrics)

    return _result("FAILURE_HARD", "夹爪闭合后未形成有效抓取证据", metrics=metrics)


def _detect_gripper_open(metrics: Dict[str, Any]) -> Dict[str, Any]:
    left_contact = bool(metrics.get("left_contact"))
    right_contact = bool(metrics.get("right_contact"))
    gripper_signal = _get(metrics, "gripper_signal")
    xy_error = _get(metrics, "place_xy_error")
    if gripper_signal is not None and gripper_signal > THRESHOLDS["open_signal_max"]:
        return _result("FAILURE_HARD", "夹爪开启信号仍然偏大", metrics=metrics)
    if left_contact or right_contact:
        return _result("UNCERTAIN", "夹爪开启后仍存在接触，需要复核", metrics=metrics)
    if xy_error is not None and xy_error > THRESHOLDS["place_xy_uncertain"]:
        return _result("FAILURE_SOFT", "目标释放后位置偏差较大", metrics=metrics)
    return _result("SUCCESS", "夹爪已张开且目标已释放", metrics=metrics)


def _detect_vertical_grasp(metrics: Dict[str, Any]) -> Dict[str, Any]:
    max_lift = _get(metrics, "target_max_lift_dz")
    end_lift = _get(metrics, "target_lift_dz")
    rel_delta = _get(metrics, "relative_translation_delta")
    gripper_signal = _get(metrics, "gripper_signal")

    if max_lift is None:
        return _result("UNCERTAIN", "缺少目标物体的 Z 轨迹", metrics=metrics)
    if max_lift < THRESHOLDS["lift_hard_failure"]:
        if gripper_signal is not None and gripper_signal < THRESHOLDS["close_signal_min"]:
            return _result("FAILURE_HARD", "夹爪未充分闭合且目标几乎没有被提起", metrics=metrics)
        return _result("FAILURE_HARD", "目标物体的 Z 坐标几乎没有上升", metrics=metrics)

    if max_lift >= THRESHOLDS["lift_success"]:
        if end_lift is not None and end_lift >= THRESHOLDS["lift_end_success"]:
            return _result("SUCCESS", "目标物体 Z 坐标明显上升，提升成功", metrics=metrics)
        return _result("UNCERTAIN", "目标一度被提起，但动作结束时有明显回落", metrics=metrics)

    if rel_delta is not None and rel_delta > THRESHOLDS["lift_rel_uncertain"]:
        return _result("FAILURE_SOFT", "目标有上升，但提升过程中稳定性不足", metrics=metrics)
    return _result("FAILURE_SOFT", "目标物体有一定上升，但提升幅度不足", metrics=metrics)


def _detect_place_move(metrics: Dict[str, Any]) -> Dict[str, Any]:
    place_xy_error = _get(metrics, "place_xy_error")
    rel_error = _get(metrics, "post_relative_error")
    if place_xy_error is None:
        return _result("UNCERTAIN", "缺少放置位置误差", metrics=metrics)
    if place_xy_error > THRESHOLDS["place_xy_uncertain"]:
        return _result("FAILURE_HARD", "目标未被搬运到指定放置区域", metrics=metrics)
    if place_xy_error <= THRESHOLDS["place_xy_success"] and (
        rel_error is None or rel_error <= THRESHOLDS["grasp_rel_uncertain"]
    ):
        return _result("SUCCESS", "目标已被稳定搬运到放置区域", metrics=metrics)
    return _result("FAILURE_SOFT", "目标接近放置区域但误差偏大，需要复核", metrics=metrics)


def _detect_execute_init(metrics: Dict[str, Any]) -> Dict[str, Any]:
    home_error = _get(metrics, "home_joint_error")
    left_contact = bool(metrics.get("left_contact"))
    right_contact = bool(metrics.get("right_contact"))
    gripper_signal = _get(metrics, "gripper_signal")
    if home_error is None:
        return _result("UNCERTAIN", "缺少回初始位的关节误差", metrics=metrics)
    if home_error > THRESHOLDS["home_joint_success"]:
        return _result("FAILURE_HARD", "机械臂未回到初始关节位", metrics=metrics)
    if gripper_signal is not None and gripper_signal > THRESHOLDS["open_signal_max"]:
        return _result("FAILURE_HARD", "回初始位时夹爪未处于打开状态", metrics=metrics)
    if left_contact or right_contact:
        return _result("UNCERTAIN", "机械臂已回初始位，但夹爪附近仍有目标接触", metrics=metrics)
    return _result("SUCCESS", "机械臂已回到初始位且目标已释放", metrics=metrics)
