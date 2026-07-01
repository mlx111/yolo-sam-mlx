"""Standard failure taxonomy for universal robot experience memory."""

from __future__ import annotations

from typing import Any

from .schema import ExperienceEntry


STANDARD_FAILURE_TYPES = (
    "grasp_miss",
    "grasp_slip",
    "object_not_lifted",
    "place_occupied",
    "place_error",
    "transport_collision",
    "dual_arm_mismatch",
    "perception_miss",
    "actuation_limit",
    "sim_success_real_fail",
    "unknown_failure",
)

NON_ACTIONABLE_FAILURE_TYPES = {"", "unknown", "unknown_failure", "none", "success"}


RULE_TO_STANDARD = {
    "object_not_lifted": "object_not_lifted",
    "place_xy_error_high": "place_error",
    "place_z_error_high": "place_error",
    "place_zone_miss": "place_error",
    "gripper_contact_missing": "grasp_miss",
    "no_contact_detected": "grasp_miss",
    "contact_lost_during_lift": "grasp_slip",
    "dual_arm_height_mismatch": "dual_arm_mismatch",
    "collision_risk": "transport_collision",
    "joint_limit_risk": "actuation_limit",
    "sim_success_real_fail": "sim_success_real_fail",
    "gripper_failure_reason": "grasp_miss",
    "task_failure_reason": "unknown_failure",
}


KEYWORD_TO_STANDARD = (
    (("sim_success_real_fail",), "sim_success_real_fail"),
    (("place_occupied", "occupied", "占用", "放置区"), "place_occupied"),
    (("slip", "drop", "lost contact", "contact_lost", "滑", "掉", "丢失接触"), "grasp_slip"),
    (("not_lifted", "object_not_lifted", "未抬升", "抬升", "lift"), "object_not_lifted"),
    (("grasp", "gripper", "pinch", "夹持", "抓取", "夹爪"), "grasp_miss"),
    (("place", "zone", "放置"), "place_error"),
    (("collision", "碰撞"), "transport_collision"),
    (("dual_arm", "height_mismatch", "双臂", "高度"), "dual_arm_mismatch"),
    (("perception", "detect", "tracked", "识别", "感知", "跟踪"), "perception_miss"),
    (("joint", "limit", "actuator", "力矩", "关节"), "actuation_limit"),
)


def normalize_failure_type(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    if text in STANDARD_FAILURE_TYPES:
        return text
    lowered = text.lower()
    if lowered in STANDARD_FAILURE_TYPES:
        return lowered
    if lowered in RULE_TO_STANDARD:
        return RULE_TO_STANDARD[lowered]
    for keywords, standard in KEYWORD_TO_STANDARD:
        if any(keyword in lowered or keyword in text for keyword in keywords):
            return standard
    return "unknown_failure"


def is_actionable_failure_type(value: Any) -> bool:
    return normalize_failure_type(value) not in NON_ACTIONABLE_FAILURE_TYPES


def infer_standard_failure_type(entry: ExperienceEntry) -> str:
    existing = normalize_failure_type(entry.failure_taxonomy.get("standard_failure_type"))
    if existing:
        return existing

    outcome_type = str(entry.sim_real_gap.outcome_gap.get("type") or "")
    if outcome_type == "sim_success_real_fail":
        return "sim_success_real_fail"

    rule_types = []
    for flag in entry.critic_result.rule_flags:
        if not isinstance(flag, dict):
            continue
        standard = normalize_failure_type(flag.get("rule"))
        if standard:
            rule_types.append(standard)
    if rule_types:
        return _pick_priority(rule_types)

    raw_failure = entry.failure_taxonomy.get("failure_type") or entry.result.get("failure_reason") or ""
    standard = normalize_failure_type(raw_failure)
    if standard:
        return standard
    if entry.result.get("success") is False:
        return "unknown_failure"
    return ""


def standardize_failure_taxonomy(entry: ExperienceEntry) -> ExperienceEntry:
    taxonomy = dict(entry.failure_taxonomy or {})
    raw_failure_type = taxonomy.get("failure_type") or entry.result.get("failure_reason") or ""
    success = bool(entry.result.get("success", entry.result.get("task_success", False)))
    outcome_type = str(entry.sim_real_gap.outcome_gap.get("type") or "")
    if success and outcome_type != "sim_success_real_fail" and not is_actionable_failure_type(raw_failure_type):
        entry.failure_taxonomy = {}
        return entry
    if raw_failure_type and not taxonomy.get("raw_failure_type"):
        taxonomy["raw_failure_type"] = raw_failure_type
    standard = infer_standard_failure_type(entry)
    if standard and (not success or standard == "sim_success_real_fail" or is_actionable_failure_type(raw_failure_type)):
        taxonomy["standard_failure_type"] = standard
        taxonomy["failure_type"] = standard
    elif success:
        taxonomy = {}
    entry.failure_taxonomy = taxonomy
    return entry


def _pick_priority(types: list[str]) -> str:
    priority = {
        "sim_success_real_fail": 100,
        "transport_collision": 90,
        "actuation_limit": 80,
        "grasp_slip": 70,
        "object_not_lifted": 65,
        "grasp_miss": 60,
        "place_occupied": 55,
        "place_error": 50,
        "dual_arm_mismatch": 45,
        "perception_miss": 40,
        "unknown_failure": 0,
    }
    return max(types, key=lambda item: priority.get(item, 0))
