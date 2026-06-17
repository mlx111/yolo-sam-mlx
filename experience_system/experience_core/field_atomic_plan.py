"""LLM planning helpers for field-style atomic robot actions."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from .llm_provider import JSON_ONLY_LINE, invoke_llm, parse_json_payload


FIELD_ATOMIC_PARAMETER_GUIDANCE: dict[str, dict[str, Any]] = {
    "left_arm_move_to_position": {
        "target_x": [-0.2, 0.8],
        "target_y": [-0.4, 0.5],
        "target_z": [0.55, 1.25],
        "control_frame": ["tcp", "pinch"],
        "target_quat_wxyz": [-1.0, 1.0, 4],
        "orientation_weight": [0.0, 1.0],
        "orientation_threshold": [0.01, 1.5],
        "steps": [100, 2000],
        "settle_steps": [0, 1000],
        "max_joint_step": [0.001, 0.02],
        "fail_threshold": [0.005, 0.08],
        "direct_qpos": [False, True],
    },
    "right_arm_move_to_position": {
        "target_x": [-0.2, 0.8],
        "target_y": [-0.5, 0.4],
        "target_z": [0.55, 1.25],
        "control_frame": ["tcp", "pinch"],
        "target_quat_wxyz": [-1.0, 1.0, 4],
        "orientation_weight": [0.0, 1.0],
        "orientation_threshold": [0.01, 1.5],
        "steps": [100, 2000],
        "settle_steps": [0, 1000],
        "max_joint_step": [0.001, 0.02],
        "fail_threshold": [0.005, 0.08],
        "direct_qpos": [False, True],
    },
    "left_gripper_set": {
        "state": [0, 1],
        "gripper_value": [0.0, 0.025],
        "direct_qpos": [False, True],
    },
    "right_gripper_set": {
        "state": [0, 1],
        "gripper_value": [0.0, 0.025],
        "direct_qpos": [False, True],
    },
    "torso_move_to_posture": {
        "target_qpos": [[-0.1, -0.1, -0.2, -0.35], [0.1, 0.15, 0.2, 0.35]],
        "steps": [100, 1200],
        "settle_steps": [0, 600],
        "max_joint_step": [0.001, 0.02],
        "fail_threshold": [0.005, 0.08],
        "direct_qpos": [False, True],
    },
    "base_move_to_pose": {
        "base_x": [-0.4, 0.4],
        "base_y": [-0.4, 0.4],
        "base_yaw": [-0.8, 0.8],
        "steps": [100, 1200],
        "settle_steps": [0, 400],
        "max_joint_step": [0.001, 0.03],
        "fail_threshold": [0.005, 0.08],
        "direct_qpos": [False, True],
    },
    "head_camera_capture": {
        "width": [64, 640],
        "height": [48, 480],
        "include_depth": [False, True],
    },
    "base_lidar_scan": {
        "ray_count": [16, 361],
        "horizontal_fov_deg": [30.0, 360.0],
        "min_range": [0.0, 1.0],
        "max_range": [0.5, 10.0],
        "exclude_sensor_body": [False, True],
    },
}


def field_atomic_plan_id(*parts: Any) -> str:
    text = "|".join(str(part) for part in parts)
    return "field_atomic_plan_" + hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]


def field_atomic_plan_prompt(
    *,
    goal: str,
    planner_input: dict[str, Any] | None = None,
    max_steps: int = 8,
) -> str:
    planner_input = planner_input or {}
    return f"""
Generate a robot field atomic action plan.

Use only these actions:
{json.dumps(sorted(FIELD_ATOMIC_PARAMETER_GUIDANCE), ensure_ascii=False, indent=2)}

These are low-level field-style actions. Do not use task names such as
move_to_pregrasp, approach_object, place_object, or recover_from_joint_limit.
All motion targets and sensor parameters must be explicit parameters.

Parameter guidance:
{json.dumps(FIELD_ATOMIC_PARAMETER_GUIDANCE, ensure_ascii=False, indent=2)}

Arm orientation guidance:
- left_arm_move_to_position and right_arm_move_to_position can be position-only
  or pose-constrained.
- If control_frame is omitted, the robot uses control_frame="pinch" by
  default. Use control_frame="tcp" only for explicit wrist/TCP motion.
- To control end-effector orientation, provide target_quat_wxyz = [w, x, y, z].
- Do not output target_xmat.
- orientation_weight and orientation_threshold are reserved advanced fields;
  do not output them unless the caller explicitly requests orientation tuning.
- If no orientation is needed, omit target_quat_wxyz.

Goal:
{goal}

Planner input and memory feedback:
{json.dumps(planner_input, ensure_ascii=False, indent=2)}

If planner_input contains field_atomic_parameter_priors, use
recommended_from_success as positive parameter examples and
avoid_from_failure as negative parameter evidence.
Prefer parameters that stay within the observed successful range.
If a field atomic action has no prior evidence, fall back to the static
parameter guidance below.

Return exactly one JSON object:
{{
  "goal": "short goal",
  "steps": [
    {{
      "action": "one allowed field atomic action",
      "parameters": {{}},
      "reason": "short reason using planner_input if available"
    }}
  ],
  "constraints": ["short safety constraint"],
  "risk_notes": ["short risk note"],
  "evidence_ids": ["experience ids copied from planner_input"],
  "confidence": 0.0
}}

Constraints:
- steps must contain 1 to {max_steps} items.
- every action must exactly match an allowed field atomic action.
- parameters must stay inside the guidance ranges.
- prefer small, conservative base/torso/arm movements before larger moves.
- return JSON only.
{JSON_ONLY_LINE}
"""


def invoke_field_atomic_plan_llm(
    prompt: str,
    *,
    provider: str = "doubao",
    model: str = "",
) -> dict[str, Any]:
    raw = invoke_llm(
        prompt,
        provider=provider,
        model=model,
        system_prompt="You generate parameterized low-level robot actions and return JSON only.",
        temperature=0.25,
    )
    payload = parse_json_payload(raw, prefer_array=False)
    if not isinstance(payload, dict):
        raise RuntimeError("field atomic plan response must be a JSON object")
    return payload


def normalize_field_atomic_plan(
    raw_plan: dict[str, Any],
    *,
    goal: str = "",
    planner_input: dict[str, Any] | None = None,
    max_steps: int = 8,
) -> dict[str, Any]:
    if not isinstance(raw_plan, dict):
        raise RuntimeError("field atomic plan must be a JSON object")
    raw_steps = raw_plan.get("steps")
    if not isinstance(raw_steps, list) or not raw_steps:
        raise RuntimeError("field atomic plan must contain non-empty steps")
    if len(raw_steps) > max_steps:
        raise RuntimeError(f"field atomic plan has too many steps: {len(raw_steps)} > {max_steps}")

    steps: list[dict[str, Any]] = []
    for index, raw_step in enumerate(raw_steps):
        if not isinstance(raw_step, dict):
            raise RuntimeError(f"step[{index}] must be an object")
        action = str(raw_step.get("action") or "").strip()
        if action not in FIELD_ATOMIC_PARAMETER_GUIDANCE:
            raise RuntimeError(f"unsupported field atomic action: {action}")
        raw_params = raw_step.get("parameters") if isinstance(raw_step.get("parameters"), dict) else {}
        steps.append({
            "action": action,
            "parameters": sanitize_field_atomic_parameters(action, raw_params),
            "reason": str(raw_step.get("reason") or "").strip(),
        })

    planner_input = planner_input or {}
    allowed_ids = _planner_evidence_ids(planner_input)
    evidence_ids = [str(item) for item in raw_plan.get("evidence_ids") or [] if str(item) in allowed_ids]
    confidence = raw_plan.get("confidence", 0.0)
    if not isinstance(confidence, (int, float)):
        confidence = 0.0
    confidence = max(0.0, min(float(confidence), 1.0))
    return {
        "schema_version": "field_atomic_plan_v1",
        "plan_id": field_atomic_plan_id(goal, json.dumps(steps, sort_keys=True, ensure_ascii=False)),
        "goal": str(raw_plan.get("goal") or goal),
        "steps": steps,
        "constraints": [str(item) for item in raw_plan.get("constraints") or [] if str(item)][:8],
        "risk_notes": [str(item) for item in raw_plan.get("risk_notes") or [] if str(item)][:8],
        "evidence_ids": evidence_ids,
        "confidence": round(confidence, 4),
    }


def sanitize_field_atomic_parameters(action: str, parameters: dict[str, Any]) -> dict[str, Any]:
    guidance = FIELD_ATOMIC_PARAMETER_GUIDANCE.get(action, {})
    clean: dict[str, Any] = {}
    for key, value in parameters.items():
        key = str(key)
        if key not in guidance:
            continue
        bounded = _coerce_parameter(guidance[key], value)
        if bounded is not None:
            clean[key] = bounded
    return clean


def _coerce_parameter(bounds: Any, value: Any) -> Any:
    if isinstance(bounds, list) and len(bounds) == 2 and all(isinstance(item, bool) for item in bounds):
        return bool(value)
    if isinstance(bounds, list) and len(bounds) == 2 and all(isinstance(item, int) and not isinstance(item, bool) for item in bounds):
        if not isinstance(value, (int, float)):
            return None
        return int(max(int(bounds[0]), min(int(value), int(bounds[1]))))
    if isinstance(bounds, list) and len(bounds) == 2 and all(isinstance(item, (int, float)) and not isinstance(item, bool) for item in bounds):
        if not isinstance(value, (int, float)):
            return None
        return max(float(bounds[0]), min(float(value), float(bounds[1])))
    if isinstance(bounds, list) and len(bounds) == 3 and bounds[2] == 4:
        if not isinstance(value, (list, tuple)) or len(value) != 4:
            return None
        out = []
        for item in value:
            if not isinstance(item, (int, float)):
                return None
            out.append(float(item))
        return out
    if isinstance(bounds, list) and len(bounds) == 2 and all(isinstance(item, list) for item in bounds):
        if not isinstance(value, list) or len(value) != len(bounds[0]) or len(value) != len(bounds[1]):
            return None
        out = []
        for index, item in enumerate(value):
            if not isinstance(item, (int, float)):
                return None
            out.append(max(float(bounds[0][index]), min(float(item), float(bounds[1][index]))))
        return out
    return value


def _planner_evidence_ids(planner_input: dict[str, Any]) -> set[str]:
    ids: set[str] = set()
    text = json.dumps(planner_input, ensure_ascii=False)
    for token in text.replace('"', " ").replace(",", " ").replace("[", " ").replace("]", " ").split():
        if token.startswith("exp_") or token.startswith("field_"):
            ids.add(token.strip())
    return ids
