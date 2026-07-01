#!/usr/bin/env python3
"""FR5 direct experience runner with UR5e-compatible memory workflow."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "experience_system"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("EXPERIENCE_RUNTIME_SKILLS_ROOT", str(ROOT))
os.environ.setdefault("EXPERIENCE_RUNTIME_SKILLS_MODULE", "skills.registry")

from fr5_mujoco import FR5MotionRuntime
from experience_bridge import FR5ExperienceBridge
from experience_system.memory.v3 import MemoryV3Library, build_retrieval_key, make_memory_v3_entry
from experience_system.ur5e_core.critic import build_critic_result, critique_ur5e_failure_experience
from experience_system.ur5e_core.planner import plan_recovery_candidates
from skills import registry
from skills.field_atomic.action_io import result_to_dict
from skills.field_atomic.atomic_executor import FR5FieldAtomicSkillExecutor

DEFAULT_PREGRASP_HEIGHT = 0.08
DEFAULT_SCENE = ROOT / "assets" / "scene.xml"
DEFAULT_RESULT = ROOT / "results" / "fr5_experiment_result.json"
DEFAULT_PLAN = ROOT / "results" / "fr5_experiment_plan.json"
DEFAULT_EXPERIENCE_LIB = ROOT / "results" / "memory" / "fr5_experience_library.json"
DEFAULT_EXPERIENCE_TOP_K = 5
DEFAULT_DIVERSITY_LAMBDA = 0.35


def _convert_numpy(value: Any) -> Any:
    if hasattr(value, "tolist") and callable(value.tolist):
        return _convert_numpy(value.tolist())
    if hasattr(value, "item") and callable(value.item):
        try:
            return _convert_numpy(value.item())
        except Exception:
            pass
    if isinstance(value, dict):
        return {str(key): _convert_numpy(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_convert_numpy(item) for item in value]
    return value


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(_convert_numpy(payload), indent=2, ensure_ascii=False), encoding="utf-8")


def _resolve_local_path(raw_path: str | Path | None) -> Path | None:
    if raw_path is None:
        return None
    path = Path(raw_path)
    return path if path.is_absolute() else (ROOT / path).resolve()


def _stable_experience_id(*parts: Any) -> str:
    text = "|".join(str(part) for part in parts)
    return "fr5_" + hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]


def _partition(entry: Any) -> str:
    if hasattr(entry, "get_partition"):
        return str(entry.get_partition())
    return str(getattr(entry, "memory_partition", ""))


def _is_failed_memory(entry: Any) -> bool:
    result = getattr(entry, "result", None)
    if result is not None and getattr(result, "task_success", None) is False:
        return True
    if result is not None and not bool(getattr(result, "success", False)):
        return True
    return _partition(entry) == "failed_memory" or getattr(entry, "status", "") == "failure"


def _memory_record(entry: Any, score: float) -> dict[str, Any]:
    result = getattr(entry, "result", None)
    retrieval_key = getattr(entry, "retrieval_key", {}) or {}
    failure_taxonomy = getattr(entry, "failure_taxonomy", {}) or {}
    text_summary = getattr(entry, "text_summary", "") or getattr(entry, "summary", "") or ""
    used_as = "negative" if _is_failed_memory(entry) else "positive"
    return {
        "experience_id": getattr(entry, "experience_id", ""),
        "partition": _partition(entry),
        "score": float(score),
        "source": getattr(entry, "source", ""),
        "status": getattr(entry, "status", ""),
        "validation_status": getattr(entry, "validation_status", ""),
        "validation_source": getattr(entry, "validation_source", ""),
        "used_as": used_as,
        "result_success": bool(getattr(result, "success", False)),
        "task_success": bool(getattr(result, "task_success", False)),
        "summary": text_summary,
        "text_summary": text_summary,
        "text_summary_preview": text_summary[:240],
        "retrieval_key": retrieval_key,
        "plan_signature": retrieval_key.get("plan_signature", getattr(entry, "plan_signature", "")),
        "anomaly_state": getattr(entry, "anomaly_state", {}) or {},
        "failure_taxonomy": failure_taxonomy,
        "failure_stage": failure_taxonomy.get("failure_stage", ""),
        "failure_type": failure_taxonomy.get("failure_type", ""),
    }


def _default_plan(*, include_place: bool) -> list[dict[str, Any]]:
    steps: list[dict[str, Any]] = [
        {"action": "camera_rgbd_save", "parameters": {}},
        {"action": "detect_object_pose", "parameters": {"target_class": "apple"}},
    ]
    if include_place:
        steps.append({"action": "detect_object_pose", "parameters": {"target_class": "plate"}})
    steps.extend(
        [
            {"action": "create_fixed_vertical_grasp", "parameters": {}},
            {"action": "move_to_pregrasp", "parameters": {"dx": 0.0, "dy": 0.0, "dz": 0.08}},
            {"action": "approach_object", "parameters": {"dx": 0.0, "dy": 0.0, "dz": 0.0}},
            {"action": "close_gripper", "parameters": {}},
            {"action": "lift", "parameters": {"lift_height": 0.10}},
        ]
    )
    if include_place:
        steps.extend(
            [
                {"action": "move_lifted_object_to", "parameters": {"target": "plate"}},
                {"action": "open_gripper", "parameters": {}},
            ]
        )
    return steps


def _fr5_execution_parameters(action: str, parameters: dict[str, Any]) -> dict[str, Any]:
    params = dict(parameters)
    if action in {"move_to_pregrasp", "approach_object"}:
        params.setdefault("duration", 2.5)
    if action == "approach_object":
        params.setdefault("settle_steps", 250)
    if action == "close_gripper":
        params.setdefault("duration", 1.5)
    if action == "open_gripper":
        params.setdefault("duration", 1.0)
    if action == "lift":
        params.setdefault("duration", 2.5)
    if action == "move_lifted_object_to":
        params.setdefault("duration", 3.0)
        params.setdefault("height", 0.13)
        params.setdefault("compensate_held_object", True)
    if action == "go_home":
        params.setdefault("duration", 3.0)
    return params


def _step_target_class(step: dict[str, Any]) -> str:
    params = step.get("parameters") if isinstance(step.get("parameters"), dict) else {}
    return str(params.get("target_class") or params.get("target") or "").strip().lower()


def _normalize_fr5_plan_order(steps: list[dict[str, Any]], *, include_place: bool, place_target: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not include_place or not place_target:
        return steps, {"changed": False, "reason": "no_place_target"}
    place_target = str(place_target).strip().lower()
    plate_detect_steps: list[dict[str, Any]] = []
    rest: list[dict[str, Any]] = []
    uses_place_target = False
    for step in steps:
        action = str(step.get("action") or "")
        target = _step_target_class(step)
        if action == "detect_object_pose" and target == place_target:
            plate_detect_steps.append(step)
            continue
        if action == "move_lifted_object_to" and target == place_target:
            uses_place_target = True
        rest.append(step)
    if not uses_place_target and not plate_detect_steps:
        return steps, {"changed": False, "reason": "place_target_not_used"}
    if not plate_detect_steps:
        plate_detect_steps = [{"action": "detect_object_pose", "parameters": {"target_class": place_target}}]
    first_plate_detect = plate_detect_steps[0]
    insert_at = 0
    for index, step in enumerate(rest):
        action = str(step.get("action") or "")
        if action in {"create_fixed_vertical_grasp", "move_to_pregrasp", "approach_object", "close_gripper", "lift", "move_lifted_object_to"}:
            insert_at = index
            break
        insert_at = index + 1
    reordered = rest[:insert_at] + [first_plate_detect] + rest[insert_at:]
    return reordered, {
        "changed": reordered != steps,
        "reason": "place_target_detection_before_grasp",
        "place_target": place_target,
        "removed_duplicate_detect_count": max(0, len(plate_detect_steps) - 1),
    }


def _physical_failure_steps_for_critic(metrics: dict[str, Any]) -> list[dict[str, Any]]:
    steps = metrics.get("executed_recovery_steps")
    if isinstance(steps, list) and steps:
        return [
            {
                "action": str(step.get("action", "")),
                "parameters": step.get("parameters") if isinstance(step.get("parameters"), dict) else {},
            }
            for step in steps
            if isinstance(step, dict) and step.get("action")
        ]
    steps = metrics.get("llm_recovery_steps")
    if isinstance(steps, list) and steps:
        return [
            {
                "action": str(step.get("action", "")),
                "parameters": step.get("parameters") if isinstance(step.get("parameters"), dict) else {},
            }
            for step in steps
            if isinstance(step, dict) and step.get("action")
        ]
    return []


def _has_contact(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    return bool(value.get("left_contact") or value.get("right_contact") or value.get("contact"))


def _step_contact(report: dict[str, Any]) -> dict[str, Any]:
    raw = report.get("raw_result") if isinstance(report.get("raw_result"), dict) else {}
    record = raw.get("skill_record") if isinstance(raw.get("skill_record"), dict) else {}
    contact = record.get("contact") if isinstance(record.get("contact"), dict) else {}
    if contact:
        return contact
    extra = record.get("extra") if isinstance(record.get("extra"), dict) else {}
    contact = extra.get("contact") if isinstance(extra.get("contact"), dict) else {}
    return contact if isinstance(contact, dict) else {}


def deterministic_rule_critic(metrics: dict[str, Any]) -> dict[str, Any]:
    flags: list[dict[str, Any]] = []
    criteria = metrics.get("task_success_criteria") if isinstance(metrics.get("task_success_criteria"), dict) else {}
    reports = metrics.get("field_atomic_reports") if isinstance(metrics.get("field_atomic_reports"), list) else []
    executed = metrics.get("executed_recovery_steps") if isinstance(metrics.get("executed_recovery_steps"), list) else []

    invalid_steps = metrics.get("invalid_skill_steps") if isinstance(metrics.get("invalid_skill_steps"), list) else []
    if invalid_steps:
        flags.append({
            "rule": "invalid_skill_steps_in_plan",
            "stage": "recovery_plan",
            "evidence": f"plan contained {len(invalid_steps)} invalid step(s): {invalid_steps[:3]}",
        })

    failed_reports = [item for item in reports if isinstance(item, dict) and not bool(item.get("success"))]
    if failed_reports:
        first = failed_reports[0]
        flags.append({
            "rule": "skill_execution_failed",
            "stage": str(first.get("action") or "recovery_execution"),
            "evidence": str(first.get("message") or first.get("status") or "field atomic skill failed"),
        })

    normalization = metrics.get("plan_order_normalization")
    if isinstance(normalization, dict) and normalization.get("changed"):
        flags.append({
            "rule": "plan_order_normalized",
            "stage": "recovery_plan",
            "severity": "warning",
            "evidence": json.dumps(_convert_numpy(normalization), ensure_ascii=False),
        })

    required_targets = {str(metrics.get("target_class") or "apple").strip().lower()}
    place_target = str(metrics.get("place_target") or "").strip().lower()
    if place_target:
        required_targets.add(place_target)
    detected = metrics.get("detected_objects") if isinstance(metrics.get("detected_objects"), dict) else {}
    failed_detection_targets = {
        str(item.get("parameters", {}).get("target_class") or "").strip().lower()
        for item in reports
        if isinstance(item, dict)
        and item.get("action") == "detect_object_pose"
        and not bool(item.get("success"))
        and isinstance(item.get("parameters"), dict)
    }
    missing_targets = sorted(target for target in required_targets if target and target not in detected)
    if missing_targets or failed_detection_targets:
        flags.append({
            "rule": "perception_target_unavailable",
            "stage": "detection",
            "evidence": f"missing_targets={missing_targets}, failed_detection_targets={sorted(failed_detection_targets)}",
        })

    close_contact = {}
    lift_contact = {}
    for report in reports:
        if not isinstance(report, dict):
            continue
        if report.get("action") == "close_gripper":
            close_contact = _step_contact(report)
        elif report.get("action") == "lift":
            lift_contact = _step_contact(report)
    close_ok = _has_contact(close_contact)
    lift_ok = _has_contact(lift_contact)
    close_executed = any(isinstance(step, dict) and step.get("action") == "close_gripper" for step in executed)
    lift_executed = any(isinstance(step, dict) and step.get("action") == "lift" for step in executed)
    if close_executed and lift_executed and not close_ok and not lift_ok:
        flags.append({
            "rule": "no_contact_detected",
            "stage": "recovery_execution",
            "evidence": "close_gripper and lift executed, but no gripper contact was recorded",
        })
    elif close_ok and lift_executed and not lift_ok:
        flags.append({
            "rule": "contact_lost_during_lift",
            "stage": "recovery_execution",
            "evidence": "contact was present after close_gripper but absent after lift",
        })

    if criteria.get("type") == "apple_on_plate" and criteria.get("on_plate") is False:
        flags.append({
            "rule": "not_on_plate",
            "stage": "placement",
            "evidence": f"xy_error_to_plate={criteria.get('xy_error_to_plate')}, apple_z={criteria.get('apple_z')}",
        })

    if criteria.get("type") == "object_secured_and_lifted":
        apple_z = criteria.get("apple_z")
        contact = criteria.get("contact") if isinstance(criteria.get("contact"), dict) else {}
        if apple_z is not None and float(apple_z) <= 0.06:
            flags.append({
                "rule": "object_not_lifted",
                "stage": "recovery_execution",
                "evidence": f"apple_z={float(apple_z):.4f} <= 0.0600",
            })
        if not _has_contact(contact):
            flags.append({
                "rule": "grasp_not_secured",
                "stage": "recovery_execution",
                "evidence": "final task criteria contains no gripper contact",
            })

    return {"enabled": True, "rule_flags": flags, "flag_count": len(flags)}


def _apply_llm_critic_to_entry(entry: Any, llm_critic_result: dict[str, Any]) -> None:
    taxonomy = dict(getattr(entry, "failure_taxonomy", {}) or {})
    taxonomy["llm_critic"] = llm_critic_result
    for key in (
        "failure_stage",
        "failure_type",
        "failed_predicates",
        "failure_evidence",
        "corrective_direction",
        "missing_phases",
    ):
        value = llm_critic_result.get(key)
        if value:
            taxonomy[key] = value
    entry.failure_taxonomy = taxonomy
    if llm_critic_result.get("failure_type") and not getattr(entry.result, "failure_reason", "") and not llm_critic_result.get("error"):
        entry.result.failure_reason = llm_critic_result["failure_type"]


class ExperimentV4:
    def __init__(
        self,
        *,
        enable_viewer: bool = False,
        scene_xml: str | Path | None = None,
        save_plan: str | Path | None = None,
        experience_lib_path: str | Path | None = None,
        experience_write_path: str | Path | None = None,
        condition: str = "direct",
        condition_id: str = "direct",
        scenario_id: str = "fr5_direct_recovery",
        target_class: str = "apple",
        place_target: str = "plate",
    ) -> None:
        self.scene_xml = str(_resolve_local_path(scene_xml) or DEFAULT_SCENE)
        self.runtime = FR5MotionRuntime.from_scene(self.scene_xml, realtime=bool(enable_viewer))
        self.runtime.reset_home()
        self.executor = FR5FieldAtomicSkillExecutor(self.runtime, default_pregrasp_height=DEFAULT_PREGRASP_HEIGHT)
        self.viewer = None
        if enable_viewer:
            import mujoco.viewer

            self.viewer = mujoco.viewer.launch_passive(self.runtime.model, self.runtime.data)
            self.runtime.set_viewer(self.viewer)
        self.condition = condition
        self.condition_id = condition_id or "direct"
        self.scenario_id = scenario_id or "fr5_direct_recovery"
        self.target_class = target_class or "apple"
        self.place_target = place_target or ""
        self.save_plan_path = _resolve_local_path(save_plan)
        self.experience_lib_path = _resolve_local_path(experience_write_path) or _resolve_local_path(experience_lib_path)
        self.experience_library = MemoryV3Library.load(_resolve_local_path(experience_lib_path)) if experience_lib_path else None
        if self.experience_library is None and self.experience_lib_path is not None:
            self.experience_library = MemoryV3Library.load(self.experience_lib_path)
        self.experience_bridge = FR5ExperienceBridge(self)
        self.recovery_plan: dict[str, Any] | None = None
        self.metrics: dict[str, Any] = self.runtime.metrics
        self._task_history: list[dict[str, Any]] = []
        self.metrics.update(
            {
                "scenario_id": self.scenario_id,
                "condition_id": self.condition_id,
                "target_class": self.target_class,
                "place_target": self.place_target,
                "scene_xml": self.scene_xml,
            }
        )

    @property
    def model(self) -> mujoco.MjModel:
        return self.runtime.model

    @property
    def data(self) -> mujoco.MjData:
        return self.runtime.data

    def close(self) -> None:
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

    def _execute_one(self, action: str, parameters: dict[str, Any], *, index: int | None = None) -> dict[str, Any]:
        result = self.executor.execute(action, _fr5_execution_parameters(action, parameters))
        report = result_to_dict(result, index=index)
        report["parameters"] = dict(parameters)
        print(f"  {action}: success={result.success} status={result.status} message={result.message}")
        return report

    def _log_task(self, action: str, status: str, reason: str = "") -> None:
        self._task_history.append(
            {
                "action": str(action),
                "status": str(status),
                "reason": str(reason or ""),
            }
        )

    def _target_observation_status(self, *, include_place: bool) -> str:
        detected = self.metrics.get("detected_objects") if isinstance(self.metrics.get("detected_objects"), dict) else {}
        if not detected:
            return (
                "当前需要通过 camera_rgbd_save 和 detect_object_pose 在执行时获取目标状态；"
                "不要假设已有可靠目标位姿。"
            )
        compact: dict[str, Any] = {}
        targets = [self.target_class]
        if include_place and self.place_target:
            targets.append(self.place_target)
        for target in targets:
            record = detected.get(str(target).strip().lower())
            if isinstance(record, dict):
                compact[str(target)] = {
                    "observed_pos": record.get("observed_pos") or record.get("pos") or record.get("position"),
                    "status": record.get("status", "detected"),
                }
        return json.dumps(compact or detected, ensure_ascii=False)[:1500]

    def _gripper_status(self) -> str:
        contact = self.runtime._contact_summary()
        opening = None
        try:
            opening = float(self.runtime._gripper_opening())
        except Exception:
            opening = None
        label = "闭合" if opening is not None and opening > 0.02 else "张开"
        return json.dumps(
            {
                "state": label,
                "opening_command_m": opening,
                "contact": contact,
            },
            ensure_ascii=False,
        )

    def initialize(self, *, camera_ready: bool = True) -> None:
        if camera_ready:
            result = self.executor.execute("go_camera_ready", {})
            if not result.success:
                raise RuntimeError(f"go_camera_ready failed: {result.message}")
            self.metrics["camera_ready_result"] = result_to_dict(result)
            self.metrics["skill_results"] = []
            self._log_task("go_camera_ready", "SUCCESS", result.message or result.status)

    def _query_recovery_experiences(self, *, include_failed: bool = True) -> list[tuple[Any, float]]:
        try:
            experiences = self.experience_bridge.query_recovery_experiences(
                top_k=DEFAULT_EXPERIENCE_TOP_K,
                diversity_lambda=DEFAULT_DIVERSITY_LAMBDA,
                include_failed=include_failed,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"  [WARN] experience query failed: {exc}")
            experiences = []
        self.metrics["retrieved_memories"] = [_memory_record(entry, score) for entry, score in experiences]
        return experiences

    def _image_paths(self) -> list[str]:
        paths: list[str] = []
        camera = self.metrics.get("last_camera_rgbd") if isinstance(self.metrics.get("last_camera_rgbd"), dict) else {}
        if camera.get("rgb_path"):
            paths.append(str(camera["rgb_path"]))
        for item in (self.metrics.get("detected_objects") or {}).values():
            if isinstance(item, dict) and item.get("annotated_path"):
                paths.append(str(item["annotated_path"]))
        return [path for path in paths if Path(path).exists()]

    def _sanitize_steps(self, raw_steps: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        cleaned: list[dict[str, Any]] = []
        invalid: list[dict[str, Any]] = []
        for index, step in enumerate(raw_steps):
            if not isinstance(step, dict):
                invalid.append({"index": index, "reason": "step_not_object", "step": step})
                continue
            action = str(step.get("action") or "")
            params = step.get("parameters") if isinstance(step.get("parameters"), dict) else {}
            normalized, reason = registry.normalize_parameters(action, params)
            if normalized is None:
                invalid.append({"index": index, "action": action, "reason": reason, "step": step})
                continue
            cleaned.append({"action": action, "parameters": normalized})
        return cleaned, invalid

    def _plan_recovery(
        self,
        *,
        method: str,
        memory_policy: str,
        no_llm: bool,
        candidate_count: int,
        include_place: bool,
        experiences: list[tuple[Any, float]],
    ) -> list[dict[str, Any]]:
        fallback = _default_plan(include_place=include_place)
        if no_llm:
            self.metrics["executed_plan_source"] = "fallback_no_llm"
            return fallback
        target_observation_status = self._target_observation_status(include_place=include_place)
        gripper_status = self._gripper_status()
        if not self._task_history:
            self._log_task("observe", "SUCCESS", f"target={self.target_class}, place={self.place_target}")
        self.metrics["planner_prompt_inputs"] = {
            "task_history": list(self._task_history),
            "image_paths": self._image_paths(),
            "target": self.target_class,
            "condition": "direct",
            "scenario_id": self.scenario_id,
            "condition_id": self.condition_id,
            "condition_name": "fr5_direct_recovery",
            "task_stage": "direct_recovery",
            "success_criteria": "apple_on_plate" if include_place else "object_secured_and_lifted",
            "target_observation_status": target_observation_status,
            "gripper_status": gripper_status,
            "candidate_count": max(1, int(candidate_count or 1)),
        }
        try:
            candidates = plan_recovery_candidates(
                task_history=self._task_history,
                image_paths=self.metrics["planner_prompt_inputs"]["image_paths"],
                target=self.target_class,
                experiences=experiences,
                condition="direct",
                scenario_id=self.scenario_id,
                condition_id=self.condition_id,
                condition_name="fr5_direct_recovery",
                task_stage="direct_recovery",
                success_criteria="apple_on_plate" if include_place else "object_secured_and_lifted",
                target_observation_status=target_observation_status,
                gripper_status=gripper_status,
                candidate_count=max(1, int(candidate_count or 1)),
            )
        except Exception as exc:  # noqa: BLE001
            self.metrics["planner_error"] = str(exc)
            self.metrics["executed_plan_source"] = "fallback_planner_error"
            return fallback
        self.metrics["candidate_plans"] = candidates
        raw_steps = candidates[0]["steps"] if candidates else []
        steps, invalid = self._sanitize_steps(raw_steps)
        steps, order_report = _normalize_fr5_plan_order(steps, include_place=include_place, place_target=self.place_target)
        self.metrics["plan_order_normalization"] = order_report
        self.metrics["invalid_skill_steps"] = invalid
        if not steps:
            self.metrics["executed_plan_source"] = "fallback_invalid_llm_plan"
            return fallback
        self.metrics["executed_plan_source"] = f"{method}_llm"
        self.metrics["llm_recovery_steps"] = steps
        return steps

    def _execute_steps(self, steps: list[dict[str, Any]], *, stop_on_failure: bool = True) -> list[dict[str, Any]]:
        reports: list[dict[str, Any]] = []
        executed: list[dict[str, Any]] = []
        for index, step in enumerate(steps):
            action = str(step.get("action") or "")
            params = step.get("parameters") if isinstance(step.get("parameters"), dict) else {}
            print(f"[{index + 1}/{len(steps)}] {action}")
            report = self._execute_one(action, params, index=index)
            reports.append(report)
            status = "SUCCESS" if bool(report.get("success")) else "FAILURE"
            self._log_task(action, status, str(report.get("message") or report.get("status") or ""))
            if action == "close_gripper":
                self.metrics["contact_after_close"] = _step_contact(report) or self.runtime._contact_summary()
            elif action == "lift":
                self.metrics["contact_after_lift"] = _step_contact(report) or self.runtime._contact_summary()
            executed.append(
                {
                    "action": action,
                    "parameters": params,
                    "success": bool(report.get("success")),
                    "status": str(report.get("status") or ""),
                    "message": str(report.get("message") or ""),
                }
            )
            if stop_on_failure and not bool(report.get("success")):
                break
        self.metrics["executed_recovery_steps"] = executed
        self.metrics["field_atomic_reports"] = reports
        self.metrics["task_history"] = list(self._task_history)
        return reports

    def _geom_pos(self, name: str) -> np.ndarray | None:
        geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)
        if geom_id < 0:
            return None
        return self.data.geom_xpos[geom_id].copy()

    def _evaluate_task_success(self, *, include_place: bool) -> bool:
        apple = self._geom_pos("apple0")
        plate = self._geom_pos("plate_geom")
        contact = self.runtime._contact_summary()
        if apple is None:
            self.metrics["task_success_criteria"] = {"type": "missing_apple_geom"}
            return False
        apple_z = float(apple[2])
        on_plate = False
        xy_error = None
        if include_place and plate is not None:
            xy_error = float(np.linalg.norm(apple[:2] - plate[:2]))
            on_plate = bool(xy_error <= 0.09 and apple_z >= float(plate[2]))
            success = on_plate
        else:
            success = bool(apple_z > 0.06 and (contact.get("left_contact") or contact.get("right_contact")))
        self.metrics["task_success_criteria"] = {
            "type": "apple_on_plate" if include_place else "object_secured_and_lifted",
            "apple_pos": apple.tolist(),
            "plate_pos": None if plate is None else plate.tolist(),
            "apple_z": apple_z,
            "xy_error_to_plate": xy_error,
            "on_plate": on_plate,
            "contact": contact,
        }
        self.metrics["apple_z_after_recovery"] = apple_z
        self.metrics["task_success"] = bool(success)
        return bool(success)

    def _build_experience_entry(self, *, task_success: bool, failure_reason: str, method: str, memory_policy: str, time_cost: float) -> Any:
        steps = self.metrics.get("executed_recovery_steps") if isinstance(self.metrics.get("executed_recovery_steps"), list) else []
        reports = self.metrics.get("field_atomic_reports") if isinstance(self.metrics.get("field_atomic_reports"), list) else []
        memory_role = "success_prior" if task_success else "failure_case"
        base_taxonomy = {
            "failure_stage": "" if task_success else "recovery_execution",
            "failure_type": "" if task_success else failure_reason,
            "failure_reason": failure_reason,
        }
        entry = make_memory_v3_entry(
            condition_id=self.condition_id,
            scenario_id=self.scenario_id,
            available_actions=registry.allowed_actions(self.scenario_id),
            skill_sequence=steps,
            task_success=task_success,
            failure_reason=failure_reason,
            source="simulation",
            summary=("Successful" if task_success else "Failed") + f" FR5 direct recovery: {failure_reason or 'ok'}",
            metadata={"robot": "fr5", "source_tool": "fr5_run_experiment_v4", "method": method, "memory_policy": memory_policy},
            validation_evidence={"actions": reports, "skill_results": self.metrics.get("skill_results", [])},
            recovery_plan={"steps": steps},
            execution_feedback={"field_atomic_reports": reports, "skill_results": self.metrics.get("skill_results", [])},
            anomaly_state=self.experience_bridge.anomaly_state(),
            failure_taxonomy=base_taxonomy,
            validation_status="simulation_success" if task_success else "failed",
            validation_source="fr5_mujoco_direct",
            critic_result={},
            time_cost=time_cost,
            memory_tags={"memory_type": "episodic", "memory_scope": "condition", "memory_role": memory_role},
        )
        entry.experience_id = _stable_experience_id(self.scenario_id, self.condition_id, json.dumps(steps, sort_keys=True, ensure_ascii=False), task_success, failure_reason)

        is_failure_entry = not bool(task_success)
        llm_critic_result: dict[str, Any] = {}
        rule_critic_result: dict[str, Any] = {"enabled": True, "rule_flags": [], "flag_count": 0}
        if is_failure_entry:
            try:
                llm_critic_result = critique_ur5e_failure_experience(
                    method=method,
                    memory_policy=memory_policy,
                    metrics=self.metrics,
                    task_history=getattr(self, "_task_history", []),
                    recovery_steps=_physical_failure_steps_for_critic(self.metrics),
                    retrieved_memories=self.metrics.get("retrieved_memories", []),
                )
                self.metrics["failure_experience_critic"] = llm_critic_result
                if llm_critic_result.get("enabled"):
                    _apply_llm_critic_to_entry(entry, llm_critic_result)
            except Exception as exc:  # noqa: BLE001
                llm_critic_result = {"enabled": True, "error": str(exc)}
                self.metrics["failure_experience_critic"] = llm_critic_result
                self.metrics["failure_experience_critic_error"] = str(exc)

            try:
                rule_critic_result = deterministic_rule_critic(self.metrics)
                self.metrics["failure_rule_critic"] = rule_critic_result
                if rule_critic_result.get("enabled") and rule_critic_result.get("rule_flags"):
                    taxonomy = dict(getattr(entry, "failure_taxonomy", {}) or {})
                    taxonomy["rule_critic"] = rule_critic_result
                    entry.failure_taxonomy = taxonomy
            except Exception as exc:  # noqa: BLE001
                rule_critic_result = {"enabled": True, "error": str(exc), "rule_flags": [], "flag_count": 0}
                self.metrics["failure_rule_critic"] = rule_critic_result
                self.metrics["failure_rule_critic_error"] = str(exc)

        critic_result = build_critic_result(
            rule_result=rule_critic_result,
            llm_result=llm_critic_result,
            is_failure=is_failure_entry,
        )
        entry.critic_result = critic_result
        self.metrics["critic_result"] = critic_result
        entry.retrieval_key = build_retrieval_key(entry)
        return entry

    def save_experience(self, entry: Any) -> Any:
        return self.experience_bridge.save_entry(entry)

    def run_recovery(
        self,
        *,
        method: str = "direct_memory",
        memory_policy: str = "hierarchical",
        no_llm: bool = False,
        candidate_count: int = 1,
        stop_on_failure: bool = True,
        camera_ready: bool = True,
        include_place: bool = True,
        save_experience_entry: bool = True,
    ) -> dict[str, Any]:
        t0 = time.time()
        self.initialize(camera_ready=camera_ready)
        include_failed = memory_policy != "no_failed"
        experiences = self._query_recovery_experiences(include_failed=include_failed) if memory_policy != "none" else []
        steps = self._plan_recovery(
            method=method,
            memory_policy=memory_policy,
            no_llm=no_llm,
            candidate_count=candidate_count,
            include_place=include_place,
            experiences=experiences,
        )
        self.recovery_plan = {"steps": steps, "method": method, "memory_policy": memory_policy}
        reports = self._execute_steps(steps, stop_on_failure=stop_on_failure)
        task_success = self._evaluate_task_success(include_place=include_place)
        failed_report = next((item for item in reports if not bool(item.get("success"))), None)
        failure_reason = "" if task_success else str((failed_report or {}).get("message") or (failed_report or {}).get("status") or "task_not_successful")
        self.metrics["failure_reason"] = failure_reason
        time_cost = time.time() - t0
        self.metrics["time_costs"] = {"total": round(time_cost, 3)}
        entry = self._build_experience_entry(task_success=task_success, failure_reason=failure_reason, method=method, memory_policy=memory_policy, time_cost=time_cost)
        self._last_experience_entry = entry
        if save_experience_entry and self.experience_library is not None:
            self.save_experience(entry)
            self.metrics["experience_saved"] = True
            self.metrics["experience_id"] = getattr(entry, "experience_id", "")
        elif self.experience_library is not None:
            self.metrics["experience_saved"] = False
        return {
            "schema_version": "fr5_experiment_v4_result_v1",
            "method": method,
            "memory_policy": memory_policy,
            "scenario_id": self.scenario_id,
            "condition_id": self.condition_id,
            "scene_xml": self.scene_xml,
            "target_class": self.target_class,
            "place_target": self.place_target,
            "task_success": task_success,
            "failure_reason": failure_reason,
            "executed_plan_source": self.metrics.get("executed_plan_source", ""),
            "llm_recovery_steps": self.metrics.get("llm_recovery_steps", []),
            "executed_recovery_steps": self.metrics.get("executed_recovery_steps", []),
            "retrieved_memories": self.metrics.get("retrieved_memories", []),
            "candidate_plans": self.metrics.get("candidate_plans", []),
            "candidate_rejections": self.metrics.get("invalid_skill_steps", []),
            "field_atomic_reports": reports,
            "metrics": self.metrics,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one FR5 direct recovery experiment.")
    parser.add_argument("--method", default="direct_memory")
    parser.add_argument("--memory-policy", default="")
    parser.add_argument("--scene-xml", type=Path, default=DEFAULT_SCENE)
    parser.add_argument("--save", type=Path, default=DEFAULT_RESULT)
    parser.add_argument("--save-plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--experience-read", type=Path, default=None)
    parser.add_argument("--experience-write", type=Path, default=DEFAULT_EXPERIENCE_LIB)
    parser.add_argument("--scenario-id", default="fr5_direct_recovery")
    parser.add_argument("--condition-id", default="direct")
    parser.add_argument("--target-class", default="apple")
    parser.add_argument("--place-target", default="plate")
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--no-camera-ready", action="store_true")
    parser.add_argument("--no-llm", action="store_true")
    parser.add_argument("--recovery-candidate-count", type=int, default=1)
    parser.add_argument("--continue-on-failure", action="store_true")
    return parser.parse_args()


def _method_policy(method: str, explicit: str = "") -> str:
    if explicit:
        return explicit
    if method == "direct_llm_weak":
        return "none"
    if method == "hierarchical_no_failed":
        return "no_failed"
    return "hierarchical"


def main() -> int:
    args = parse_args()
    policy = _method_policy(args.method, args.memory_policy)
    exp = ExperimentV4(
        enable_viewer=args.viewer,
        scene_xml=args.scene_xml,
        save_plan=args.save_plan,
        experience_lib_path=args.experience_read,
        experience_write_path=args.experience_write,
        condition_id=args.condition_id,
        scenario_id=args.scenario_id,
        target_class=args.target_class,
        place_target=args.place_target,
    )
    try:
        result = exp.run_recovery(
            method=args.method,
            memory_policy=policy,
            no_llm=args.no_llm,
            candidate_count=args.recovery_candidate_count,
            stop_on_failure=not args.continue_on_failure,
            camera_ready=not args.no_camera_ready,
            include_place=bool(args.place_target),
        )
        _write_json(args.save, result)
        _write_json(args.save_plan, exp.recovery_plan or {"steps": []})
        print(f"saved result: {args.save}")
        print(f"saved plan: {args.save_plan}")
        return 0 if result.get("task_success") else 1
    finally:
        exp.close()


if __name__ == "__main__":
    raise SystemExit(main())
