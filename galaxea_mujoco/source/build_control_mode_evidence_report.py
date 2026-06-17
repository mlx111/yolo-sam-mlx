"""Compare physical-actuator sandbox control modes."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("MUJOCO_GL", "osmesa")

from source.run_r1pro_task_chain import run_task_chain


DEFAULT_CASES = (
    ("G3", "clean", "g3_cautious_place"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build evidence report for R1Pro physical sandbox control modes.")
    parser.add_argument("--scenario", choices=["G3"], action="append", default=[])
    parser.add_argument("--condition", choices=["clean", "place_occupied"], action="append", default=[])
    parser.add_argument("--candidate-id", action="append", default=[])
    parser.add_argument("--control-mode", choices=["ideal", "direct", "direct_position", "physical", "actuator"], action="append", default=[])
    parser.add_argument("--control-profile", action="append", default=[], help="Physical control profile id. Can be repeated.")
    parser.add_argument("--sandbox-initial-state", type=Path, default=None)
    parser.add_argument("--save-json", type=Path, required=True)
    parser.add_argument("--save-md", type=Path, required=True)
    return parser.parse_args()


def _load_json(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _cases(args: argparse.Namespace) -> list[tuple[str, str, str]]:
    if not args.scenario and not args.condition and not args.candidate_id:
        return list(DEFAULT_CASES)
    scenarios = args.scenario or ["G3"]
    conditions = args.condition or ["clean"]
    candidate_ids = args.candidate_id or [""]
    out: list[tuple[str, str, str]] = []
    for scenario in scenarios:
        for condition in conditions:
            for candidate_id in candidate_ids:
                resolved = candidate_id
                if not resolved:
                    resolved = "g3_cautious_place"
                out.append((scenario, condition, resolved))
    return out


def _failed_skills(result: dict[str, Any]) -> list[dict[str, Any]]:
    failed = []
    for item in result.get("skill_trace") or []:
        if not isinstance(item, dict) or item.get("success"):
            continue
        failed.append({
            "skill": item.get("skill", ""),
            "error": item.get("error"),
            "message": item.get("message", ""),
            "motion": item.get("motion") if isinstance(item.get("motion"), dict) else {},
        })
    return failed


def _summarize_result(
    *,
    scenario: str,
    condition: str,
    candidate_id: str,
    control_mode: str,
    result: dict[str, Any] | None,
    exception: str = "",
    elapsed_s: float,
) -> dict[str, Any]:
    if result is None:
        return {
            "scenario": scenario,
            "condition": condition,
            "candidate_id": candidate_id,
            "requested_control_mode": control_mode,
            "runner_exception": exception,
            "runner_completed": False,
            "success": False,
            "task_success": False,
            "failure_reason": exception,
            "elapsed_s": round(elapsed_s, 4),
            "control_execution": {},
            "failed_skills": [],
        }
    metrics = result.get("metrics") if isinstance(result.get("metrics"), dict) else {}
    return {
        "scenario": scenario,
        "condition": condition,
        "candidate_id": candidate_id,
        "requested_control_mode": control_mode,
        "runner_exception": exception,
        "runner_completed": True,
        "success": bool(result.get("success")),
        "task_success": bool(result.get("task_success")),
        "failure_reason": str(result.get("failure_reason") or ""),
        "elapsed_s": round(elapsed_s, 4),
        "control_execution": metrics.get("control_execution") if isinstance(metrics.get("control_execution"), dict) else {},
        "sandbox_dynamics_profile": metrics.get("sandbox_dynamics_profile") if isinstance(metrics.get("sandbox_dynamics_profile"), dict) else {},
        "motion_critic": metrics.get("motion_critic") if isinstance(metrics.get("motion_critic"), dict) else {},
        "contact_stability": metrics.get("contact_stability") if isinstance(metrics.get("contact_stability"), dict) else {},
        "object_lift": metrics.get("object_lift"),
        "failed_skills": _failed_skills(result),
        "skill_count": len(result.get("skill_trace") or []),
    }


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    sandbox_initial_state = _load_json(args.sandbox_initial_state)
    control_modes = args.control_mode or ["physical"]
    control_profiles = args.control_profile or ["real_driver_like_v1"]
    rollouts: list[dict[str, Any]] = []

    for scenario, condition, candidate_id in _cases(args):
        for control_mode in control_modes:
            profiles = control_profiles if control_mode in {"physical", "actuator"} else [""]
            for control_profile in profiles:
                started = time.time()
                result_dict: dict[str, Any] | None = None
                exception = ""
                try:
                    result = run_task_chain(
                        scenario,
                        condition,
                        control_mode,
                        candidate_id=candidate_id,
                        control_profile=control_profile or "real_driver_like_v1",
                        sandbox_initial_state=sandbox_initial_state,
                    )
                    result_dict = result.to_dict()
                except Exception as exc:  # Keep the report useful even when physical mode fails hard.
                    exception = f"{type(exc).__name__}: {exc}"
                elapsed_s = time.time() - started
                item = _summarize_result(
                    scenario=scenario,
                    condition=condition,
                    candidate_id=candidate_id,
                    control_mode=control_mode,
                    result=result_dict,
                    exception=exception,
                    elapsed_s=elapsed_s,
                )
                item["requested_control_profile"] = control_profile
                rollouts.append(item)

    by_execution = Counter(
        str((item.get("control_execution") or {}).get("execution_type") or "unknown")
        for item in rollouts
    )
    failed_skill_counts = Counter()
    for item in rollouts:
        for failed in item.get("failed_skills") or []:
            failed_skill_counts[str(failed.get("skill") or "")] += 1

    return {
        "schema_version": "control_mode_evidence_report_v1",
        "sandbox_initial_state": str(args.sandbox_initial_state) if args.sandbox_initial_state else "",
        "case_count": len(_cases(args)),
        "rollout_count": len(rollouts),
        "summary": {
            "runner_completed_count": sum(1 for item in rollouts if item["runner_completed"]),
            "success_count": sum(1 for item in rollouts if item["success"]),
            "task_success_count": sum(1 for item in rollouts if item["task_success"]),
            "execution_type_counts": dict(by_execution),
            "failed_skill_counts": dict(failed_skill_counts),
            "physical_rollout_count": sum(
                1 for item in rollouts
                if (item.get("control_execution") or {}).get("execution_type") == "physical_actuator"
            ),
            "direct_rollout_count": sum(
                1 for item in rollouts
                if (item.get("control_execution") or {}).get("execution_type") == "direct_position"
            ),
        },
        "rollouts": rollouts,
        "paper_wording": {
            "safe_claim": (
                "The implementation uses physical-actuator sandbox execution as the default path and keeps "
                "direct-position only as an explicit compatibility or debugging baseline."
            ),
            "avoid_claim": (
                "Do not claim the physical-actuator sandbox matches real robot IK/control without real FK/IK alignment and response calibration data."
            ),
        },
    }


def render_markdown(report: dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "# Control Mode Evidence Report",
        "",
        "This report uses physical-actuator sandbox execution by default; direct-position is retained only as an explicit compatibility or debugging baseline.",
        "",
        "## Summary",
        "",
        f"- Rollouts: {report['rollout_count']}",
        f"- Completed runners: {summary['runner_completed_count']}",
        f"- Successful task chains: {summary['success_count']}",
        f"- Task-success count: {summary['task_success_count']}",
        f"- Execution types: `{json.dumps(summary['execution_type_counts'], ensure_ascii=False, sort_keys=True)}`",
        f"- Failed skills: `{json.dumps(summary['failed_skill_counts'], ensure_ascii=False, sort_keys=True)}`",
        "",
        "## Rollouts",
        "",
        "| Scenario | Condition | Candidate | Mode | Profile | Arm control | Execution | Success | Task success | Failure | Failed skills |",
        "|---|---|---|---|---|---|---|---:|---:|---|---|",
    ]
    for item in report["rollouts"]:
        control_execution = item.get("control_execution") or {}
        execution = control_execution.get("execution_type", "")
        profile = control_execution.get("control_profile_id") or item.get("requested_control_profile") or ""
        arm_control = control_execution.get("arm_control_mode") or ""
        failed_skills = ", ".join(str(failed.get("skill") or "") for failed in item.get("failed_skills") or [])
        lines.append(
            f"| {item['scenario']} | {item['condition']} | {item['candidate_id']} | "
            f"{item['requested_control_mode']} | {profile} | {arm_control} | {execution} | {item['success']} | "
            f"{item['task_success']} | {item['failure_reason']} | {failed_skills} |"
        )
    lines.extend([
        "",
        "## Paper Wording",
        "",
        f"- Safe claim: {report['paper_wording']['safe_claim']}",
        f"- Avoid claim: {report['paper_wording']['avoid_claim']}",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    report = build_report(args)
    args.save_json.parent.mkdir(parents=True, exist_ok=True)
    args.save_md.parent.mkdir(parents=True, exist_ok=True)
    args.save_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    args.save_md.write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps({
        "schema_version": report["schema_version"],
        "rollout_count": report["rollout_count"],
        "summary": report["summary"],
        "save_json": str(args.save_json),
        "save_md": str(args.save_md),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
