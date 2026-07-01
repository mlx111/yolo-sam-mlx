"""Run physical sandbox perturbation sweeps and summarize failure sensitivity."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
GALAXEA_ROOT = ROOT.parent / "galaxea_mujoco"
if str(GALAXEA_ROOT) not in sys.path:
    sys.path.insert(0, str(GALAXEA_ROOT))

os.environ.setdefault("MUJOCO_GL", "osmesa")

from source.legacy_r1pro.run_r1pro_task_chain import run_task_chain


DEFAULT_VARIANTS: tuple[dict[str, Any], ...] = (
    {"variant_id": "nominal", "perturbations": {}},
    {"variant_id": "pose_noise_x_plus_2cm", "perturbations": {"object_pose_noise_xyz": [0.02, 0.0, 0.0]}},
    {"variant_id": "pose_noise_y_plus_2cm", "perturbations": {"object_pose_noise_xyz": [0.0, 0.02, 0.0]}},
    {"variant_id": "control_delay_3_steps", "perturbations": {"actuation_delay_steps": 3}},
    {"variant_id": "low_gain_0_75", "perturbations": {"controller_gain_scale": 0.75}},
    {"variant_id": "gripper_underclose_1cm", "perturbations": {"gripper_closure_bias": 0.01}},
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build physical sandbox control perturbation evidence report.")
    parser.add_argument("--scenario", default="G3")
    parser.add_argument("--condition", default="clean")
    parser.add_argument("--candidate-id", default="g3_cautious_place")
    parser.add_argument("--control-profile", default="real_driver_like_v1")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--save-json", type=Path, required=True)
    parser.add_argument("--save-md", type=Path, required=True)
    return parser.parse_args()


def _failed_skills(result: dict[str, Any]) -> list[dict[str, Any]]:
    failed: list[dict[str, Any]] = []
    for item in result.get("skill_trace") or []:
        if not isinstance(item, dict) or item.get("success"):
            continue
        failed.append({
            "skill": item.get("skill") or item.get("name") or "",
            "message": item.get("message") or "",
            "error": item.get("error"),
            "motion": item.get("motion") if isinstance(item.get("motion"), dict) else {},
        })
    return failed


def _failure_labels(result: dict[str, Any]) -> list[str]:
    labels: list[str] = []
    reason = str(result.get("failure_reason") or "")
    if reason:
        labels.append(reason)
    metrics = result.get("metrics") if isinstance(result.get("metrics"), dict) else {}
    motion = metrics.get("motion_critic") if isinstance(metrics.get("motion_critic"), dict) else {}
    for flag in motion.get("flags") or motion.get("rule_flags") or []:
        if isinstance(flag, dict):
            name = str(flag.get("name") or flag.get("type") or "")
            if name:
                labels.append(name)
    contact = metrics.get("contact_stability") if isinstance(metrics.get("contact_stability"), dict) else {}
    if contact.get("slip_risk_high"):
        labels.append("slip_risk_high")
    if metrics.get("object_lift") is not None and float(metrics.get("object_lift") or 0.0) < 0.02:
        labels.append("object_not_lifted")
    for failed in _failed_skills(result):
        skill = str(failed.get("skill") or "")
        if skill:
            labels.append(f"skill_failed:{skill}")
    return sorted(set(labels))


def _summarize_rollout(
    *,
    variant: dict[str, Any],
    repeat_index: int,
    result: dict[str, Any] | None,
    exception: str,
    elapsed_s: float,
) -> dict[str, Any]:
    base = {
        "variant_id": variant["variant_id"],
        "repeat_index": repeat_index,
        "perturbations": variant["perturbations"],
        "runner_completed": result is not None,
        "runner_exception": exception,
        "elapsed_s": round(elapsed_s, 4),
    }
    if result is None:
        return {
            **base,
            "success": False,
            "task_success": False,
            "failure_reason": exception,
            "failure_labels": [exception] if exception else [],
            "failed_skills": [],
            "metrics": {},
        }
    metrics = result.get("metrics") if isinstance(result.get("metrics"), dict) else {}
    selected_metrics = {
        "object_lift": metrics.get("object_lift"),
        "place_error": metrics.get("place_error"),
        "control_execution": metrics.get("control_execution") if isinstance(metrics.get("control_execution"), dict) else {},
        "sandbox_dynamics_profile": metrics.get("sandbox_dynamics_profile") if isinstance(metrics.get("sandbox_dynamics_profile"), dict) else {},
        "motion_critic": metrics.get("motion_critic") if isinstance(metrics.get("motion_critic"), dict) else {},
        "contact_stability": metrics.get("contact_stability") if isinstance(metrics.get("contact_stability"), dict) else {},
    }
    return {
        **base,
        "success": bool(result.get("success")),
        "task_success": bool(result.get("task_success")),
        "failure_reason": str(result.get("failure_reason") or ""),
        "failure_labels": _failure_labels(result),
        "failed_skills": _failed_skills(result),
        "metrics": selected_metrics,
    }


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    rollouts: list[dict[str, Any]] = []
    repeat = max(1, int(args.repeat))
    for variant in DEFAULT_VARIANTS:
        for repeat_index in range(repeat):
            started = time.time()
            result_dict: dict[str, Any] | None = None
            exception = ""
            try:
                result = run_task_chain(
                    args.scenario,
                    args.condition,
                    "physical",
                    candidate_id=args.candidate_id,
                    control_profile=args.control_profile,
                    sandbox_initial_state=dict(variant["perturbations"]),
                )
                result_dict = result.to_dict()
            except Exception as exc:
                exception = f"{type(exc).__name__}: {exc}"
            rollouts.append(_summarize_rollout(
                variant=variant,
                repeat_index=repeat_index,
                result=result_dict,
                exception=exception,
                elapsed_s=time.time() - started,
            ))

    by_variant: dict[str, dict[str, Any]] = {}
    failure_counter_by_variant: dict[str, Counter[str]] = defaultdict(Counter)
    for rollout in rollouts:
        variant_id = str(rollout["variant_id"])
        row = by_variant.setdefault(variant_id, {
            "variant_id": variant_id,
            "perturbations": rollout["perturbations"],
            "rollout_count": 0,
            "runner_completed_count": 0,
            "success_count": 0,
            "task_success_count": 0,
        })
        row["rollout_count"] += 1
        row["runner_completed_count"] += 1 if rollout["runner_completed"] else 0
        row["success_count"] += 1 if rollout["success"] else 0
        row["task_success_count"] += 1 if rollout["task_success"] else 0
        for label in rollout.get("failure_labels") or []:
            failure_counter_by_variant[variant_id][str(label)] += 1
    for variant_id, row in by_variant.items():
        total = max(1, int(row["rollout_count"]))
        row["success_rate"] = round(float(row["success_count"]) / total, 4)
        row["task_success_rate"] = round(float(row["task_success_count"]) / total, 4)
        row["failure_label_counts"] = dict(failure_counter_by_variant[variant_id])

    nominal = by_variant.get("nominal", {})
    nominal_success_rate = float(nominal.get("success_rate") or 0.0)
    for row in by_variant.values():
        row["success_rate_delta_vs_nominal"] = round(float(row.get("success_rate") or 0.0) - nominal_success_rate, 4)

    failure_counts = Counter()
    for rollout in rollouts:
        for label in rollout.get("failure_labels") or []:
            failure_counts[str(label)] += 1

    return {
        "schema_version": "physical_sandbox_perturbation_report_v1",
        "scenario": args.scenario,
        "condition": args.condition,
        "candidate_id": args.candidate_id,
        "control_mode": "physical",
        "control_profile": args.control_profile,
        "repeat": repeat,
        "variant_count": len(DEFAULT_VARIANTS),
        "rollout_count": len(rollouts),
        "summary": {
            "runner_completed_count": sum(1 for item in rollouts if item["runner_completed"]),
            "success_count": sum(1 for item in rollouts if item["success"]),
            "task_success_count": sum(1 for item in rollouts if item["task_success"]),
            "failure_label_counts": dict(failure_counts),
            "variant_summary": list(by_variant.values()),
        },
        "rollouts": rollouts,
        "paper_wording": {
            "safe_claim": "The physical MuJoCo sandbox can sweep pose, delay, gain, and gripper perturbations and report which failures are sensitive to those controls.",
            "avoid_claim": "Do not claim real-driver calibration or real-robot robustness without measured hardware response data.",
        },
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Physical Sandbox Perturbation Report",
        "",
        f"- Scenario: {report['scenario']}",
        f"- Condition: {report['condition']}",
        f"- Candidate: {report['candidate_id']}",
        f"- Control profile: {report['control_profile']}",
        f"- Rollouts: {report['rollout_count']}",
        "",
        "## Variant Summary",
        "",
        "| Variant | Perturbations | Success | Task success | Delta vs nominal | Failure labels |",
        "|---|---|---:|---:|---:|---|",
    ]
    for row in report["summary"]["variant_summary"]:
        lines.append(
            "| "
            + " | ".join([
                str(row["variant_id"]),
                f"`{json.dumps(row['perturbations'], ensure_ascii=False, sort_keys=True)}`",
                str(row["success_count"]),
                str(row["task_success_count"]),
                str(row["success_rate_delta_vs_nominal"]),
                f"`{json.dumps(row['failure_label_counts'], ensure_ascii=False, sort_keys=True)}`",
            ])
            + " |"
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
        "success_count": report["summary"]["success_count"],
        "save_json": str(args.save_json),
        "save_md": str(args.save_md),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
