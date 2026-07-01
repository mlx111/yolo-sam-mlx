"""Ablate gap-derived sandbox calibration effects."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experience_adapters import R1ProMujocoAdapter
from experience_core import ExperienceLibrary, apply_critic
from source.legacy_r1pro.candidate_sandbox import (
    DEFAULT_CRITIC_THRESHOLDS,
    _calibration_dict,
    _galaxea_workdir,
    calibration_risk,
    sandbox_score,
    select_sandbox_calibration,
)
from source.legacy_r1pro.run_r1pro_memory_policy_smoke import candidates_for_scenario, object_class_for_scenario
from source.legacy_r1pro.run_r1pro_task_chain import run_task_chain


VARIANTS = [
    "sandbox_no_calibration",
    "sandbox_score_calibration_only",
    "sandbox_pose_and_score_calibration",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare sandbox rollout with no, score-only, and pose+score calibration.")
    parser.add_argument("--scenario", choices=["G3"], required=True)
    parser.add_argument("--condition", choices=["clean", "place_occupied"], required=True)
    parser.add_argument("--candidate-id", default="", help="candidate to ablate; default runs all scenario candidates")
    parser.add_argument("--control-mode", choices=["ideal", "physical"], default="physical")
    parser.add_argument("--universal-experience-lib", type=Path, required=True)
    parser.add_argument("--keyframe-dir", type=Path, default=None)
    parser.add_argument("--save", type=Path, required=True)
    return parser.parse_args()


def _vector(value: Any) -> list[float]:
    if not isinstance(value, list):
        return []
    out = []
    for item in value:
        try:
            out.append(float(item))
        except (TypeError, ValueError):
            return []
    return out


def _l1_delta(left: Any, right: Any) -> float:
    a = _vector(left)
    b = _vector(right)
    n = min(len(a), len(b), 3)
    return round(sum(abs(a[i] - b[i]) for i in range(n)), 6) if n else 0.0


def _result_object_start(result: Any) -> list[float]:
    direct = getattr(result, "object_start", None)
    if isinstance(direct, list):
        return _vector(direct)
    metrics = dict(result.metrics or {})
    start = metrics.get("object_start")
    if isinstance(start, list):
        return _vector(start)
    legacy = result.to_dict()
    value = legacy.get("object_start") if isinstance(legacy, dict) else None
    if not isinstance(value, list) and isinstance(legacy, dict):
        value = (legacy.get("execution_feedback") or {}).get("object_start")
    return _vector(value)


def _calibration_effect_from_result(result: Any) -> dict[str, Any]:
    metrics = dict(result.metrics or {})
    effect = metrics.get("sandbox_calibration")
    return dict(effect) if isinstance(effect, dict) else {}


def _evaluate_variant(
    *,
    scenario: str,
    condition: str,
    candidate_id: str,
    control_mode: str,
    variant: str,
    calibration: dict[str, Any] | None,
    keyframe_dir: Path | None,
) -> dict[str, Any]:
    rollout_calibration = calibration if variant == "sandbox_pose_and_score_calibration" else None
    scoring_calibration = calibration if variant in {"sandbox_score_calibration_only", "sandbox_pose_and_score_calibration"} else None
    with _galaxea_workdir():
        result = run_task_chain(
            scenario,
            condition,
            control_mode,
            candidate_id=candidate_id,
            keyframe_dir=keyframe_dir,
            sandbox_calibration=_calibration_dict(rollout_calibration),
        )
    entry = R1ProMujocoAdapter().normalize_episode(result)
    apply_critic(entry, thresholds=DEFAULT_CRITIC_THRESHOLDS)
    score = sandbox_score(entry, calibration=scoring_calibration)
    effect = _calibration_effect_from_result(result)
    return {
        "variant": variant,
        "candidate_id": candidate_id,
        "rollout_calibration_enabled": bool(rollout_calibration),
        "score_calibration_enabled": bool(scoring_calibration),
        "object_start": _result_object_start(result),
        "sandbox_calibration_effect": effect,
        "sandbox_score": score["sandbox_score"],
        "raw_sandbox_score": score["raw_sandbox_score"],
        "decision": score["decision"],
        "success": score["success"],
        "task_success": score["task_success"],
        "critic_status": score["critic_status"],
        "critic_risk_score": score["critic_risk_score"],
        "calibration_applied": score["calibration_applied"],
        "calibration_id": score["calibration_id"],
        "calibration_risk_penalty": score["calibration_risk_penalty"],
        "object_pose_bias": score["object_pose_bias"],
        "motion_critic": (result.metrics or {}).get("motion_critic", {}),
        "failed_skills": [item.get("skill") for item in result.skill_trace if not item.get("success")],
        "experience_id": entry.experience_id,
    }


def _candidate_ids(scenario: str, candidate_id: str) -> list[str]:
    candidates = candidates_for_scenario(scenario, include_risky=True)
    if candidate_id:
        known = {candidate.candidate_id for candidate in candidates}
        if candidate_id not in known:
            raise ValueError(f"Unknown candidate_id for {scenario}: {candidate_id}")
        return [candidate_id]
    return [candidate.candidate_id for candidate in candidates]


def _best_variant(variants: list[dict[str, Any]]) -> dict[str, Any]:
    return max(
        variants,
        key=lambda item: (
            float(item.get("sandbox_score") or 0.0),
            -float(item.get("critic_risk_score") or 0.0),
            str(item.get("variant") or ""),
        ),
    )


def _summarize_candidate(candidate_id: str, variants: dict[str, dict[str, Any]]) -> dict[str, Any]:
    no_cal = variants["sandbox_no_calibration"]
    score_only = variants["sandbox_score_calibration_only"]
    full = variants["sandbox_pose_and_score_calibration"]
    selected_without = _best_variant([no_cal])
    selected_with_full = _best_variant([full])
    return {
        "candidate_id": candidate_id,
        "object_start_delta_score_only_vs_no_calibration": _l1_delta(no_cal.get("object_start"), score_only.get("object_start")),
        "object_start_delta_full_vs_no_calibration": _l1_delta(no_cal.get("object_start"), full.get("object_start")),
        "sandbox_score_delta_score_only_vs_no_calibration": round(float(score_only["sandbox_score"]) - float(no_cal["sandbox_score"]), 4),
        "sandbox_score_delta_full_vs_no_calibration": round(float(full["sandbox_score"]) - float(no_cal["sandbox_score"]), 4),
        "raw_sandbox_score_delta_score_only_vs_no_calibration": round(float(score_only["raw_sandbox_score"]) - float(no_cal["raw_sandbox_score"]), 4),
        "raw_sandbox_score_delta_full_vs_no_calibration": round(float(full["raw_sandbox_score"]) - float(no_cal["raw_sandbox_score"]), 4),
        "critic_status_delta_score_only": f"{no_cal['critic_status']}->{score_only['critic_status']}",
        "critic_status_delta_full": f"{no_cal['critic_status']}->{full['critic_status']}",
        "critic_risk_delta_score_only": round(float(score_only["critic_risk_score"]) - float(no_cal["critic_risk_score"]), 4),
        "critic_risk_delta_full": round(float(full["critic_risk_score"]) - float(no_cal["critic_risk_score"]), 4),
        "calibration_risk_penalty_score_only": score_only["calibration_risk_penalty"],
        "calibration_risk_penalty_full": full["calibration_risk_penalty"],
        "selected_variant_without_calibration": selected_without["variant"],
        "selected_variant_with_full_calibration": selected_with_full["variant"],
        "selected_candidate_delta": selected_without["candidate_id"] != selected_with_full["candidate_id"],
        "nominal_object_start": no_cal.get("object_start", []),
        "calibrated_object_start": full.get("object_start", []),
        "object_pose_bias": full.get("object_pose_bias", []),
    }


def _summarize_report(candidate_summaries: list[dict[str, Any]]) -> dict[str, Any]:
    if not candidate_summaries:
        return {
            "candidate_count": 0,
            "object_start_delta_full_avg": 0.0,
            "sandbox_score_delta_score_only_avg": 0.0,
            "sandbox_score_delta_full_avg": 0.0,
            "selected_candidate_delta_count": 0,
        }
    return {
        "candidate_count": len(candidate_summaries),
        "object_start_delta_full_avg": round(sum(float(item["object_start_delta_full_vs_no_calibration"]) for item in candidate_summaries) / len(candidate_summaries), 6),
        "sandbox_score_delta_score_only_avg": round(sum(float(item["sandbox_score_delta_score_only_vs_no_calibration"]) for item in candidate_summaries) / len(candidate_summaries), 4),
        "sandbox_score_delta_full_avg": round(sum(float(item["sandbox_score_delta_full_vs_no_calibration"]) for item in candidate_summaries) / len(candidate_summaries), 4),
        "selected_candidate_delta_count": sum(int(bool(item["selected_candidate_delta"])) for item in candidate_summaries),
        "calibration_risk_penalty_avg": round(sum(float(item["calibration_risk_penalty_full"]) for item in candidate_summaries) / len(candidate_summaries), 4),
    }


def main() -> None:
    args = parse_args()
    library = ExperienceLibrary.load(args.universal_experience_lib)
    object_class = object_class_for_scenario(args.scenario)
    calibration = select_sandbox_calibration(
        library,
        scenario=args.scenario,
        condition=args.condition,
        object_class=object_class,
    )
    if not calibration:
        raise SystemExit(f"No sandbox calibration found for {args.scenario}/{args.condition}/{object_class}")

    candidates = _candidate_ids(args.scenario, args.candidate_id)
    reports = []
    summaries = []
    for candidate_id in candidates:
        variant_reports: dict[str, dict[str, Any]] = {}
        for variant in VARIANTS:
            keyframe_dir = args.keyframe_dir / candidate_id / variant if args.keyframe_dir is not None else None
            variant_reports[variant] = _evaluate_variant(
                scenario=args.scenario,
                condition=args.condition,
                candidate_id=candidate_id,
                control_mode=args.control_mode,
                variant=variant,
                calibration=calibration,
                keyframe_dir=keyframe_dir,
            )
        reports.append({
            "candidate_id": candidate_id,
            "variants": variant_reports,
            "summary": _summarize_candidate(candidate_id, variant_reports),
        })
        summaries.append(reports[-1]["summary"])

    report = {
        "schema_version": "sandbox_calibration_ablation_v1",
        "scenario": args.scenario,
        "condition": args.condition,
        "control_mode": args.control_mode,
        "experience_library": str(args.universal_experience_lib),
        "calibration": calibration,
        "calibration_risk": calibration_risk(calibration),
        "variants": VARIANTS,
        "summary": _summarize_report(summaries),
        "candidates": reports,
        "safe_paper_wording": "Gap-derived calibration changes sandbox initialization and contributes an explicit risk penalty during candidate evaluation.",
        "claim_boundary": "Do not claim dynamics, friction, or full contact-model calibration; this ablation covers object initial-state shift and score penalty.",
    }
    args.save.parent.mkdir(parents=True, exist_ok=True)
    args.save.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({
        "save": str(args.save),
        "scenario": args.scenario,
        "condition": args.condition,
        "candidate_count": len(reports),
        "calibration_id": calibration.get("calibration_id", ""),
        "object_pose_bias": calibration.get("object_pose_bias", []),
        "summary": report["summary"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
