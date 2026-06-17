"""Generate compact experience lessons with an LLM only.

This script does not use deterministic templates. If the configured LLM API
does not return valid JSON, it fails.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experience_core import JSON_ONLY_LINE, invoke_llm, parse_json_payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate concise LLM lessons from policy/sandbox evidence.")
    parser.add_argument("--policy-report", type=Path, required=True)
    parser.add_argument("--sandbox-report", type=Path, default=None)
    parser.add_argument("--save", type=Path, required=True)
    parser.add_argument("--provider", choices=["doubao", "openai"], default="doubao")
    parser.add_argument("--model", default="", help="Optional model name for the experience-system LLM provider")
    parser.add_argument("--max-candidates", type=int, default=6)
    parser.add_argument("--max-failure-risks", type=int, default=3)
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _compact_failure_risk(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "experience_id": item.get("experience_id", ""),
        "failure_type": item.get("failure_type", ""),
        "critic_status": item.get("critic_status", ""),
        "critic_risk": item.get("critic_risk"),
        "sim_real_gap_type": item.get("sim_real_gap_type", ""),
        "failed_action_overlap": item.get("failed_action_overlap"),
        "failure_similarity": item.get("failure_similarity"),
        "terminal_risk_score": item.get("terminal_risk_score"),
        "mitigations": (item.get("mitigation") or {}).get("mitigations", []),
    }


def _compact_candidate(candidate: dict[str, Any], *, max_failure_risks: int) -> dict[str, Any]:
    score = candidate.get("candidate_score") or candidate.get("memory", {}).get("candidate_score") or {}
    sandbox = candidate.get("sandbox") or {}
    fused = candidate.get("fused_score") or {}
    return {
        "candidate_id": candidate.get("candidate_id", ""),
        "description": candidate.get("description", ""),
        "executable": candidate.get("executable"),
        "candidate_steps": candidate.get("candidate_steps") or (candidate.get("memory") or {}).get("candidate_steps") or [],
        "memory_score": score.get("candidate_score", fused.get("memory_score")),
        "risk_score": score.get("risk_score", fused.get("memory_risk_score")),
        "terminal_risk_score": score.get("terminal_risk_score"),
        "failure_risk_penalty": score.get("failure_risk_penalty"),
        "top_failure_risks": [
            _compact_failure_risk(item)
            for item in (score.get("top_failure_risks") or [])[:max_failure_risks]
            if isinstance(item, dict)
        ],
        "sandbox_score": sandbox.get("sandbox_score", fused.get("sandbox_score")),
        "sandbox_decision": sandbox.get("decision", fused.get("sandbox_decision")),
        "critic_status": sandbox.get("critic_status"),
        "critic_flags": sandbox.get("critic_flags", [])[:5] if isinstance(sandbox.get("critic_flags"), list) else [],
        "motion_critic": sandbox.get("motion_critic", {}),
        "combined_score": fused.get("combined_score"),
        "fused_decision": fused.get("decision"),
    }


def _build_evidence(policy_report: dict[str, Any], sandbox_report: dict[str, Any] | None, args: argparse.Namespace) -> dict[str, Any]:
    candidates = policy_report.get("candidates") if isinstance(policy_report.get("candidates"), list) else []
    if sandbox_report and isinstance(sandbox_report.get("candidates"), list):
        by_id = {str(item.get("candidate_id", "")): item for item in candidates if isinstance(item, dict)}
        merged = []
        for item in sandbox_report["candidates"]:
            if not isinstance(item, dict):
                continue
            base = dict(by_id.get(str(item.get("candidate_id", "")), {}))
            base.update(item)
            merged.append(base)
        candidates = merged
    return {
        "scenario": policy_report.get("scenario") or (sandbox_report or {}).get("scenario"),
        "condition": policy_report.get("condition") or (sandbox_report or {}).get("condition"),
        "control_mode": policy_report.get("control_mode") or (sandbox_report or {}).get("control_mode"),
        "selected_before_sandbox": policy_report.get("selected_before_sandbox") or (sandbox_report or {}).get("selected_before_sandbox"),
        "selected_after_sandbox": policy_report.get("selected_after_sandbox") or (sandbox_report or {}).get("selected_after_sandbox"),
        "candidate_changed_by_sandbox": bool(policy_report.get("candidate_changed_by_sandbox") or (sandbox_report or {}).get("candidate_changed_by_sandbox")),
        "sandbox_summary": policy_report.get("sandbox_summary") or (sandbox_report or {}).get("sandbox_summary", {}),
        "candidates": [
            _compact_candidate(item, max_failure_risks=args.max_failure_risks)
            for item in candidates[: args.max_candidates]
            if isinstance(item, dict)
        ],
    }


def _lesson_prompt(evidence: dict[str, Any]) -> str:
    return f"""
You are writing compact robot anomaly-recovery lessons from structured evidence.

Use only facts, skill names, candidate ids, and evidence ids present in the input.
Do not invent robot capabilities, hidden observations, or new skills.
Do not output a deterministic rule template; synthesize concise lessons from the evidence.

Input evidence JSON:
{json.dumps(evidence, ensure_ascii=False, indent=2)}

Return only one JSON object with this schema:
{{
  "lessons": [
    {{
      "lesson": "one concise lesson, max 25 Chinese characters or 20 English words",
      "if": "trigger condition using scenario/condition/critic evidence",
      "then": "recommended strategy using only input skill names",
      "avoid": "risk pattern to avoid using only input skill names",
      "evidence_ids": ["ids copied exactly from input top_failure_risks"],
      "supporting_candidate_ids": ["candidate ids copied exactly from input"],
      "confidence": 0.0
    }}
  ]
}}

Constraints:
- Generate 1 to 5 lessons.
- Every evidence_id must appear in input top_failure_risks.
- Every skill name in then/avoid must appear in input candidate_steps.
- confidence must be a number from 0 to 1.
- Return JSON only, no Markdown.
"""


def _invoke_lessons(prompt: str, *, provider: str, model: str) -> dict[str, Any]:
    raw_text = invoke_llm(
        f"{prompt}\n\n{JSON_ONLY_LINE}",
        provider=provider,
        model=model,
        system_prompt="You return JSON only.",
    )
    payload = parse_json_payload(raw_text, prefer_array=False)
    if not isinstance(payload, dict):
        raise RuntimeError(f"LLM response was not a JSON object: {str(raw_text)[:500]}")
    lessons = payload.get("lessons")
    if not isinstance(lessons, list) or not lessons:
        raise RuntimeError(f"LLM response missing non-empty lessons list: {payload}")
    return payload


def _validate_lessons(payload: dict[str, Any], evidence: dict[str, Any]) -> None:
    evidence_ids = {
        str(risk.get("experience_id"))
        for candidate in evidence.get("candidates", [])
        for risk in candidate.get("top_failure_risks", [])
        if isinstance(risk, dict) and risk.get("experience_id")
    }
    candidate_ids = {str(candidate.get("candidate_id")) for candidate in evidence.get("candidates", []) if candidate.get("candidate_id")}
    for index, lesson in enumerate(payload.get("lessons") or []):
        if not isinstance(lesson, dict):
            raise RuntimeError(f"lesson[{index}] is not an object")
        for key in ("lesson", "if", "then", "avoid"):
            if not str(lesson.get(key) or "").strip():
                raise RuntimeError(f"lesson[{index}] missing {key}")
        for evidence_id in lesson.get("evidence_ids") or []:
            if str(evidence_id) not in evidence_ids:
                raise RuntimeError(f"lesson[{index}] uses unknown evidence_id: {evidence_id}")
        for candidate_id in lesson.get("supporting_candidate_ids") or []:
            if str(candidate_id) not in candidate_ids:
                raise RuntimeError(f"lesson[{index}] uses unknown candidate_id: {candidate_id}")
        confidence = lesson.get("confidence")
        if not isinstance(confidence, (int, float)) or not 0.0 <= float(confidence) <= 1.0:
            raise RuntimeError(f"lesson[{index}] confidence must be 0..1")


def main() -> None:
    args = parse_args()
    policy_report = _load_json(args.policy_report)
    sandbox_report = _load_json(args.sandbox_report) if args.sandbox_report else None
    evidence = _build_evidence(policy_report, sandbox_report, args)
    payload = _invoke_lessons(
        _lesson_prompt(evidence),
        provider=args.provider,
        model=args.model,
    )
    _validate_lessons(payload, evidence)

    output = {
        "schema_version": "llm_experience_lessons_v1",
        "generator": "generate_experience_lessons_llm.py",
        "policy_report": str(args.policy_report),
        "sandbox_report": str(args.sandbox_report) if args.sandbox_report else "",
        "llm_model": args.model,
        "llm_provider": args.provider,
        "evidence": evidence,
        "lessons": payload["lessons"],
    }
    args.save.parent.mkdir(parents=True, exist_ok=True)
    args.save.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"lesson_count": len(output["lessons"]), "save": str(args.save)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
