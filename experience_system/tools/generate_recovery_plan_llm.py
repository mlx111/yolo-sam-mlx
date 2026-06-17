"""Generate structured LLM recovery plans from stage-aware planner input."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experience_core import ExperienceLibrary, build_stage_planner_context, normalize_recovery_plan, recovery_plan_prompt, invoke_recovery_plan_llm
from source.run_r1pro_memory_policy_smoke import CandidatePlan, candidates_for_scenario
from experience_core import run_stage_retrieval


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate structured recovery plans with an LLM.")
    parser.add_argument("--scenario", choices=["G3"], required=True)
    parser.add_argument("--condition", choices=["clean", "place_occupied"], required=True)
    parser.add_argument("--candidate-id", default="", help="seed candidate id; defaults to scenario baseline")
    parser.add_argument("--universal-experience-lib", type=Path, required=True)
    parser.add_argument("--provider", choices=["doubao", "openai"], default="doubao")
    parser.add_argument("--model", default="")
    parser.add_argument("--stage-top-k", type=int, default=None)
    parser.add_argument("--save", type=Path, required=True)
    return parser.parse_args()


def _candidate_from_scenario(scenario: str, candidate_id: str) -> CandidatePlan:
    candidates = candidates_for_scenario(scenario, include_risky=True)
    if candidate_id:
        for candidate in candidates:
            if candidate.candidate_id == candidate_id:
                return candidate
    return candidates[0]


def main() -> None:
    args = parse_args()
    library = ExperienceLibrary.load(args.universal_experience_lib)
    candidate = _candidate_from_scenario(args.scenario, args.candidate_id)
    stage_report = run_stage_retrieval(
        library,
        scenario=args.scenario,
        condition=args.condition,
        object_class="sortable_object" if args.scenario == "G3" else "large_object",
        candidate_id=candidate.candidate_id,
        candidate_steps=list(candidate.steps),
        top_k=args.stage_top_k,
    )
    context = build_stage_planner_context(
        stage_report,
        scenario=args.scenario,
        condition=args.condition,
        candidate_id=candidate.candidate_id,
        candidate_steps=list(candidate.steps),
        candidate_description=candidate.description,
    )
    prompt = recovery_plan_prompt(
        scenario=args.scenario,
        condition=args.condition,
        planner_input=context.get("planner_input") or {},
        candidate=candidate,
        candidates=candidates_for_scenario(args.scenario, include_risky=True),
    )
    raw_plan = invoke_recovery_plan_llm(
        prompt,
        provider=args.provider,
        model=args.model,
    )
    plan = normalize_recovery_plan(
        raw_plan,
        scenario=args.scenario,
        condition=args.condition,
        candidate=candidate,
        candidates=candidates_for_scenario(args.scenario, include_risky=True),
        planner_input=context.get("planner_input") or {},
        provider=args.provider,
        model=args.model,
    )
    output = {
        "schema_version": "llm_recovery_plan_report_v1",
        "scenario": args.scenario,
        "condition": args.condition,
        "seed_candidate_id": candidate.candidate_id,
        "stage_planner_context": context,
        "recovery_plan": plan,
    }
    args.save.parent.mkdir(parents=True, exist_ok=True)
    args.save.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({
        "save": str(args.save),
        "plan_id": plan["plan_id"],
        "confidence": plan["confidence"],
        "step_count": len(plan["steps"]),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
