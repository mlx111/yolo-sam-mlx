"""Render stage-aware planner context for one or all candidate plans."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experience_core import ExperienceLibrary, build_stage_planner_context, run_stage_retrieval, summarize_stage_planner_contexts
from source.legacy_r1pro.run_r1pro_memory_policy_smoke import candidates_for_scenario, object_class_for_scenario


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render compact stage-aware planner context from retrieval evidence.")
    parser.add_argument("--input", type=Path, required=True, help="universal experience library JSON")
    parser.add_argument("--scenario", choices=["G3"], required=True)
    parser.add_argument("--condition", choices=["clean", "place_occupied"], required=True)
    parser.add_argument("--candidate-id", default="", help="render only this candidate; default renders all scenario candidates")
    parser.add_argument("--top-k", type=int, default=None, help="override stage retrieval top-k")
    parser.add_argument("--max-examples", type=int, default=3)
    parser.add_argument("--max-risks", type=int, default=4)
    parser.add_argument("--max-warnings", type=int, default=4)
    parser.add_argument("--max-writeback", type=int, default=3)
    parser.add_argument("--save", type=Path, default=None)
    parser.add_argument("--save-text", type=Path, default=None)
    return parser.parse_args()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_text(path: Path, contexts: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n\n---\n\n".join(str(context.get("prompt_text") or "") for context in contexts), encoding="utf-8")


def main() -> None:
    args = parse_args()
    library = ExperienceLibrary.load(args.input)
    object_class = object_class_for_scenario(args.scenario)
    candidates = candidates_for_scenario(args.scenario)
    if args.candidate_id:
        candidates = [candidate for candidate in candidates if candidate.candidate_id == args.candidate_id]
        if not candidates:
            raise SystemExit(f"Unknown candidate_id for {args.scenario}: {args.candidate_id}")

    contexts = []
    stage_reports = []
    for candidate in candidates:
        stage_report = run_stage_retrieval(
            library,
            scenario=args.scenario,
            condition=args.condition,
            object_class=object_class,
            candidate_id=candidate.candidate_id,
            candidate_steps=list(candidate.steps),
            top_k=args.top_k,
        )
        context = build_stage_planner_context(
            stage_report,
            scenario=args.scenario,
            condition=args.condition,
            candidate_id=candidate.candidate_id,
            candidate_steps=list(candidate.steps),
            candidate_description=candidate.description,
            max_examples=args.max_examples,
            max_risks=args.max_risks,
            max_warnings=args.max_warnings,
            max_writeback=args.max_writeback,
        )
        contexts.append(context)
        stage_reports.append(stage_report)

    report = {
        "schema_version": "stage_planner_context_report_v1",
        "experience_library": str(args.input),
        "scenario": args.scenario,
        "condition": args.condition,
        "candidate_count": len(contexts),
        "summary": summarize_stage_planner_contexts(contexts),
        "contexts": contexts,
        "stage_reports": stage_reports,
    }
    if args.save is not None:
        _write_json(args.save, report)
    if args.save_text is not None:
        _write_text(args.save_text, contexts)
    print(json.dumps({
        "scenario": args.scenario,
        "condition": args.condition,
        "candidate_count": len(contexts),
        "summary": report["summary"],
        "save": str(args.save) if args.save else "",
        "save_text": str(args.save_text) if args.save_text else "",
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
