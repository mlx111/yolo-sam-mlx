from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from statistics import mean
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
EXPERIENCE_ROOT = ROOT.parent / "experience_system"
for path in (ROOT, EXPERIENCE_ROOT):
    text = str(path)
    if text not in sys.path:
        sys.path.insert(0, text)

from experience_core import (  # noqa: E402
    ExperienceLibrary,
    GALAXEA_R1PRO_TORSO_NAMESPACE,
    field_atomic_action,
    field_atomic_success,
    is_field_atomic_entry,
    query_field_atomic_experience_matches,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Galaxea field-atomic retrieval quality offline.")
    parser.add_argument("--library", type=Path, action="append", required=True)
    parser.add_argument("--save", type=Path, required=True)
    parser.add_argument("--save-csv", type=Path, default=None)
    parser.add_argument("--scenario-id", default="")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--query-limit", type=int, default=50)
    parser.add_argument("--include-success-queries", action="store_true")
    parser.add_argument("--gap-aware", action="store_true")
    parser.add_argument("--risk-aware", action="store_true")
    parser.add_argument("--diversity-lambda", type=float, default=0.25)
    return parser.parse_args()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _entry_anomaly_state(entry: Any) -> dict[str, Any]:
    key = entry.retrieval_key if isinstance(entry.retrieval_key, dict) else {}
    state = key.get("anomaly_state")
    if isinstance(state, dict):
        return state
    spatial = entry.spatial_state if isinstance(entry.spatial_state, dict) else {}
    state = spatial.get("anomaly_state")
    if isinstance(state, dict):
        return state
    metadata = entry.metadata if isinstance(entry.metadata, dict) else {}
    state = metadata.get("anomaly_state")
    return state if isinstance(state, dict) else {}


def _entry_text(entry: Any) -> str:
    metadata = entry.metadata if isinstance(entry.metadata, dict) else {}
    feedback = entry.execution_feedback if isinstance(entry.execution_feedback, dict) else {}
    taxonomy = entry.failure_taxonomy if isinstance(entry.failure_taxonomy, dict) else {}
    parts = [
        metadata.get("text_summary", ""),
        feedback.get("memory_lesson", ""),
        taxonomy.get("failure_reason", ""),
        taxonomy.get("critic_root_cause", ""),
        taxonomy.get("corrective_direction", ""),
    ]
    return " ".join(str(item) for item in parts if item)


def _entry_target_class(entry: Any) -> str:
    key = entry.retrieval_key if isinstance(entry.retrieval_key, dict) else {}
    if key.get("target_class"):
        return str(key.get("target_class"))
    state = _entry_anomaly_state(entry)
    if state.get("target_class"):
        return str(state.get("target_class"))
    return str(entry.object_state.object_class or entry.object_state.target_object or "")


def _entry_failure_stage(entry: Any) -> str:
    key = entry.retrieval_key if isinstance(entry.retrieval_key, dict) else {}
    return str(key.get("failure_stage") or entry.failure_taxonomy.get("failure_stage") or field_atomic_action(entry) or "")


def _entry_failure_type(entry: Any) -> str:
    key = entry.retrieval_key if isinstance(entry.retrieval_key, dict) else {}
    return str(key.get("failure_type") or entry.failure_taxonomy.get("failure_type") or "")


def _query_entries(entries: list[Any], *, scenario_id: str, include_success: bool, limit: int) -> list[Any]:
    rows = []
    for entry in entries:
        if not is_field_atomic_entry(entry):
            continue
        if entry.skill_namespace != GALAXEA_R1PRO_TORSO_NAMESPACE:
            continue
        if scenario_id and entry.scenario_id != scenario_id:
            continue
        if not include_success and field_atomic_success(entry):
            continue
        if str(entry.memory_tags.get("memory_type") or "") != "field_atomic_episode":
            continue
        rows.append(entry)
    return rows[: max(0, int(limit))]


def _is_relevant(query: Any, candidate: Any) -> bool:
    if query.experience_id == candidate.experience_id:
        return False
    same_stage = _entry_failure_stage(query) and _entry_failure_stage(query) == _entry_failure_stage(candidate)
    same_type = _entry_failure_type(query) and _entry_failure_type(query) == _entry_failure_type(candidate)
    same_target = _entry_target_class(query) and _entry_target_class(query) == _entry_target_class(candidate)
    same_success = field_atomic_success(query) == field_atomic_success(candidate)
    if not field_atomic_success(query):
        return bool((same_stage or same_type) and (same_target or same_success))
    return bool(same_target and same_success)


def _evaluate_query(entry: Any, entries: list[Any], args: argparse.Namespace) -> dict[str, Any]:
    anomaly_state = _entry_anomaly_state(entry)
    matches = query_field_atomic_experience_matches(
        entries,
        scenario_id=entry.scenario_id,
        condition_id="",
        skill_namespace=GALAXEA_R1PRO_TORSO_NAMESPACE,
        available_actions=[],
        retrieval_key=entry.retrieval_key if isinstance(entry.retrieval_key, dict) else {},
        anomaly_state=anomaly_state,
        task_stage=_entry_failure_stage(entry),
        text_summary=_entry_text(entry),
        include_failed=True,
        gap_aware=bool(args.gap_aware),
        risk_aware=bool(args.risk_aware),
        diversity_lambda=float(args.diversity_lambda),
        limit=max(int(args.top_k) + 1, int(args.top_k)),
    )
    filtered = [(candidate, score, expl) for candidate, score, expl in matches if candidate.experience_id != entry.experience_id]
    top = filtered[: max(0, int(args.top_k))]
    relevant_flags = [_is_relevant(entry, candidate) for candidate, _, _ in top]
    first_relevant_rank = 0
    for index, flag in enumerate(relevant_flags, 1):
        if flag:
            first_relevant_rank = index
            break
    return {
        "query_experience_id": entry.experience_id,
        "scenario_id": entry.scenario_id,
        "query_success": field_atomic_success(entry),
        "query_action": field_atomic_action(entry),
        "query_failure_stage": _entry_failure_stage(entry),
        "query_failure_type": _entry_failure_type(entry),
        "query_target_class": _entry_target_class(entry),
        "top_k": int(args.top_k),
        "hit_at_1": bool(relevant_flags[:1] and relevant_flags[0]),
        "hit_at_k": any(relevant_flags),
        "precision_at_k": round(sum(1 for item in relevant_flags if item) / max(len(top), 1), 4),
        "first_relevant_rank": first_relevant_rank,
        "matches": [
            {
                "rank": index,
                "experience_id": candidate.experience_id,
                "score": round(float(score), 4),
                "relevant": relevant,
                "success": field_atomic_success(candidate),
                "action": field_atomic_action(candidate),
                "failure_stage": _entry_failure_stage(candidate),
                "failure_type": _entry_failure_type(candidate),
                "target_class": _entry_target_class(candidate),
                "explanation": explanation,
            }
            for index, ((candidate, score, explanation), relevant) in enumerate(zip(top, relevant_flags), 1)
        ],
    }


def _avg(values: list[float]) -> float:
    return round(mean(values), 4) if values else 0.0


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "query_count": len(rows),
        "hit_at_1": _avg([1.0 if row["hit_at_1"] else 0.0 for row in rows]),
        "hit_at_k": _avg([1.0 if row["hit_at_k"] else 0.0 for row in rows]),
        "precision_at_k": _avg([float(row["precision_at_k"]) for row in rows]),
        "mean_first_relevant_rank": _avg([float(row["first_relevant_rank"]) for row in rows if int(row["first_relevant_rank"]) > 0]),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "query_experience_id",
                "scenario_id",
                "query_success",
                "query_action",
                "query_failure_stage",
                "query_failure_type",
                "query_target_class",
                "hit_at_1",
                "hit_at_k",
                "precision_at_k",
                "first_relevant_rank",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in writer.fieldnames})


def main() -> None:
    args = parse_args()
    library = ExperienceLibrary()
    loaded_libraries = []
    for path in args.library:
        current = ExperienceLibrary.load(path)
        loaded_libraries.append({"path": str(path), "entry_count": len(current.entries)})
        library.entries.extend(current.entries)
    queries = _query_entries(
        library.entries,
        scenario_id=str(args.scenario_id or ""),
        include_success=bool(args.include_success_queries),
        limit=int(args.query_limit),
    )
    rows = [_evaluate_query(entry, library.entries, args) for entry in queries]
    report = {
        "schema_version": "field_atomic_retrieval_quality_v1",
        "libraries": loaded_libraries,
        "scenario_id": str(args.scenario_id or ""),
        "top_k": int(args.top_k),
        "gap_aware": bool(args.gap_aware),
        "risk_aware": bool(args.risk_aware),
        "diversity_lambda": float(args.diversity_lambda),
        "summary": _summary(rows),
        "queries": rows,
    }
    _write_json(args.save, report)
    if args.save_csv is not None:
        _write_csv(args.save_csv, rows)
    print(json.dumps({"query_count": len(rows), **report["summary"], "save": str(args.save)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
