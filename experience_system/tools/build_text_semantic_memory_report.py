"""Build a text-semantic memory report for the universal experience library."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experience_core import ExperienceEntry, TextSemanticRetrievalIndex, semantic_query_text, semantic_summary
from experience_core.text_semantic_retrieval import tokens


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a text-semantic memory coverage and retrieval report.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("results/memory/galaxea_field_atomic_experience_library.json"),
        help="Universal experience library JSON.",
    )
    parser.add_argument("--save-json", type=Path, required=True)
    parser.add_argument("--save-md", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--backend",
        choices=["auto", "faiss", "token_overlap"],
        default="auto",
        help="Semantic retrieval backend. auto uses FAISS TF-IDF when dependencies are available.",
    )
    return parser.parse_args()


def _load_entries(path: Path) -> list[ExperienceEntry]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw_entries = payload.get("entries", []) if isinstance(payload, dict) else payload
    return [ExperienceEntry(**entry) for entry in raw_entries if isinstance(entry, dict)]


def _failure_type(entry: ExperienceEntry) -> str:
    return str(entry.failure_taxonomy.get("standard_failure_type") or entry.failure_taxonomy.get("failure_type") or "")


def _gap_type(entry: ExperienceEntry) -> str:
    return str(entry.sim_real_gap.outcome_gap.get("type") or "")


def _memory_role(entry: ExperienceEntry) -> str:
    return str(entry.memory_tags.get("memory_role") or "")


def _build_queries(entries: list[ExperienceEntry]) -> list[ExperienceEntry]:
    selected: list[ExperienceEntry] = []
    seen: set[tuple[str, str, str]] = set()
    for entry in entries:
        key = (entry.scenario_id, entry.condition_id, _failure_type(entry))
        if key in seen:
            continue
        seen.add(key)
        selected.append(entry)
    return selected


def _query_text_for_entry(entry: ExperienceEntry) -> str:
    return semantic_query_text(
        scenario=entry.scenario_id,
        condition=entry.condition_id,
        object_class=entry.object_state.object_class,
        task_stage=str(entry.task.get("stage") or ""),
        failure_type=_failure_type(entry),
        critic_status=entry.critic_result.overall_status,
        memory_role=_memory_role(entry),
    )


def _coverage(entries: list[ExperienceEntry], summaries: dict[str, str]) -> dict[str, Any]:
    token_lists = {experience_id: tokens(text) for experience_id, text in summaries.items()}
    lengths = [len(items) for items in token_lists.values()]
    return {
        "semantic_summary_count": len(summaries),
        "semantic_summary_nonempty_count": sum(1 for text in summaries.values() if text.strip()),
        "semantic_summary_nonempty_rate": round(
            sum(1 for text in summaries.values() if text.strip()) / len(entries), 4
        ) if entries else 0.0,
        "avg_token_count": round(sum(lengths) / len(lengths), 4) if lengths else 0.0,
        "max_token_count": max(lengths) if lengths else 0,
        "min_token_count": min(lengths) if lengths else 0,
        "source_distribution": dict(Counter(entry.source for entry in entries)),
        "scenario_distribution": dict(Counter(entry.scenario_id for entry in entries)),
        "condition_distribution": dict(Counter(entry.condition_id for entry in entries)),
        "memory_role_distribution": dict(Counter(_memory_role(entry) for entry in entries)),
        "failure_type_distribution": dict(Counter(_failure_type(entry) for entry in entries)),
        "gap_type_distribution": dict(Counter(_gap_type(entry) for entry in entries)),
        "query_token_coverage": {
            "scenario": sum(1 for entry in entries if entry.scenario_id),
            "condition": sum(1 for entry in entries if entry.condition_id),
            "task": sum(1 for entry in entries if entry.task.get("name")),
            "anomaly": sum(1 for entry in entries if entry.anomaly.get("type")),
            "failure": sum(1 for entry in entries if _failure_type(entry)),
            "critic": sum(1 for entry in entries if entry.critic_result.overall_status),
            "gap": sum(1 for entry in entries if _gap_type(entry)),
            "plan": sum(1 for entry in entries if entry.retrieval_key.get("plan_signature")),
        },
        "token_lists": token_lists,
    }


def build_report(entries: list[ExperienceEntry], *, input_path: Path, top_k: int, backend: str = "auto") -> dict[str, Any]:
    index = TextSemanticRetrievalIndex(entries, backend=backend)
    entries_by_id = {entry.experience_id: entry for entry in entries}
    summaries = {entry.experience_id: semantic_summary(entry) for entry in entries}
    coverage = _coverage(entries, summaries)
    token_lists = coverage.pop("token_lists")
    retrieval_reports = []
    sem_hits = 0
    total_pairs = 0

    for query_entry in _build_queries(entries):
        query_text = _query_text_for_entry(query_entry)
        query_tokens = tokens(query_text)
        if not query_tokens:
            continue
        scores = index.search_scores(query_text, top_k=min(max(int(top_k) + 1, 1), len(entries)))
        scores.pop(query_entry.experience_id, None)
        top_scores = dict(list(scores.items())[: max(int(top_k), 0)])
        if top_scores:
            sem_hits += 1
        total_pairs += 1
        retrieval_reports.append({
            "query_experience_id": query_entry.experience_id,
            "query_text": query_text,
            "query_tokens": len(query_tokens),
            "top_matches": [
                {
                    "experience_id": experience_id,
                    "score": score,
                    "scenario_id": entries_by_id[experience_id].scenario_id,
                    "condition_id": entries_by_id[experience_id].condition_id,
                    "failure_type": _failure_type(entries_by_id[experience_id]),
                    "memory_role": _memory_role(entries_by_id[experience_id]),
                }
                for experience_id, score in top_scores.items()
                if experience_id in entries_by_id
            ],
        })

    cross_scenario_match_count = 0
    cross_condition_match_count = 0
    same_scenario_match_count = 0
    for report in retrieval_reports:
        query_entry = entries_by_id[report["query_experience_id"]]
        for match in report["top_matches"]:
            if match["scenario_id"] and match["scenario_id"] != query_entry.scenario_id:
                cross_scenario_match_count += 1
            if match["condition_id"] and match["condition_id"] != query_entry.condition_id:
                cross_condition_match_count += 1
            if match["scenario_id"] == query_entry.scenario_id:
                same_scenario_match_count += 1

    return {
        "schema_version": "text_semantic_memory_report_v1",
        "input": str(input_path),
        "entry_count": len(entries),
        "retrieval_backend": index.backend,
        "backend_fallback_reason": index.fallback_reason,
        "faiss_index": index.faiss_metadata,
        "summary": {
            **coverage,
            "semantic_signal_rate": round(sem_hits / total_pairs, 4) if total_pairs else 0.0,
            "query_count": total_pairs,
            "same_scenario_topk_match_count": same_scenario_match_count,
            "cross_scenario_topk_match_count": cross_scenario_match_count,
            "cross_condition_topk_match_count": cross_condition_match_count,
        },
        "semantic_summaries": [
            {
                "experience_id": entry.experience_id,
                "summary": summaries[entry.experience_id],
                "token_count": len(token_lists[entry.experience_id]),
                "source": entry.source,
                "scenario_id": entry.scenario_id,
                "condition_id": entry.condition_id,
                "failure_type": _failure_type(entry),
                "memory_role": _memory_role(entry),
            }
            for entry in entries
        ],
        "retrieval_reports": retrieval_reports,
        "paper_wording": {
            "safe_claim": (
                "The system constructs explicit text-semantic summaries from scenario, condition, task, anomaly, "
                "failure taxonomy, critic, gap, and retrieval-key fields, and uses TF-IDF vectors indexed by FAISS "
                "as an auxiliary semantic retrieval signal."
            ),
            "avoid_claim": (
                "Do not claim this is a learned language encoder or neural embedding benchmark; the report measures lightweight TF-IDF + FAISS semantic retrieval only."
            ),
        },
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Text Semantic Memory Report",
        "",
        "This report summarizes explicit text-semantic evidence extracted from the universal experience library.",
        "",
        "## Summary",
        "",
        f"- Input: `{report['input']}`",
        f"- Retrieval backend: `{report['retrieval_backend']}`",
        f"- FAISS index: `{json.dumps(report.get('faiss_index') or {}, ensure_ascii=False, sort_keys=True)}`",
        f"- Entry count: {report['entry_count']}",
        f"- Nonempty semantic summaries: {report['summary']['semantic_summary_nonempty_count']} "
        f"({report['summary']['semantic_summary_nonempty_rate']})",
        f"- Average token count: {report['summary']['avg_token_count']}",
        f"- Semantic signal rate: {report['summary']['semantic_signal_rate']}",
        f"- Query count: {report['summary']['query_count']}",
        f"- Same-scenario top-k matches: {report['summary']['same_scenario_topk_match_count']}",
        f"- Cross-scenario top-k matches: {report['summary']['cross_scenario_topk_match_count']}",
        f"- Cross-condition top-k matches: {report['summary']['cross_condition_topk_match_count']}",
        "",
        "## Field Coverage",
        "",
        f"- Scenario tokens: {report['summary']['query_token_coverage']['scenario']}",
        f"- Condition tokens: {report['summary']['query_token_coverage']['condition']}",
        f"- Task tokens: {report['summary']['query_token_coverage']['task']}",
        f"- Anomaly tokens: {report['summary']['query_token_coverage']['anomaly']}",
        f"- Failure tokens: {report['summary']['query_token_coverage']['failure']}",
        f"- Critic tokens: {report['summary']['query_token_coverage']['critic']}",
        f"- Gap tokens: {report['summary']['query_token_coverage']['gap']}",
        f"- Plan tokens: {report['summary']['query_token_coverage']['plan']}",
        "",
        "## Paper Wording Boundary",
        "",
        f"- Safe claim: {report['paper_wording']['safe_claim']}",
        f"- Avoid claim: {report['paper_wording']['avoid_claim']}",
        "",
        "## Retrieval Samples",
        "",
    ]
    for item in report["retrieval_reports"][:8]:
        lines.append(f"### {item['query_experience_id']}")
        lines.append("")
        lines.append(f"- Query: `{item['query_text']}`")
        lines.append(f"- Top matches: {len(item['top_matches'])}")
        for match in item["top_matches"][:5]:
            lines.append(
                f"- {match['experience_id']} score={match['score']} "
                f"({match['scenario_id']}/{match['condition_id']}/{match['failure_type']})"
            )
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    entries = _load_entries(args.input)
    report = build_report(entries, input_path=args.input, top_k=args.top_k, backend=args.backend)
    args.save_json.parent.mkdir(parents=True, exist_ok=True)
    args.save_md.parent.mkdir(parents=True, exist_ok=True)
    args.save_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    args.save_md.write_text(render_markdown(report), encoding="utf-8")
    print(
        json.dumps(
            {
                "input": str(args.input),
                "save_json": str(args.save_json),
                "save_md": str(args.save_md),
                "entry_count": report["entry_count"],
                "retrieval_backend": report["retrieval_backend"],
                "semantic_signal_rate": report["summary"]["semantic_signal_rate"],
                "query_count": report["summary"]["query_count"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
