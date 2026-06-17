"""Ablate visual keyframe retrieval inside candidate memory ranking."""

from __future__ import annotations

import argparse
import csv
import io
import json
import sys
from pathlib import Path
from statistics import mean
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experience_core import ExperienceLibrary, VisualRetrievalIndex, load_policy_risk_calibration
from source.run_r1pro_memory_policy_smoke import (
    candidates_for_scenario,
    evaluate_candidate,
    object_class_for_scenario,
    selection_rank,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare memory ranking with and without visual keyframe retrieval.")
    parser.add_argument("--scenario", choices=["G3"], required=True)
    parser.add_argument("--condition", choices=["clean", "place_occupied"], required=True)
    parser.add_argument("--universal-experience-lib", type=Path, required=True)
    parser.add_argument("--visual-index-dir", type=Path, required=True)
    parser.add_argument("--query-image", type=Path, action="append", default=[], help="optional query image; defaults to a matching keyframe")
    parser.add_argument("--policy-calibration", type=Path, default=None)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--visual-top-k", type=int, default=10)
    parser.add_argument("--visual-weight", type=float, default=0.12)
    parser.add_argument("--include-risky-candidates", action="store_true")
    parser.add_argument("--save", type=Path, required=True)
    parser.add_argument("--save-csv", type=Path, default=None)
    return parser.parse_args()


def _mean(values: list[float]) -> float:
    return round(mean(values), 4) if values else 0.0


def _candidate_score(report: dict[str, Any]) -> float:
    return float((report.get("candidate_score") or {}).get("candidate_score") or 0.0)


def _risk_score(report: dict[str, Any]) -> float:
    return float((report.get("candidate_score") or {}).get("risk_score") or 0.0)


def _decision(report: dict[str, Any]) -> str:
    return str((report.get("candidate_score") or {}).get("decision") or "")


def _match_ids(report: dict[str, Any]) -> list[str]:
    matches = ((report.get("retrieval") or {}).get("matches") or [])
    return [str(item.get("experience_id") or "") for item in matches if isinstance(item, dict) and item.get("experience_id")]


def _top_visual_ids(visual_scores: dict[str, float], *, top_k: int) -> list[str]:
    return [
        experience_id
        for experience_id, _ in sorted(visual_scores.items(), key=lambda item: (-float(item[1]), item[0]))[:top_k]
    ]


def _overlap_ratio(left: list[str], right: list[str]) -> float:
    a = set(left)
    b = set(right)
    union = a | b
    return round(len(a & b) / len(union), 4) if union else 1.0


def _entry_matches(entry: Any, *, scenario: str, condition: str) -> bool:
    return entry.scenario_id == scenario and entry.condition_id == condition


def select_default_query_images(library: ExperienceLibrary, *, scenario: str, condition: str) -> list[Path]:
    candidates = [entry for entry in library.entries if _entry_matches(entry, scenario=scenario, condition=condition) and entry.keyframes]
    if not candidates:
        return []
    candidates = sorted(
        candidates,
        key=lambda entry: (
            0 if entry.source == "simulation" else 1,
            0 if bool(entry.result.get("success", False)) else 1,
            entry.experience_id,
        ),
    )
    preferred_stages = ["before_task", "before_place", "after_grasp", "after_lift", "after_place"]
    frames = candidates[0].keyframes
    for stage in preferred_stages:
        for frame in frames:
            path = Path(str(frame.get("image_path") or ""))
            if str(frame.get("stage") or "") == stage and path.exists():
                return [path]
    for frame in frames:
        path = Path(str(frame.get("image_path") or ""))
        if path.exists():
            return [path]
    return []


def load_visual_scores(index_dir: Path, query_images: list[Path], *, top_k: int) -> dict[str, float]:
    if not query_images:
        return {}
    index = VisualRetrievalIndex()
    index.load(index_dir)
    paths = [str(path.resolve()) for path in query_images if path.exists()]
    return {experience_id: score for experience_id, score in index.search(paths, top_k=top_k)}


def evaluate_variant(
    library: ExperienceLibrary,
    *,
    scenario: str,
    condition: str,
    object_class: str,
    top_k: int,
    policy_calibration: dict[str, Any] | None,
    visual_scores: dict[str, float],
    visual_weight: float,
    include_risky: bool,
) -> dict[str, Any]:
    reports = [
        evaluate_candidate(
            library,
            candidate,
            scenario=scenario,
            condition=condition,
            object_class=object_class,
            top_k=top_k,
            risk_aware=True,
            policy_calibration=policy_calibration,
            visual_scores=visual_scores,
            visual_weight=visual_weight,
        )
        for candidate in candidates_for_scenario(scenario, include_risky=include_risky)
    ]
    ranked = sorted(reports, key=selection_rank, reverse=True)
    selected = ranked[0] if ranked else None
    return {
        "candidate_count": len(reports),
        "selected_candidate_id": str(selected.get("candidate_id") or "") if selected else "",
        "selected_candidate_score": round(_candidate_score(selected), 4) if selected else 0.0,
        "selected_risk_score": round(_risk_score(selected), 4) if selected else 0.0,
        "candidates": ranked,
    }


def _compact_candidate_pair(no_visual: dict[str, Any], with_visual: dict[str, Any]) -> dict[str, Any]:
    left_ids = _match_ids(no_visual)
    right_ids = _match_ids(with_visual)
    score_delta = _candidate_score(with_visual) - _candidate_score(no_visual)
    risk_delta = _risk_score(with_visual) - _risk_score(no_visual)
    return {
        "candidate_id": no_visual.get("candidate_id", with_visual.get("candidate_id", "")),
        "description": no_visual.get("description", with_visual.get("description", "")),
        "score_without_visual": round(_candidate_score(no_visual), 4),
        "score_with_visual": round(_candidate_score(with_visual), 4),
        "candidate_score_delta": round(score_delta, 4),
        "risk_without_visual": round(_risk_score(no_visual), 4),
        "risk_with_visual": round(_risk_score(with_visual), 4),
        "risk_score_delta": round(risk_delta, 4),
        "decision_without_visual": _decision(no_visual),
        "decision_with_visual": _decision(with_visual),
        "retrieval_without_visual": left_ids,
        "retrieval_with_visual": right_ids,
        "retrieval_changed": left_ids != right_ids,
        "top_match_overlap": _overlap_ratio(left_ids, right_ids),
    }


def compare_variants(no_visual: dict[str, Any], with_visual: dict[str, Any], visual_scores: dict[str, float], *, visual_top_k: int) -> dict[str, Any]:
    no_by_id = {str(item.get("candidate_id") or ""): item for item in no_visual.get("candidates") or []}
    with_by_id = {str(item.get("candidate_id") or ""): item for item in with_visual.get("candidates") or []}
    pairs = []
    for candidate_id in sorted(set(no_by_id) & set(with_by_id)):
        pair = _compact_candidate_pair(no_by_id[candidate_id], with_by_id[candidate_id])
        pair["visual_top_match_overlap_with_retrieval"] = _overlap_ratio(
            _top_visual_ids(visual_scores, top_k=visual_top_k),
            pair["retrieval_with_visual"],
        )
        pair["visual_top_match_ids_in_retrieval"] = [
            item for item in pair["retrieval_with_visual"]
            if item in set(_top_visual_ids(visual_scores, top_k=visual_top_k))
        ]
        pairs.append(pair)
    score_deltas = [float(item["candidate_score_delta"]) for item in pairs]
    retrieval_changed = [item for item in pairs if item["retrieval_changed"]]
    selected_before = str(no_visual.get("selected_candidate_id") or "")
    selected_after = str(with_visual.get("selected_candidate_id") or "")
    return {
        "selected_candidate_before_visual": selected_before,
        "selected_candidate_after_visual": selected_after,
        "selected_candidate_change": selected_before != selected_after,
        "visual_score_count": len(visual_scores),
        "visual_top_match_ids": _top_visual_ids(visual_scores, top_k=visual_top_k),
        "retrieval_changed_candidate_count": len(retrieval_changed),
        "retrieval_changed_rate": round(len(retrieval_changed) / len(pairs), 4) if pairs else 0.0,
        "candidate_score_delta_avg": _mean(score_deltas),
        "candidate_score_delta_max": round(max(score_deltas), 4) if score_deltas else 0.0,
        "top_match_overlap_avg": _mean([float(item["top_match_overlap"]) for item in pairs]),
        "candidate_comparisons": pairs,
    }


def _csv_text(comparison: dict[str, Any]) -> str:
    output = io.StringIO()
    writer = csv.DictWriter(
        output,
        fieldnames=[
            "candidate_id",
            "score_without_visual",
            "score_with_visual",
            "candidate_score_delta",
            "risk_without_visual",
            "risk_with_visual",
            "risk_score_delta",
            "decision_without_visual",
            "decision_with_visual",
            "retrieval_changed",
            "top_match_overlap",
        ],
    )
    writer.writeheader()
    for item in comparison.get("candidate_comparisons") or []:
        writer.writerow({key: item.get(key, "") for key in writer.fieldnames})
    return output.getvalue()


def main() -> None:
    args = parse_args()
    library = ExperienceLibrary.load(args.universal_experience_lib)
    policy_calibration = load_policy_risk_calibration(args.policy_calibration) if args.policy_calibration else None
    object_class = object_class_for_scenario(args.scenario)
    query_images = list(args.query_image or []) or select_default_query_images(library, scenario=args.scenario, condition=args.condition)
    visual_scores = load_visual_scores(args.visual_index_dir, query_images, top_k=args.visual_top_k)

    no_visual = evaluate_variant(
        library,
        scenario=args.scenario,
        condition=args.condition,
        object_class=object_class,
        top_k=args.top_k,
        policy_calibration=policy_calibration,
        visual_scores={},
        visual_weight=args.visual_weight,
        include_risky=args.include_risky_candidates,
    )
    with_visual = evaluate_variant(
        library,
        scenario=args.scenario,
        condition=args.condition,
        object_class=object_class,
        top_k=args.top_k,
        policy_calibration=policy_calibration,
        visual_scores=visual_scores,
        visual_weight=args.visual_weight,
        include_risky=args.include_risky_candidates,
    )
    comparison = compare_variants(no_visual, with_visual, visual_scores, visual_top_k=args.visual_top_k)
    report = {
        "schema_version": "visual_retrieval_ablation_v1",
        "scenario": args.scenario,
        "condition": args.condition,
        "experience_library": str(args.universal_experience_lib),
        "visual_index_dir": str(args.visual_index_dir),
        "query_images": [str(path) for path in query_images],
        "visual_weight": args.visual_weight,
        "visual_top_k": args.visual_top_k,
        "top_k": args.top_k,
        "visual_scores": visual_scores,
        "without_visual": no_visual,
        "with_visual": with_visual,
        "comparison": comparison,
        "safe_paper_wording": "Visual keyframes are indexed and used as an auxiliary retrieval signal during candidate ranking.",
        "claim_boundary": "Do not claim multimodal semantic reasoning unless visual evidence changes or explains semantic decisions.",
    }
    args.save.parent.mkdir(parents=True, exist_ok=True)
    args.save.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    if args.save_csv is not None:
        args.save_csv.parent.mkdir(parents=True, exist_ok=True)
        args.save_csv.write_text(_csv_text(comparison), encoding="utf-8")
    print(json.dumps({
        "save": str(args.save),
        "save_csv": str(args.save_csv) if args.save_csv else "",
        "query_images": [str(path) for path in query_images],
        "selected_candidate_before_visual": comparison["selected_candidate_before_visual"],
        "selected_candidate_after_visual": comparison["selected_candidate_after_visual"],
        "selected_candidate_change": comparison["selected_candidate_change"],
        "visual_score_count": comparison["visual_score_count"],
        "retrieval_changed_rate": comparison["retrieval_changed_rate"],
        "candidate_score_delta_avg": comparison["candidate_score_delta_avg"],
        "top_match_overlap_avg": comparison["top_match_overlap_avg"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
