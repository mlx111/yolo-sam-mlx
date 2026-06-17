"""Analyze LLM-generated experience lessons for grounding and conflicts."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experience_core import normalize_lesson


TEMPLATE_LIKE_PATTERNS = [
    re.compile(r"\bif\s+.*\bthen\b", re.IGNORECASE),
    re.compile(r"\banomaly\b.*\brecover\b", re.IGNORECASE),
    re.compile(r"\bdo\s+recovery\b", re.IGNORECASE),
    re.compile(r"\bavoid\s+failure\b", re.IGNORECASE),
    re.compile(r"\buse\s+memory\b", re.IGNORECASE),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit LLM experience lessons for evidence grounding, validity, concision, and conflicts.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--save", type=Path, required=True)
    parser.add_argument("--max-lesson-words", type=int, default=20)
    parser.add_argument("--max-field-words", type=int, default=30)
    return parser.parse_args()


def _load_payload(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected lesson JSON object: {path}")
    return payload


def _tokens(text: Any) -> list[str]:
    return [item for item in re.split(r"[^A-Za-z0-9_]+", str(text or "")) if item]


def _word_count(text: Any) -> int:
    return len(_tokens(text))


def _candidate_evidence(evidence: dict[str, Any]) -> tuple[dict[str, list[str]], set[str], set[str]]:
    candidate_steps: dict[str, list[str]] = {}
    evidence_ids: set[str] = set()
    all_skills: set[str] = set()
    for candidate in evidence.get("candidates") or []:
        if not isinstance(candidate, dict):
            continue
        candidate_id = str(candidate.get("candidate_id") or "")
        steps = [str(item) for item in candidate.get("candidate_steps") or [] if str(item)]
        if candidate_id:
            candidate_steps[candidate_id] = steps
        all_skills.update(steps)
        for risk in candidate.get("top_failure_risks") or []:
            if isinstance(risk, dict) and risk.get("experience_id"):
                evidence_ids.add(str(risk["experience_id"]))
    return candidate_steps, evidence_ids, all_skills


def _skill_refs(text: Any, all_skills: set[str]) -> list[str]:
    return [token for token in _tokens(text) if token in all_skills]


def _unknown_skill_like_refs(text: Any, all_skills: set[str]) -> list[str]:
    unknown = []
    for token in _tokens(text):
        if "_" not in token:
            continue
        if token in all_skills:
            continue
        if token.startswith(("scenario", "condition", "candidate", "experience")):
            continue
        unknown.append(token)
    return sorted(set(unknown))


def _avoid_refs(text: Any, all_skills: set[str]) -> tuple[list[str], list[str], list[str], list[str]]:
    """Split avoid text into avoided skills, trigger skills, abstract risk terms, and invalid terms."""

    raw = str(text or "")
    tokens = _tokens(raw)
    known = [token for token in tokens if token in all_skills]
    unknown = _unknown_skill_like_refs(raw, all_skills)
    lowered = raw.lower()
    if "after" in lowered and tokens:
        bad_action = tokens[0]
        trigger = tokens[-1]
        avoided = [bad_action] if bad_action in all_skills else []
        triggers = [trigger] if trigger in all_skills else []
        abstract_risks = [bad_action] if bad_action not in all_skills and "_" in bad_action else []
        invalid = [item for item in unknown if item not in abstract_risks]
        return sorted(set(avoided)), sorted(set(triggers)), sorted(set(abstract_risks)), sorted(set(invalid))
    if "before" in lowered and tokens:
        bad_action = tokens[0]
        trigger = tokens[-1]
        avoided = [bad_action] if bad_action in all_skills else []
        triggers = [trigger] if trigger in all_skills else []
        abstract_risks = [bad_action] if bad_action not in all_skills and "_" in bad_action else []
        invalid = [item for item in unknown if item not in abstract_risks]
        return sorted(set(avoided)), sorted(set(triggers)), sorted(set(abstract_risks)), sorted(set(invalid))
    return sorted(set(known)), [], [], unknown


def _template_like_fields(lesson: dict[str, Any]) -> list[str]:
    hits = []
    for field in ("lesson", "if", "then", "avoid"):
        text = str(lesson.get(field) or "")
        if any(pattern.search(text) for pattern in TEMPLATE_LIKE_PATTERNS):
            hits.append(field)
    return hits


def _lesson_key(lesson: dict[str, Any]) -> str:
    return "|".join(str(lesson.get(field) or "").strip().lower() for field in ("lesson", "if", "then", "avoid"))


def _conflict_pairs(analyses: list[dict[str, Any]]) -> list[dict[str, Any]]:
    then_by_skill: dict[str, set[str]] = defaultdict(set)
    avoid_by_skill: dict[str, set[str]] = defaultdict(set)
    for item in analyses:
        lesson_id = str(item.get("lesson_id") or "")
        for skill in item.get("then_skill_refs") or []:
            then_by_skill[str(skill)].add(lesson_id)
        for skill in item.get("avoid_skill_refs") or []:
            avoid_by_skill[str(skill)].add(lesson_id)

    conflicts = []
    for skill in sorted(set(then_by_skill) & set(avoid_by_skill)):
        for then_id in sorted(then_by_skill[skill]):
            for avoid_id in sorted(avoid_by_skill[skill]):
                if then_id == avoid_id:
                    conflict_type = "self_conflict"
                else:
                    conflict_type = "cross_lesson_conflict"
                conflicts.append({
                    "skill": skill,
                    "then_lesson_id": then_id,
                    "avoid_lesson_id": avoid_id,
                    "conflict_type": conflict_type,
                })
    return conflicts


def _analyze_lesson(
    raw: dict[str, Any],
    *,
    source_path: str,
    valid_evidence_ids: set[str],
    valid_candidate_ids: set[str],
    all_skills: set[str],
    max_lesson_words: int,
    max_field_words: int,
) -> dict[str, Any]:
    lesson = normalize_lesson(raw, source_path=source_path)
    evidence_ids = set(lesson.get("evidence_ids") or [])
    candidate_ids = set(lesson.get("supporting_candidate_ids") or [])
    then_refs = sorted(set(_skill_refs(lesson.get("then"), all_skills)))
    avoid_refs, avoid_trigger_refs, abstract_risk_terms, invalid_avoid_refs = _avoid_refs(lesson.get("avoid"), all_skills)
    unknown_refs = sorted(set(_unknown_skill_like_refs(lesson.get("then"), all_skills) + invalid_avoid_refs))
    invalid_evidence = sorted(evidence_ids - valid_evidence_ids)
    invalid_candidates = sorted(candidate_ids - valid_candidate_ids)
    lesson_words = _word_count(lesson.get("lesson"))
    field_word_counts = {
        field: _word_count(lesson.get(field))
        for field in ("lesson", "if", "then", "avoid")
    }
    concise = lesson_words <= max_lesson_words and all(value <= max_field_words for value in field_word_counts.values())
    actionable = bool(then_refs or avoid_refs or (abstract_risk_terms and avoid_trigger_refs)) and not unknown_refs and not invalid_evidence and not invalid_candidates
    return {
        "lesson_id": lesson["lesson_id"],
        "lesson": lesson["lesson"],
        "if": lesson["if"],
        "then": lesson["then"],
        "avoid": lesson["avoid"],
        "confidence": lesson["confidence"],
        "lesson_word_count": lesson_words,
        "field_word_counts": field_word_counts,
        "concise": concise,
        "evidence_ids": sorted(evidence_ids),
        "invalid_evidence_ids": invalid_evidence,
        "supporting_candidate_ids": sorted(candidate_ids),
        "invalid_candidate_ids": invalid_candidates,
        "then_skill_refs": then_refs,
        "avoid_skill_refs": avoid_refs,
        "avoid_trigger_skill_refs": avoid_trigger_refs,
        "abstract_risk_terms": abstract_risk_terms,
        "unknown_skill_like_refs": unknown_refs,
        "skill_reference_valid": not unknown_refs,
        "template_like_fields": _template_like_fields(lesson),
        "actionable": actionable,
    }


def _rate(values: list[bool]) -> float:
    return round(sum(int(item) for item in values) / len(values), 4) if values else 0.0


def build_report(payload: dict[str, Any], *, input_path: Path, max_lesson_words: int, max_field_words: int) -> dict[str, Any]:
    raw_lessons = [item for item in payload.get("lessons") or [] if isinstance(item, dict)]
    evidence = payload.get("evidence") if isinstance(payload.get("evidence"), dict) else {}
    candidate_steps, valid_evidence_ids, all_skills = _candidate_evidence(evidence)
    valid_candidate_ids = set(candidate_steps)
    analyses = [
        _analyze_lesson(
            raw,
            source_path=str(input_path),
            valid_evidence_ids=valid_evidence_ids,
            valid_candidate_ids=valid_candidate_ids,
            all_skills=all_skills,
            max_lesson_words=max_lesson_words,
            max_field_words=max_field_words,
        )
        for raw in raw_lessons
    ]
    duplicate_keys = Counter(_lesson_key(item) for item in analyses)
    duplicate_lesson_ids = [
        item["lesson_id"]
        for item in analyses
        if duplicate_keys[_lesson_key(item)] > 1
    ]
    conflicts = _conflict_pairs(analyses)
    confidence_values = [float(item.get("confidence") or 0.0) for item in analyses]
    lesson_lengths = [int(item.get("lesson_word_count") or 0) for item in analyses]
    field_lengths = [
        int(count)
        for item in analyses
        for count in (item.get("field_word_counts") or {}).values()
    ]
    template_hits = [item for item in analyses if item.get("template_like_fields")]
    evidence_valid = [not item.get("invalid_evidence_ids") for item in analyses]
    candidate_valid = [not item.get("invalid_candidate_ids") for item in analyses]
    skill_valid = [bool(item.get("skill_reference_valid")) for item in analyses]
    concise = [bool(item.get("concise")) for item in analyses]
    actionable = [bool(item.get("actionable")) for item in analyses]

    return {
        "schema_version": "experience_lesson_quality_report_v1",
        "input": str(input_path),
        "scenario": evidence.get("scenario", ""),
        "condition": evidence.get("condition", ""),
        "evidence_summary": {
            "candidate_count": len(valid_candidate_ids),
            "valid_candidate_ids": sorted(valid_candidate_ids),
            "valid_evidence_id_count": len(valid_evidence_ids),
            "valid_evidence_ids": sorted(valid_evidence_ids),
            "known_skill_count": len(all_skills),
        },
        "metrics": {
            "lesson_count": len(analyses),
            "avg_lesson_length": round(mean(lesson_lengths), 4) if lesson_lengths else 0.0,
            "max_lesson_length": max(lesson_lengths) if lesson_lengths else 0,
            "avg_field_length": round(mean(field_lengths), 4) if field_lengths else 0.0,
            "max_field_length": max(field_lengths) if field_lengths else 0,
            "evidence_id_valid_rate": _rate(evidence_valid),
            "candidate_id_valid_rate": _rate(candidate_valid),
            "skill_reference_valid_rate": _rate(skill_valid),
            "duplicate_lesson_count": len(duplicate_lesson_ids),
            "conflict_pair_count": len(conflicts),
            "template_like_phrase_count": len(template_hits),
            "actionable_lesson_rate": _rate(actionable),
            "concise_lesson_rate": _rate(concise),
            "confidence_avg": round(mean(confidence_values), 4) if confidence_values else 0.0,
        },
        "quality_pass": bool(
            analyses
            and _rate(evidence_valid) == 1.0
            and _rate(candidate_valid) == 1.0
            and _rate(skill_valid) == 1.0
            and len(duplicate_lesson_ids) == 0
            and len(conflicts) == 0
            and len(template_hits) == 0
            and _rate(actionable) == 1.0
        ),
        "duplicate_lesson_ids": duplicate_lesson_ids,
        "conflicts": conflicts,
        "template_like_lessons": [
            {
                "lesson_id": item["lesson_id"],
                "template_like_fields": item["template_like_fields"],
            }
            for item in template_hits
        ],
        "lessons": analyses,
        "safe_paper_wording": "Generated lessons are checked for evidence grounding, candidate/skill validity, concision, and internal conflicts before being used for policy adjustment.",
        "claim_boundary": "This is a static quality audit for generated lessons, not proof of learned policy rules.",
    }


def main() -> None:
    args = parse_args()
    payload = _load_payload(args.input)
    report = build_report(
        payload,
        input_path=args.input,
        max_lesson_words=args.max_lesson_words,
        max_field_words=args.max_field_words,
    )
    args.save.parent.mkdir(parents=True, exist_ok=True)
    args.save.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({
        "save": str(args.save),
        "quality_pass": report["quality_pass"],
        "metrics": report["metrics"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
