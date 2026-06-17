"""LLM-verbalized experience lessons and candidate score adjustment."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(value, high))


def _stable_id(*parts: Any) -> str:
    raw = "|".join(str(part) for part in parts)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:10]


def _tokens(text: Any) -> list[str]:
    return [item for item in re.split(r"[^A-Za-z0-9_]+", str(text or "")) if item]


def _contains_order(actions: list[str], ordered_terms: list[str]) -> bool:
    if not ordered_terms:
        return False
    index = 0
    for action in actions:
        if action == ordered_terms[index]:
            index += 1
            if index == len(ordered_terms):
                return True
    return False


def _skill_terms(text: Any, known_actions: list[str]) -> list[str]:
    known = set(known_actions)
    return [token for token in _tokens(text) if token in known]


def _first_index(actions: list[str], term: str) -> int | None:
    try:
        return actions.index(term)
    except ValueError:
        return None


def _contains_all(actions: list[str], terms: list[str]) -> bool:
    return bool(terms) and all(term in actions for term in terms)


def _relation_match(text: Any, actions: list[str], *, polarity: str) -> tuple[bool, list[dict[str, Any]]]:
    """Interpret compact LLM lesson text as ordered action evidence.

    The LLM is intentionally allowed to write natural language lessons. This
    parser only consumes explicit action names and simple "A before B"/"A after
    B" relations, so lesson scoring stays explainable instead of relying on
    broad token overlap.
    """

    raw = str(text or "")
    terms = _skill_terms(raw, actions)
    if not terms:
        return False, []

    evidence: list[dict[str, Any]] = []
    lowered = raw.lower()
    relation_patterns = [
        ("before", re.compile(r"([A-Za-z0-9_]+)\s+before\s+([A-Za-z0-9_]+)", re.IGNORECASE)),
        ("after", re.compile(r"([A-Za-z0-9_]+)\s+(?:\(|\[)?\s*after\s+([A-Za-z0-9_]+)", re.IGNORECASE)),
    ]
    for relation, pattern in relation_patterns:
        for match in pattern.finditer(raw):
            left, right = match.group(1), match.group(2)
            if left not in actions or right not in actions:
                continue
            left_index = _first_index(actions, left)
            right_index = _first_index(actions, right)
            if left_index is None or right_index is None:
                continue
            if relation == "before":
                matched = left_index < right_index
            else:
                matched = left_index > right_index
            evidence.append({
                "type": relation,
                "left": left,
                "right": right,
                "left_index": left_index,
                "right_index": right_index,
                "matched": matched,
            })

    if evidence:
        return any(item["matched"] for item in evidence), evidence

    if polarity == "avoid" and ("before" in lowered or "after" in lowered):
        return False, [{
            "type": "unresolved_relation",
            "terms": terms,
            "matched": False,
        }]

    # Natural phrasing such as "detect, choose before transport" is common.
    if "before" in lowered and len(terms) >= 2:
        pivot = terms[-1]
        pivot_index = _first_index(actions, pivot)
        if pivot_index is not None:
            relation_evidence = []
            for term in terms[:-1]:
                term_index = _first_index(actions, term)
                if term_index is None:
                    continue
                relation_evidence.append({
                    "type": "before",
                    "left": term,
                    "right": pivot,
                    "left_index": term_index,
                    "right_index": pivot_index,
                    "matched": term_index < pivot_index,
                })
            if relation_evidence:
                return all(item["matched"] for item in relation_evidence), relation_evidence

    # Parenthetical "A (after B)" tokenizes as [A, B]; avoid should only fire
    # if the bad action happens after its trigger, not just because both exist.
    if "after" in lowered and len(terms) >= 2:
        bad_action = terms[0]
        trigger = terms[-1]
        bad_index = _first_index(actions, bad_action)
        trigger_index = _first_index(actions, trigger)
        if bad_index is not None and trigger_index is not None:
            relation_evidence = [{
                "type": "after",
                "left": bad_action,
                "right": trigger,
                "left_index": bad_index,
                "right_index": trigger_index,
                "matched": bad_index > trigger_index,
            }]
            return relation_evidence[0]["matched"], relation_evidence

    if polarity == "then":
        if len(terms) >= 2:
            return _contains_order(actions, terms), [{
                "type": "ordered_terms",
                "terms": terms,
                "matched": _contains_order(actions, terms),
            }]
        return _contains_all(actions, terms), [{
            "type": "contains_terms",
            "terms": terms,
            "matched": _contains_all(actions, terms),
        }]

    # Avoid text without a relation is treated as conservative overlap. This
    # keeps older lessons usable, while explicit relations above avoid false
    # penalties for good plans.
    return _contains_all(actions, terms), [{
        "type": "contains_terms",
        "terms": terms,
        "matched": _contains_all(actions, terms),
    }]


def normalize_lesson(raw: dict[str, Any], *, source_path: str = "") -> dict[str, Any]:
    lesson = {
        "lesson_id": str(raw.get("lesson_id") or ""),
        "lesson": str(raw.get("lesson") or "").strip(),
        "if": str(raw.get("if") or "").strip(),
        "then": str(raw.get("then") or "").strip(),
        "avoid": str(raw.get("avoid") or "").strip(),
        "evidence_ids": [str(item) for item in raw.get("evidence_ids") or [] if str(item)],
        "supporting_candidate_ids": [str(item) for item in raw.get("supporting_candidate_ids") or [] if str(item)],
        "confidence": _clamp(float(raw.get("confidence") or 0.0), 0.0, 1.0),
        "source_path": source_path,
    }
    if not lesson["lesson_id"]:
        lesson["lesson_id"] = "lesson_" + _stable_id(lesson["if"], lesson["then"], lesson["avoid"], lesson["lesson"])
    return lesson


def load_lesson_library(path: Path | str | None) -> list[dict[str, Any]]:
    if path is None:
        return []
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"lesson library not found: {source}")
    payload = json.loads(source.read_text(encoding="utf-8"))
    lessons = payload.get("lessons") if isinstance(payload, dict) else payload
    if not isinstance(lessons, list):
        raise ValueError(f"lesson library must contain a lessons list: {source}")
    return [
        normalize_lesson(item, source_path=str(source))
        for item in lessons
        if isinstance(item, dict)
    ]


def lesson_matches_context(lesson: dict[str, Any], *, scenario: str, condition: str) -> bool:
    condition_text = f"{lesson.get('if', '')} {lesson.get('lesson', '')}".lower()
    scenario = scenario.lower()
    condition = condition.lower()
    scenario_ok = not scenario or scenario in condition_text
    condition_ok = not condition or condition in condition_text
    return scenario_ok and condition_ok


def adjust_candidate_with_lessons(
    candidate_report: dict[str, Any],
    lessons: list[dict[str, Any]],
    *,
    scenario: str,
    condition: str,
    lesson_weight: float = 0.08,
) -> dict[str, Any]:
    candidate_actions = [str(item) for item in candidate_report.get("candidate_steps") or []]
    if not candidate_actions or not lessons:
        return {
            "matched_lessons": [],
            "lesson_bonus": 0.0,
            "lesson_penalty": 0.0,
            "net_adjustment": 0.0,
        }

    matched: list[dict[str, Any]] = []
    bonus = 0.0
    penalty = 0.0
    for lesson in lessons:
        if not lesson_matches_context(lesson, scenario=scenario, condition=condition):
            continue
        then_match, then_evidence = _relation_match(lesson.get("then", ""), candidate_actions, polarity="then")
        avoid_match, avoid_evidence = _relation_match(lesson.get("avoid", ""), candidate_actions, polarity="avoid")
        if not then_match and not avoid_match:
            continue
        confidence = float(lesson.get("confidence", 0.0))
        strength = _clamp(float(lesson_weight), 0.0, 0.5) * confidence
        if then_match:
            bonus += strength
        if avoid_match:
            penalty += strength
        matched.append({
            "lesson_id": lesson.get("lesson_id", ""),
            "lesson": lesson.get("lesson", ""),
            "then_match": then_match,
            "avoid_match": avoid_match,
            "then_evidence": then_evidence,
            "avoid_evidence": avoid_evidence,
            "confidence": round(confidence, 4),
            "evidence_ids": lesson.get("evidence_ids", []),
        })

    bonus = _clamp(bonus, 0.0, 0.25)
    penalty = _clamp(penalty, 0.0, 0.25)
    net = bonus - penalty
    score = candidate_report.get("candidate_score") or {}
    if score:
        original = float(score.get("candidate_score", 0.0))
        risk = float(score.get("risk_score", 0.0))
        adjusted = _clamp(original + net, 0.0, 1.0)
        adjusted_risk = _clamp(risk + penalty - 0.5 * bonus, 0.0, 1.0)
        score["base_candidate_score"] = round(original, 4)
        score["candidate_score"] = round(adjusted, 4)
        score["risk_score"] = round(adjusted_risk, 4)
        if adjusted_risk >= 0.65:
            score["decision"] = "rewrite"
        elif adjusted >= 0.60:
            score["decision"] = "accept"
        elif adjusted >= 0.40:
            score["decision"] = "review"
        else:
            score["decision"] = "reject"
        score["lesson_adjustment"] = {
            "matched_lessons": matched,
            "lesson_bonus": round(bonus, 4),
            "lesson_penalty": round(penalty, 4),
            "net_adjustment": round(net, 4),
        }
    return {
        "matched_lessons": matched,
        "lesson_bonus": round(bonus, 4),
        "lesson_penalty": round(penalty, 4),
        "net_adjustment": round(net, 4),
    }
