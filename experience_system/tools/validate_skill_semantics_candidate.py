"""Validate an LLM-induced skill semantics candidate before registry use."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experience_core import SkillSemantics, default_r1pro_skill_semantics


FACT_RE = re.compile(r"^[a-z][a-z0-9_]*$")
REQUIRED_KEYS = {"schema_version", "skill", "description", "requires", "optional_requires", "effects", "consumes", "confidence"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate skill_semantics_candidate_v1 JSON.")
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--save-report", type=Path, required=True)
    parser.add_argument("--save-normalized", type=Path, default=None)
    parser.add_argument("--allow-new-facts", action="store_true")
    parser.add_argument("--fail-on-warnings", action="store_true")
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError("candidate must be a JSON object")
    return payload


def _list_str(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _known_facts() -> set[str]:
    facts: set[str] = set()
    for semantic in default_r1pro_skill_semantics().values():
        payload = semantic.to_dict()
        for key in ("requires", "optional_requires", "effects", "consumes"):
            facts.update(payload[key])
    return facts


def _issue(severity: str, code: str, message: str, **extra: Any) -> dict[str, Any]:
    return {"severity": severity, "code": code, "message": message, **extra}


def normalize_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "skill_semantics_candidate_v1",
        "skill": str(candidate.get("skill") or "").strip(),
        "description": str(candidate.get("description") or "").strip(),
        "requires": sorted(set(_list_str(candidate.get("requires")))),
        "optional_requires": sorted(set(_list_str(candidate.get("optional_requires")))),
        "effects": sorted(set(_list_str(candidate.get("effects")))),
        "consumes": sorted(set(_list_str(candidate.get("consumes")))),
        "parameters": candidate.get("parameters") if isinstance(candidate.get("parameters"), dict) else {},
        "risks": sorted(set(_list_str(candidate.get("risks")))),
        "evidence": candidate.get("evidence") if isinstance(candidate.get("evidence"), dict) else {},
        "confidence": candidate.get("confidence", 0.0),
    }


def validate_candidate(candidate: dict[str, Any], *, allow_new_facts: bool, fail_on_warnings: bool) -> dict[str, Any]:
    normalized = normalize_candidate(candidate)
    issues: list[dict[str, Any]] = []
    missing_keys = sorted(REQUIRED_KEYS - set(candidate))
    if missing_keys:
        issues.append(_issue("fatal", "missing_required_keys", f"candidate missing keys: {', '.join(missing_keys)}", keys=missing_keys))
    if candidate.get("schema_version") != "skill_semantics_candidate_v1":
        issues.append(_issue("fatal", "bad_schema_version", "schema_version must be skill_semantics_candidate_v1"))
    if not normalized["skill"]:
        issues.append(_issue("fatal", "empty_skill_name", "skill must be non-empty"))
    if not normalized["description"]:
        issues.append(_issue("warning", "empty_description", "description is empty"))
    if not normalized["effects"]:
        issues.append(_issue("warning", "no_effects", "candidate has no effects; validator cannot learn useful state transitions"))

    confidence = normalized["confidence"]
    if not isinstance(confidence, (int, float)) or not 0.0 <= float(confidence) <= 1.0:
        issues.append(_issue("fatal", "bad_confidence", "confidence must be a number from 0 to 1"))
        normalized["confidence"] = 0.0
    else:
        normalized["confidence"] = round(float(confidence), 4)

    fact_fields = ("requires", "optional_requires", "effects", "consumes")
    all_facts = sorted(set().union(*(set(normalized[key]) for key in fact_fields)))
    bad_names = [fact for fact in all_facts if not FACT_RE.match(fact)]
    if bad_names:
        issues.append(_issue("fatal", "bad_fact_names", "facts must be lowercase snake_case", facts=bad_names))

    known = _known_facts()
    new_facts = sorted(set(all_facts) - known)
    if new_facts and not allow_new_facts:
        issues.append(_issue("warning", "new_fact_names", "candidate introduces facts not in current vocabulary", facts=new_facts))

    overlap_requires_effects = sorted(set(normalized["requires"]) & set(normalized["effects"]))
    if overlap_requires_effects:
        issues.append(_issue("warning", "requires_effects_overlap", "facts appear in both requires and effects", facts=overlap_requires_effects))
    overlap_effects_consumes = sorted(set(normalized["effects"]) & set(normalized["consumes"]))
    if overlap_effects_consumes:
        issues.append(_issue("fatal", "effects_consumes_overlap", "facts cannot be both produced and consumed by the same skill", facts=overlap_effects_consumes))

    existing = default_r1pro_skill_semantics().get(normalized["skill"])
    if existing is not None:
        existing_payload = existing.to_dict()
        changed = {
            key: {"existing": existing_payload[key], "candidate": normalized[key]}
            for key in fact_fields
            if list(existing_payload[key]) != list(normalized[key])
        }
        if changed:
            issues.append(_issue("warning", "overrides_existing_skill_semantics", "candidate changes existing skill semantics", changes=changed))

    fatal_count = sum(1 for issue in issues if issue["severity"] == "fatal")
    warning_count = sum(1 for issue in issues if issue["severity"] == "warning")
    status = "fail" if fatal_count or (fail_on_warnings and warning_count) else "warn" if warning_count else "pass"
    registry_fragment = SkillSemantics(
        name=normalized["skill"],
        requires=frozenset(normalized["requires"]),
        optional_requires=frozenset(normalized["optional_requires"]),
        effects=frozenset(normalized["effects"]),
        consumes=frozenset(normalized["consumes"]),
        description=normalized["description"],
    ).to_dict() if normalized["skill"] else {}
    return {
        "schema_version": "skill_semantics_candidate_validation_v1",
        "status": status,
        "fatal_count": fatal_count,
        "warning_count": warning_count,
        "issues": issues,
        "normalized_candidate": normalized,
        "registry_fragment": registry_fragment,
        "known_fact_count": len(known),
        "new_facts": new_facts,
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    args = parse_args()
    candidate = _load_json(args.candidate)
    report = validate_candidate(candidate, allow_new_facts=bool(args.allow_new_facts), fail_on_warnings=bool(args.fail_on_warnings))
    _write_json(args.save_report, report)
    if args.save_normalized is not None:
        _write_json(args.save_normalized, report["normalized_candidate"])
    print(json.dumps({
        "status": report["status"],
        "fatal_count": report["fatal_count"],
        "warning_count": report["warning_count"],
        "save_report": str(args.save_report),
        "save_normalized": str(args.save_normalized) if args.save_normalized else "",
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
