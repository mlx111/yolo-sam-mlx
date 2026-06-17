"""Induce skill precondition/effect metadata with the experience-system LLM."""

from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experience_core import JSON_ONLY_LINE, default_r1pro_skill_semantics, invoke_llm, parse_json_payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a skill_semantics_candidate_v1 JSON draft for a robot skill.")
    parser.add_argument("--skill-name", required=True, help="Canonical skill action name used in plans.")
    parser.add_argument("--skill-file", type=Path, required=True, help="Python file implementing or wrapping the skill.")
    parser.add_argument("--class-name", default="", help="Optional class to inspect; otherwise all classes are summarized.")
    parser.add_argument("--function-name", default="execute_recovery_action", help="Main method/function to inspect.")
    parser.add_argument("--extra-doc", type=Path, action="append", default=[], help="Optional README/spec files to include.")
    parser.add_argument("--provider", choices=["doubao", "openai"], default="doubao")
    parser.add_argument("--model", default="")
    parser.add_argument("--dry-run", action="store_true", help="Do not call LLM; emit a conservative deterministic draft.")
    parser.add_argument("--save", type=Path, required=True)
    return parser.parse_args()


def _read_text(path: Path, *, limit: int = 16000) -> str:
    text = path.read_text(encoding="utf-8", errors="ignore")
    return text[:limit]


def _annotation_text(node: ast.AST | None) -> str:
    if node is None:
        return ""
    try:
        return ast.unparse(node)
    except Exception:
        return ""


def _function_signature(node: ast.FunctionDef | ast.AsyncFunctionDef) -> dict[str, Any]:
    args = []
    all_args = list(node.args.posonlyargs) + list(node.args.args) + list(node.args.kwonlyargs)
    for arg in all_args:
        args.append({
            "name": arg.arg,
            "annotation": _annotation_text(arg.annotation),
        })
    return {
        "name": node.name,
        "args": args,
        "returns": _annotation_text(node.returns),
        "docstring": ast.get_docstring(node) or "",
    }


def _extract_code_evidence(skill_file: Path, *, class_name: str, function_name: str) -> dict[str, Any]:
    source = _read_text(skill_file, limit=26000)
    tree = ast.parse(source)
    module_doc = ast.get_docstring(tree) or ""
    classes: list[dict[str, Any]] = []
    functions: list[dict[str, Any]] = []
    relevant_source: list[str] = []

    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            if class_name and node.name != class_name:
                continue
            methods = [
                _function_signature(item)
                for item in node.body
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
                and (not function_name or item.name == function_name or item.name == "__init__")
            ]
            classes.append({
                "name": node.name,
                "docstring": ast.get_docstring(node) or "",
                "methods": methods,
            })
            try:
                relevant_source.append(ast.get_source_segment(source, node) or "")
            except Exception:
                pass
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and (not function_name or node.name == function_name or node.name == "load_skill"):
            functions.append(_function_signature(node))
            try:
                relevant_source.append(ast.get_source_segment(source, node) or "")
            except Exception:
                pass

    return {
        "skill_file": str(skill_file),
        "module_docstring": module_doc,
        "classes": classes,
        "functions": functions,
        "source_excerpt": "\n\n".join(item for item in relevant_source if item)[:14000],
    }


def _extra_docs(paths: list[Path]) -> list[dict[str, str]]:
    docs = []
    for path in paths:
        docs.append({"path": str(path), "text": _read_text(path, limit=8000)})
    return docs


def _known_fact_context() -> dict[str, Any]:
    facts: set[str] = set()
    skills = {}
    for name, semantic in default_r1pro_skill_semantics().items():
        payload = semantic.to_dict()
        skills[name] = payload
        facts.update(payload["requires"])
        facts.update(payload["effects"])
        facts.update(payload["consumes"])
        facts.update(payload["optional_requires"])
    return {
        "known_facts": sorted(facts),
        "known_skill_examples": skills,
    }


def _prompt(*, skill_name: str, code_evidence: dict[str, Any], docs: list[dict[str, str]]) -> str:
    context = _known_fact_context()
    return f"""
You infer robot skill semantics for a planning validator.

The output is a candidate draft, not trusted ground truth. Be conservative:
- Only put facts in requires when the skill cannot execute safely without them.
- Put useful but non-mandatory facts in optional_requires.
- Put facts made true after successful execution in effects.
- Put facts no longer guaranteed after execution in consumes.
- Prefer existing fact names when they fit.
- Invent new fact names only when necessary, using lowercase snake_case.
- Do not infer a full task order or scenario-specific sequence.

Skill name:
{skill_name}

Known fact vocabulary and examples:
{json.dumps(context, ensure_ascii=False, indent=2)}

Code evidence:
{json.dumps(code_evidence, ensure_ascii=False, indent=2)}

Extra docs:
{json.dumps(docs, ensure_ascii=False, indent=2)}

Return one JSON object:
{{
  "schema_version": "skill_semantics_candidate_v1",
  "skill": "{skill_name}",
  "description": "short description",
  "requires": ["required_fact"],
  "optional_requires": ["optional_fact"],
  "effects": ["effect_fact"],
  "consumes": ["consumed_fact"],
  "parameters": {{
    "param_name": {{
      "type": "string|number|boolean|object|array",
      "required": false,
      "description": "short description"
    }}
  }},
  "risks": ["short risk"],
  "evidence": {{
    "source_files": ["paths"],
    "reasoning": "brief explanation grounded in code/docs"
  }},
  "confidence": 0.0
}}

{JSON_ONLY_LINE}
"""


def _dry_run_candidate(skill_name: str, code_evidence: dict[str, Any], docs: list[dict[str, str]]) -> dict[str, Any]:
    text = json.dumps(code_evidence, ensure_ascii=False).lower()
    requires: list[str] = []
    optional_requires: list[str] = []
    effects: list[str] = []
    risks = ["dry_run_candidate_requires_llm_or_human_review"]

    if "camera" in text or "rgb" in text or "depth" in text:
        optional_requires.append("sensor_available")
        effects.extend(["scene_observed", "rgbd_observation_available"])
        risks.extend(["sensor_timeout", "invalid_depth"])
    elif "lidar" in text or "scan" in text:
        optional_requires.append("sensor_available")
        effects.extend(["lidar_scan_available", "scene_observed"])
        risks.append("sparse_or_noisy_scan")
    elif "force" in text or "wrist" in text:
        optional_requires.append("force_sensor_available")
        effects.append("wrist_force_observed")
        risks.append("force_sensor_bias")
    else:
        effects.append(f"{skill_name}_completed")

    return {
        "schema_version": "skill_semantics_candidate_v1",
        "skill": skill_name,
        "description": "Conservative dry-run semantics candidate generated without calling an LLM.",
        "requires": requires,
        "optional_requires": sorted(set(optional_requires)),
        "effects": sorted(set(effects)),
        "consumes": [],
        "parameters": {},
        "risks": sorted(set(risks)),
        "evidence": {
            "source_files": [str(code_evidence.get("skill_file") or "")] + [doc["path"] for doc in docs],
            "reasoning": "Dry-run heuristic based on code text keywords; use LLM and validation before registry write.",
        },
        "confidence": 0.15,
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    args = parse_args()
    code_evidence = _extract_code_evidence(args.skill_file, class_name=args.class_name, function_name=args.function_name)
    docs = _extra_docs(args.extra_doc)
    if args.dry_run:
        candidate = _dry_run_candidate(args.skill_name, code_evidence, docs)
    else:
        raw = invoke_llm(
            _prompt(skill_name=args.skill_name, code_evidence=code_evidence, docs=docs),
            provider=args.provider,
            model=args.model,
            system_prompt="You infer robot skill semantics and return JSON only.",
            temperature=0.1,
        )
        payload = parse_json_payload(raw, prefer_array=False)
        if not isinstance(payload, dict):
            raise RuntimeError("LLM skill semantics response was not a JSON object")
        candidate = payload
    candidate.setdefault("schema_version", "skill_semantics_candidate_v1")
    candidate.setdefault("skill", args.skill_name)
    candidate.setdefault("evidence", {}).setdefault("source_files", [str(args.skill_file)] + [str(path) for path in args.extra_doc])
    _write_json(args.save, candidate)
    print(json.dumps({"skill": candidate.get("skill"), "save": str(args.save), "dry_run": bool(args.dry_run)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
