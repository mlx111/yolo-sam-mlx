"""Run a cold-start vs writeback field_atomic memory ablation."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
GALAXEA_ROOT = ROOT.parent / "galaxea_mujoco"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare field_atomic planning with empty memory vs writeback memory.")
    parser.add_argument("--goal", required=True)
    parser.add_argument("--model-path", default="r1pro_g3_sorting_scene.xml")
    parser.add_argument("--scenario-id", default="field_atomic")
    parser.add_argument("--condition-id", default="default")
    parser.add_argument("--provider", choices=["doubao", "openai"], default="doubao")
    parser.add_argument("--model", default="")
    parser.add_argument("--dry-run-llm", action="store_true")
    parser.add_argument("--save-json", type=Path, required=True)
    parser.add_argument("--save-md", type=Path, required=True)
    return parser.parse_args()


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_md(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _run_plan(
    *,
    goal: str,
    model_path: str,
    scenario_id: str,
    condition_id: str,
    library_input: Path | None,
    library_output: Path,
    plan_path: Path,
    report_path: Path,
    dry_run_llm: bool,
    provider: str,
    model: str,
) -> dict[str, Any]:
    cmd = [
        sys.executable,
        "-B",
        "source/run_field_atomic_llm_plan.py",
        "--goal",
        goal,
        "--model-path",
        model_path,
        "--writeback-library-output",
        str(library_output),
        "--scenario-id",
        scenario_id,
        "--condition-id",
        condition_id,
        "--save-plan",
        str(plan_path),
        "--save-report",
        str(report_path),
    ]
    if library_input is not None:
        cmd.extend(["--universal-experience-lib", str(library_input)])
    if dry_run_llm:
        cmd.append("--dry-run-llm")
    else:
        cmd.extend(["--provider", provider])
        if model:
            cmd.extend(["--model", model])
    completed = subprocess.run(
        cmd,
        cwd=GALAXEA_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    return {
        "returncode": completed.returncode,
        "stdout": completed.stdout[-2000:],
        "stderr": completed.stderr[-3000:],
        "report": _read(report_path) if report_path.exists() else {},
        "plan": _read(plan_path) if plan_path.exists() else {},
    }


def _param_signature(plan: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for step in plan.get("steps") or []:
        if not isinstance(step, dict):
            continue
        action = str(step.get("action") or "")
        params = step.get("parameters") if isinstance(step.get("parameters"), dict) else {}
        out[action] = params
    return out


def _compare_plans(cold: dict[str, Any], warm: dict[str, Any]) -> dict[str, Any]:
    cold_steps = cold.get("steps") or []
    warm_steps = warm.get("steps") or []
    cold_sig = _param_signature(cold)
    warm_sig = _param_signature(warm)
    actions = sorted(set(cold_sig) | set(warm_sig))
    deltas = {}
    changed = 0
    for action in actions:
        if cold_sig.get(action) != warm_sig.get(action):
            changed += 1
            deltas[action] = {
                "cold": cold_sig.get(action, {}),
                "warm": warm_sig.get(action, {}),
            }
    return {
        "cold_step_count": len(cold_steps),
        "warm_step_count": len(warm_steps),
        "plan_step_count_delta": len(warm_steps) - len(cold_steps),
        "action_set_changed": sorted(cold_sig) != sorted(warm_sig),
        "parameter_changed_count": changed,
        "parameter_delta_by_action": deltas,
    }


def _planner_metrics(report: dict[str, Any]) -> dict[str, Any]:
    planner = report.get("planner_input") if isinstance(report.get("planner_input"), dict) else {}
    priors = planner.get("field_atomic_parameter_priors") if isinstance(planner.get("field_atomic_parameter_priors"), dict) else {}
    return {
        "field_atomic_memory_count": int(planner.get("field_atomic_memory_count") or 0),
        "prior_action_count": len((priors.get("by_action") or {}) if isinstance(priors.get("by_action"), dict) else {}),
        "prior_entry_count": int(priors.get("field_atomic_entry_count") or 0),
        "prior_success_count": int(priors.get("field_atomic_success_count") or 0),
        "prior_failure_count": int(priors.get("field_atomic_failure_count") or 0),
    }


def _render_md(report: dict[str, Any]) -> str:
    lines = [
        "# Field Atomic Memory Ablation",
        "",
        f"- Goal: {report['goal']}",
        f"- Dry-run LLM: {report['dry_run_llm']}",
        "",
        "## Memory",
        "",
        f"- Cold memory count: {report['cold_start']['planner_metrics']['field_atomic_memory_count']}",
        f"- Warm memory count: {report['warm_start']['planner_metrics']['field_atomic_memory_count']}",
        f"- Warm prior action count: {report['warm_start']['planner_metrics']['prior_action_count']}",
        "",
        "## Plan Difference",
        "",
        f"- Cold step count: {report['comparison']['cold_step_count']}",
        f"- Warm step count: {report['comparison']['warm_step_count']}",
        f"- Parameter changed count: {report['comparison']['parameter_changed_count']}",
        f"- Action set changed: {report['comparison']['action_set_changed']}",
        "",
        "## Execution",
        "",
        f"- Cold success count: {report['cold_start']['success_count']}",
        f"- Warm success count: {report['warm_start']['success_count']}",
        f"- Cold write count: {report['cold_start']['write_count']}",
        f"- Warm write count: {report['warm_start']['write_count']}",
    ]
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    with tempfile.TemporaryDirectory(prefix="field_atomic_ablation_") as tmp:
        root = Path(tmp)
        cold_library = root / "cold_library.json"
        warm_library = root / "warm_library.json"
        cold_plan = root / "cold_plan.json"
        warm_plan = root / "warm_plan.json"
        cold_report_path = root / "cold_report.json"
        warm_report_path = root / "warm_report.json"
        cold = _run_plan(
            goal=args.goal,
            model_path=args.model_path,
            scenario_id=args.scenario_id,
            condition_id=args.condition_id,
            library_input=None,
            library_output=cold_library,
            plan_path=cold_plan,
            report_path=cold_report_path,
            dry_run_llm=args.dry_run_llm,
            provider=args.provider,
            model=args.model,
        )
        warm = _run_plan(
            goal=args.goal,
            model_path=args.model_path,
            scenario_id=args.scenario_id,
            condition_id=args.condition_id,
            library_input=cold_library,
            library_output=warm_library,
            plan_path=warm_plan,
            report_path=warm_report_path,
            dry_run_llm=args.dry_run_llm,
            provider=args.provider,
            model=args.model,
        )

    report = {
        "schema_version": "field_atomic_memory_ablation_v1",
        "goal": args.goal,
        "model_path": args.model_path,
        "scenario_id": args.scenario_id,
        "condition_id": args.condition_id,
        "dry_run_llm": bool(args.dry_run_llm),
        "cold_start": {
            "returncode": cold["returncode"],
            "planner_metrics": _planner_metrics(cold["report"]),
            "success_count": int(cold["report"].get("success_count") or 0),
            "failure_count": int(cold["report"].get("failure_count") or 0),
            "write_count": int((cold["report"].get("writeback") or {}).get("write_count") or 0),
            "plan": cold["plan"],
        },
        "warm_start": {
            "returncode": warm["returncode"],
            "planner_metrics": _planner_metrics(warm["report"]),
            "success_count": int(warm["report"].get("success_count") or 0),
            "failure_count": int(warm["report"].get("failure_count") or 0),
            "write_count": int((warm["report"].get("writeback") or {}).get("write_count") or 0),
            "plan": warm["plan"],
        },
        "comparison": _compare_plans(cold["plan"], warm["plan"]),
        "paper_wording": {
            "safe_claim": "Field atomic writeback can be reloaded as explicit planner_input priors in a subsequent planning round.",
            "avoid_claim": "Do not claim parameter improvement from dry-run LLM if the mock plan is deterministic; use real LLM or varied goals for parameter-change evidence.",
        },
    }
    _write_json(args.save_json, report)
    _write_md(args.save_md, _render_md(report))
    print(json.dumps({
        "cold_memory_count": report["cold_start"]["planner_metrics"]["field_atomic_memory_count"],
        "warm_memory_count": report["warm_start"]["planner_metrics"]["field_atomic_memory_count"],
        "parameter_changed_count": report["comparison"]["parameter_changed_count"],
        "save_json": str(args.save_json),
        "save_md": str(args.save_md),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
