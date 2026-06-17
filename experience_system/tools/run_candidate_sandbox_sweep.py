"""Run lightweight perturbation sweeps for R1Pro sandbox candidates."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experience_core import (
    ExperienceLibrary,
    apply_perturbation_to_state,
    generate_sandbox_perturbations,
    robust_sandbox_summary,
    sandbox_perturbation_from_dict,
)
from source.candidate_sandbox import evaluate_candidate_in_sandbox, select_sandbox_calibration
from source.run_r1pro_memory_policy_smoke import candidates_for_scenario, object_class_for_scenario


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run perturbation sweep sandbox rollout for candidate recovery plans.")
    parser.add_argument("--scenario", choices=["G3"], required=False)
    parser.add_argument("--condition", choices=["clean", "place_occupied"], required=False)
    parser.add_argument("--control-mode", choices=["ideal", "physical"], default="physical")
    parser.add_argument("--universal-experience-lib", type=Path, default=None)
    parser.add_argument("--sandbox-initial-state", type=Path, default=None)
    parser.add_argument("--candidate-id", default="", help="optional single candidate id")
    parser.add_argument("--num-rollouts", type=int, default=3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--object-pose-noise", type=float, default=0.015)
    parser.add_argument("--sandbox-parameter-profile", type=Path, default=None, help="optional sandbox_parameter_profile_v1 JSON")
    parser.add_argument("--include-risky-candidates", action="store_true")
    parser.add_argument("--use-sandbox-calibration", action="store_true")
    parser.add_argument("--keyframe-dir", type=Path, default=None)
    parser.add_argument("--trace-dir", type=Path, default=None)
    parser.add_argument("--parallel-workers", type=int, default=1, help="number of subprocess rollout workers; 1 keeps serial behavior")
    parser.add_argument("--worker-rollout", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--worker-job", type=Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--worker-output", type=Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--determinism-check", action="store_true", help="repeat the nominal rollout once and compare stable output fields")
    parser.add_argument("--save", type=Path, default=None)
    args = parser.parse_args()
    if not args.worker_rollout:
        missing = []
        if not args.scenario:
            missing.append("--scenario")
        if not args.condition:
            missing.append("--condition")
        if args.universal_experience_lib is None:
            missing.append("--universal-experience-lib")
        if args.sandbox_initial_state is None:
            missing.append("--sandbox-initial-state")
        if args.save is None:
            missing.append("--save")
        if missing:
            parser.error("required arguments missing outside --worker-rollout: " + ", ".join(missing))
    return args


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _candidate_ids(args: argparse.Namespace) -> list[str]:
    candidates = candidates_for_scenario(args.scenario, include_risky=args.include_risky_candidates)
    if args.candidate_id:
        return [args.candidate_id]
    return [candidate.candidate_id for candidate in candidates if candidate.executable]


def _load_sweep_inputs(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], float, Any]:
    library = ExperienceLibrary.load(args.universal_experience_lib)
    base_state = json.loads(args.sandbox_initial_state.read_text(encoding="utf-8"))
    parameter_profile = {}
    if args.sandbox_parameter_profile is not None:
        parameter_profile = json.loads(args.sandbox_parameter_profile.read_text(encoding="utf-8"))
    profile_ranges = parameter_profile.get("parameter_ranges") if isinstance(parameter_profile.get("parameter_ranges"), dict) else {}
    profile_pose_range = profile_ranges.get("object_pose_noise_xyz") if isinstance(profile_ranges, dict) else None
    object_pose_noise = args.object_pose_noise
    if isinstance(profile_pose_range, list) and len(profile_pose_range) >= 2:
        object_pose_noise = max(abs(float(profile_pose_range[0])), abs(float(profile_pose_range[1])))
    object_class = object_class_for_scenario(args.scenario)
    sandbox_calibration = select_sandbox_calibration(
        library,
        scenario=args.scenario,
        condition=args.condition,
        object_class=object_class,
    ) if args.use_sandbox_calibration else None
    return base_state, parameter_profile, profile_ranges, object_pose_noise, sandbox_calibration


def _evaluate_rollout_job(job: dict[str, Any]) -> dict[str, Any]:
    started = time.time()
    perturbation = sandbox_perturbation_from_dict(job["perturbation"])
    state = apply_perturbation_to_state(job["base_state"], perturbation)
    rollout = evaluate_candidate_in_sandbox(
        scenario=str(job["scenario"]),
        condition=str(job["condition"]),
        candidate_id=str(job["candidate_id"]),
        control_mode=str(job["control_mode"]),
        keyframe_dir=Path(job["keyframe_dir"]) if job.get("keyframe_dir") else None,
        trace_dir=Path(job["trace_dir"]) if job.get("trace_dir") else None,
        sandbox_calibration=job.get("sandbox_calibration") if isinstance(job.get("sandbox_calibration"), dict) else None,
        sandbox_initial_state=state,
    )
    rollout["rollout_index"] = perturbation.rollout_index
    rollout["perturbation"] = perturbation.to_dict()
    rollout["worker"] = {
        "success": True,
        "elapsed_s": round(time.time() - started, 4),
        "pid": os.getpid(),
    }
    return rollout


def _worker_main(args: argparse.Namespace) -> None:
    if args.worker_job is None or args.worker_output is None:
        raise RuntimeError("--worker-job and --worker-output are required for --worker-rollout")
    job = json.loads(args.worker_job.read_text(encoding="utf-8"))
    try:
        payload = _evaluate_rollout_job(job)
    except Exception as exc:
        payload = {
            "candidate_id": str(job.get("candidate_id") or ""),
            "rollout_index": int((job.get("perturbation") or {}).get("rollout_index") or -1),
            "decision": "reject",
            "critic_status": "block",
            "sandbox_score": 0.0,
            "critic_risk_score": 1.0,
            "critic_flags": ["sweep_worker_failed"],
            "failure_reason": f"sweep worker failed: {exc}",
            "perturbation": job.get("perturbation") or {},
            "worker": {
                "success": False,
                "error": str(exc),
                "pid": os.getpid(),
            },
        }
    _write_json(args.worker_output, payload)


def _rollout_job(
    *,
    args: argparse.Namespace,
    candidate_id: str,
    perturbation: Any,
    base_state: dict[str, Any],
    sandbox_calibration: Any,
    keyframe_dir: Path | None,
    trace_dir: Path | None,
    job_index: int,
) -> dict[str, Any]:
    return {
        "job_index": job_index,
        "scenario": args.scenario,
        "condition": args.condition,
        "control_mode": args.control_mode,
        "candidate_id": candidate_id,
        "perturbation": perturbation.to_dict(),
        "base_state": base_state,
        "sandbox_calibration": sandbox_calibration or {},
        "keyframe_dir": str(keyframe_dir) if keyframe_dir else "",
        "trace_dir": str(trace_dir) if trace_dir else "",
    }


def _run_job_subprocess(job: dict[str, Any], *, work_dir: Path) -> dict[str, Any]:
    job_path = work_dir / f"job_{int(job['job_index']):05d}.json"
    output_path = work_dir / f"out_{int(job['job_index']):05d}.json"
    _write_json(job_path, job)
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker-rollout",
        "--worker-job",
        str(job_path),
        "--worker-output",
        str(output_path),
    ]
    started = time.time()
    completed = subprocess.run(cmd, cwd=Path.cwd(), text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=None)
    if output_path.exists():
        payload = json.loads(output_path.read_text(encoding="utf-8"))
    else:
        payload = {
            "candidate_id": str(job.get("candidate_id") or ""),
            "rollout_index": int((job.get("perturbation") or {}).get("rollout_index") or -1),
            "decision": "reject",
            "critic_status": "block",
            "sandbox_score": 0.0,
            "critic_risk_score": 1.0,
            "critic_flags": ["sweep_worker_no_output"],
            "failure_reason": "sweep worker produced no output",
            "perturbation": job.get("perturbation") or {},
            "worker": {"success": False},
        }
    payload.setdefault("worker", {})
    payload["worker"].update({
        "subprocess_returncode": completed.returncode,
        "subprocess_elapsed_s": round(time.time() - started, 4),
        "stdout_tail": completed.stdout[-1000:],
        "stderr_tail": completed.stderr[-2000:],
    })
    if completed.returncode != 0:
        payload["worker"]["success"] = False
        payload.setdefault("critic_flags", []).append("sweep_worker_nonzero_exit")
        payload["failure_reason"] = payload.get("failure_reason") or f"sweep worker exited {completed.returncode}"
    return payload


def _build_jobs(
    *,
    args: argparse.Namespace,
    candidate_ids: list[str],
    perturbations: list[Any],
    base_state: dict[str, Any],
    sandbox_calibration: Any,
) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    for candidate_id in candidate_ids:
        for perturbation in perturbations:
            keyframe_dir = args.keyframe_dir / candidate_id / f"rollout_{perturbation.rollout_index:03d}" if args.keyframe_dir is not None else None
            trace_dir = args.trace_dir / candidate_id / f"rollout_{perturbation.rollout_index:03d}" if args.trace_dir is not None else None
            jobs.append(_rollout_job(
                args=args,
                candidate_id=candidate_id,
                perturbation=perturbation,
                base_state=base_state,
                sandbox_calibration=sandbox_calibration,
                keyframe_dir=keyframe_dir,
                trace_dir=trace_dir,
                job_index=len(jobs),
            ))
    return jobs


def _run_jobs(jobs: list[dict[str, Any]], *, parallel_workers: int) -> list[dict[str, Any]]:
    if parallel_workers <= 1:
        return [_evaluate_rollout_job(job) for job in jobs]
    with tempfile.TemporaryDirectory(prefix="candidate_sandbox_sweep_workers_") as tmp:
        work_dir = Path(tmp)
        reports: list[dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=max(1, int(parallel_workers))) as pool:
            futures = [pool.submit(_run_job_subprocess, job, work_dir=work_dir) for job in jobs]
            for future in as_completed(futures):
                reports.append(future.result())
        return reports


def _stable_rollout_signature(rollout: dict[str, Any]) -> dict[str, Any]:
    return {
        "success": bool(rollout.get("success")),
        "task_success": bool(rollout.get("task_success")),
        "critic_status": str(rollout.get("critic_status") or ""),
        "decision": str(rollout.get("decision") or ""),
        "sandbox_score": round(float(rollout.get("sandbox_score") or 0.0), 4),
        "critic_risk_score": round(float(rollout.get("critic_risk_score") or 0.0), 4),
        "failure_reason": str(rollout.get("failure_reason") or ""),
    }


def _determinism_check(jobs: list[dict[str, Any]]) -> dict[str, Any]:
    if not jobs:
        return {"enabled": False, "pass": False, "reason": "no jobs"}
    nominal = sorted(jobs, key=lambda item: (str(item["candidate_id"]), int((item["perturbation"] or {}).get("rollout_index") or 0)))[0]
    first = _evaluate_rollout_job({**nominal, "job_index": 900001})
    second = _evaluate_rollout_job({**nominal, "job_index": 900002})
    first_sig = _stable_rollout_signature(first)
    second_sig = _stable_rollout_signature(second)
    return {
        "enabled": True,
        "pass": first_sig == second_sig,
        "candidate_id": str(nominal.get("candidate_id") or ""),
        "rollout_index": int((nominal.get("perturbation") or {}).get("rollout_index") or 0),
        "first_signature": first_sig,
        "second_signature": second_sig,
    }


def main() -> None:
    args = parse_args()
    if args.worker_rollout:
        _worker_main(args)
        return
    started = time.time()
    base_state, parameter_profile, profile_ranges, object_pose_noise, sandbox_calibration = _load_sweep_inputs(args)
    perturbations = generate_sandbox_perturbations(
        num_rollouts=args.num_rollouts,
        seed=args.seed,
        object_pose_noise=object_pose_noise,
        parameter_ranges=profile_ranges,
        include_nominal=True,
    )
    candidate_ids = _candidate_ids(args)
    jobs = _build_jobs(
        args=args,
        candidate_ids=candidate_ids,
        perturbations=perturbations,
        base_state=base_state,
        sandbox_calibration=sandbox_calibration,
    )

    rollout_reports_all = _run_jobs(jobs, parallel_workers=max(1, int(args.parallel_workers)))
    candidate_reports: list[dict[str, Any]] = []
    for candidate_id in candidate_ids:
        rollout_reports = sorted(
            [item for item in rollout_reports_all if str(item.get("candidate_id") or "") == candidate_id],
            key=lambda item: int(item.get("rollout_index") or 0),
        )
        summary = robust_sandbox_summary(rollout_reports)
        candidate_reports.append({
            "candidate_id": candidate_id,
            "summary": summary,
            "rollouts": rollout_reports,
        })

    ranked = sorted(
        candidate_reports,
        key=lambda item: (
            float(item["summary"].get("robust_sandbox_score") or 0.0),
            float(item["summary"].get("success_rate") or 0.0),
            str(item["candidate_id"]),
        ),
        reverse=True,
    )
    elapsed = time.time() - started
    rollout_count = sum(len(item["rollouts"]) for item in ranked)
    worker_elapsed = [
        float((rollout.get("worker") or {}).get("elapsed_s") or (rollout.get("worker") or {}).get("subprocess_elapsed_s") or 0.0)
        for rollout in rollout_reports_all
    ]
    failed_worker_count = sum(1 for rollout in rollout_reports_all if (rollout.get("worker") or {}).get("success") is False)
    determinism = _determinism_check(jobs) if args.determinism_check else {"enabled": False, "pass": None}
    report = {
        "scenario": args.scenario,
        "condition": args.condition,
        "control_mode": args.control_mode,
        "experience_library": str(args.universal_experience_lib),
        "sandbox_initial_state": {
            "path": str(args.sandbox_initial_state),
            "source_episode_id": str(base_state.get("source_episode_id") or ""),
            "confidence": float(base_state.get("confidence") or 0.0),
            "missing_fields": list(base_state.get("missing_fields") or []),
        },
        "sweep_config": {
            "num_rollouts": len(perturbations),
            "seed": args.seed,
            "object_pose_noise": args.object_pose_noise,
            "effective_object_pose_noise": object_pose_noise,
            "sandbox_parameter_profile": {
                "enabled": bool(parameter_profile),
                "path": str(args.sandbox_parameter_profile) if args.sandbox_parameter_profile else "",
                "profile_id": str(parameter_profile.get("profile_id") or ""),
                "confidence": float(parameter_profile.get("confidence") or 0.0) if parameter_profile else 0.0,
                "expected_failure_modes": list(parameter_profile.get("expected_failure_modes") or []) if parameter_profile else [],
                "parameter_ranges": profile_ranges if parameter_profile else {},
            },
            "include_risky_candidates": args.include_risky_candidates,
            "sandbox_calibration_enabled": args.use_sandbox_calibration,
            "sandbox_calibration": sandbox_calibration or {},
            "parallel_workers": max(1, int(args.parallel_workers)),
            "parallel_mode": "subprocess" if int(args.parallel_workers) > 1 else "serial",
            "determinism_check_enabled": bool(args.determinism_check),
        },
        "selected_candidate_id": ranked[0]["candidate_id"] if ranked else "",
        "candidate_count": len(ranked),
        "rollout_count": rollout_count,
        "elapsed_s": round(elapsed, 4),
        "rollouts_per_minute": round((rollout_count / elapsed) * 60.0, 4) if elapsed > 0 else 0.0,
        "parallel_worker_count": max(1, int(args.parallel_workers)),
        "failed_worker_count": failed_worker_count,
        "mean_rollout_time_s": round(sum(worker_elapsed) / len(worker_elapsed), 4) if worker_elapsed else 0.0,
        "determinism_check": determinism,
        "determinism_check_pass": determinism.get("pass"),
        "candidates": ranked,
    }
    _write_json(args.save, report)
    print(json.dumps({
        "selected_candidate_id": report["selected_candidate_id"],
        "candidate_count": report["candidate_count"],
        "rollout_count": report["rollout_count"],
        "rollouts_per_minute": report["rollouts_per_minute"],
        "parallel_worker_count": report["parallel_worker_count"],
        "failed_worker_count": report["failed_worker_count"],
        "determinism_check_pass": report["determinism_check_pass"],
        "save": str(args.save),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
