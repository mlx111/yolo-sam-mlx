"""Lightweight sandbox perturbation sweeps for robustness scoring."""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass
from typing import Any


def _round4(value: float) -> float:
    return round(float(value), 4)


def _quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int(round((len(ordered) - 1) * q))))
    return float(ordered[index])


@dataclass(frozen=True)
class SandboxPerturbation:
    rollout_index: int
    seed: int
    object_pose_noise_xyz: list[float]
    object_yaw_noise: float = 0.0
    friction_scale: float = 1.0
    mass_scale: float = 1.0
    grasp_offset_noise: list[float] | None = None
    gripper_closure_bias: float = 0.0
    actuation_delay_steps: int = 0
    controller_gain_scale: float = 1.0
    contact_solref_time_scale: float = 1.0
    contact_solimp_margin_scale: float = 1.0
    perception_noise_xyz: list[float] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "rollout_index": self.rollout_index,
            "seed": self.seed,
            "object_pose_noise_xyz": [_round4(item) for item in self.object_pose_noise_xyz],
            "object_yaw_noise": _round4(self.object_yaw_noise),
            "friction_scale": _round4(self.friction_scale),
            "mass_scale": _round4(self.mass_scale),
            "grasp_offset_noise": [_round4(item) for item in (self.grasp_offset_noise or [0.0, 0.0, 0.0])],
            "gripper_closure_bias": _round4(self.gripper_closure_bias),
            "actuation_delay_steps": int(self.actuation_delay_steps),
            "controller_gain_scale": _round4(self.controller_gain_scale),
            "contact_solref_time_scale": _round4(self.contact_solref_time_scale),
            "contact_solimp_margin_scale": _round4(self.contact_solimp_margin_scale),
            "perception_noise_xyz": [_round4(item) for item in (self.perception_noise_xyz or [0.0, 0.0, 0.0])],
        }


def sandbox_perturbation_from_dict(value: dict[str, Any]) -> SandboxPerturbation:
    """Reconstruct a perturbation from serialized sweep-worker JSON."""

    def floats(name: str, default: list[float]) -> list[float]:
        raw = value.get(name)
        if not isinstance(raw, list):
            return list(default)
        out = [float(item) for item in raw[: len(default)]]
        while len(out) < len(default):
            out.append(float(default[len(out)]))
        return out

    return SandboxPerturbation(
        rollout_index=int(value.get("rollout_index") or 0),
        seed=int(value.get("seed") or 0),
        object_pose_noise_xyz=floats("object_pose_noise_xyz", [0.0, 0.0, 0.0]),
        object_yaw_noise=float(value.get("object_yaw_noise") or 0.0),
        friction_scale=float(value.get("friction_scale") or 1.0),
        mass_scale=float(value.get("mass_scale") or 1.0),
        grasp_offset_noise=floats("grasp_offset_noise", [0.0, 0.0, 0.0]),
        gripper_closure_bias=float(value.get("gripper_closure_bias") or 0.0),
        actuation_delay_steps=int(value.get("actuation_delay_steps") or 0),
        controller_gain_scale=float(value.get("controller_gain_scale") or 1.0),
        contact_solref_time_scale=float(value.get("contact_solref_time_scale") or 1.0),
        contact_solimp_margin_scale=float(value.get("contact_solimp_margin_scale") or 1.0),
        perception_noise_xyz=floats("perception_noise_xyz", [0.0, 0.0, 0.0]),
    )


def generate_sandbox_perturbations(
    *,
    num_rollouts: int,
    seed: int = 0,
    object_pose_noise: float = 0.015,
    parameter_ranges: dict[str, Any] | None = None,
    include_nominal: bool = True,
) -> list[SandboxPerturbation]:
    rng = random.Random(seed)
    ranges = parameter_ranges if isinstance(parameter_ranges, dict) else {}

    def sample_range(name: str, default: float) -> float:
        value = ranges.get(name)
        if isinstance(value, list) and len(value) >= 2:
            return rng.uniform(float(value[0]), float(value[1]))
        return default

    def sample_int_range(name: str, default: int) -> int:
        value = ranges.get(name)
        if isinstance(value, list) and len(value) >= 2:
            return rng.randint(int(value[0]), int(value[1]))
        return default

    perturbations: list[SandboxPerturbation] = []
    for index in range(max(1, int(num_rollouts))):
        if index == 0 and include_nominal:
            xyz = [0.0, 0.0, 0.0]
        else:
            xyz = [rng.uniform(-object_pose_noise, object_pose_noise) for _ in range(3)]
            xyz[2] *= 0.35
        friction_scale = 1.0 if index == 0 and include_nominal else sample_range("friction_scale", 1.0)
        mass_scale = 1.0 if index == 0 and include_nominal else sample_range("mass_scale", 1.0)
        controller_gain_scale = 1.0 if index == 0 and include_nominal else sample_range("controller_gain_scale", 1.0)
        contact_solref_time_scale = 1.0 if index == 0 and include_nominal else sample_range("contact_solref_time_scale", 1.0)
        contact_solimp_margin_scale = 1.0 if index == 0 and include_nominal else sample_range("contact_solimp_margin_scale", 1.0)
        actuation_delay_steps = 0 if index == 0 and include_nominal else sample_int_range("actuation_delay_steps", 0)
        gripper_bias_range = ranges.get("gripper_closure_bias")
        gripper_bias = 0.0
        if not (index == 0 and include_nominal) and isinstance(gripper_bias_range, list) and len(gripper_bias_range) >= 2:
            gripper_bias = rng.uniform(float(gripper_bias_range[0]), float(gripper_bias_range[1]))
        perturbations.append(
            SandboxPerturbation(
                rollout_index=index,
                seed=seed + index,
                object_pose_noise_xyz=xyz,
                friction_scale=friction_scale,
                mass_scale=mass_scale,
                actuation_delay_steps=actuation_delay_steps,
                controller_gain_scale=controller_gain_scale,
                contact_solref_time_scale=contact_solref_time_scale,
                contact_solimp_margin_scale=contact_solimp_margin_scale,
                gripper_closure_bias=gripper_bias,
                perception_noise_xyz=list(xyz),
            )
        )
    return perturbations


def apply_perturbation_to_state(state: dict[str, Any], perturbation: SandboxPerturbation) -> dict[str, Any]:
    updated = copy.deepcopy(state or {})
    updated.setdefault("evidence", {})
    updated["evidence"]["sandbox_perturbation"] = perturbation.to_dict()
    updated["perturbation"] = perturbation.to_dict()

    for group_name in ("object_poses", "obstacle_poses"):
        group = updated.get(group_name)
        if not isinstance(group, dict):
            continue
        for pose in group.values():
            if not isinstance(pose, dict):
                continue
            position = pose.get("position")
            if not isinstance(position, list) or len(position) < 3:
                continue
            pose["position"] = [
                float(position[0]) + perturbation.object_pose_noise_xyz[0],
                float(position[1]) + perturbation.object_pose_noise_xyz[1],
                float(position[2]) + perturbation.object_pose_noise_xyz[2],
            ]
    return updated


def robust_sandbox_summary(rollouts: list[dict[str, Any]]) -> dict[str, Any]:
    if not rollouts:
        return {
            "num_rollouts": 0,
            "success_rate": 0.0,
            "critic_block_rate": 0.0,
            "critic_warn_rate": 0.0,
            "risk_score_mean": 0.0,
            "risk_score_p95": 0.0,
            "score_mean": 0.0,
            "score_p10": 0.0,
            "robust_sandbox_score": 0.0,
            "robust_decision": "reject",
            "worst_case_failure_reason": "no_rollouts",
        }

    scores = [float(item.get("sandbox_score") or 0.0) for item in rollouts]
    risks = [float(item.get("critic_risk_score") or 0.0) for item in rollouts]
    success_rate = sum(1 for item in rollouts if item.get("task_success") or item.get("success")) / len(rollouts)
    block_rate = sum(1 for item in rollouts if item.get("critic_status") == "block") / len(rollouts)
    warn_rate = sum(1 for item in rollouts if item.get("critic_status") == "warn") / len(rollouts)
    score_mean = sum(scores) / len(scores)
    risk_mean = sum(risks) / len(risks)
    risk_p95 = _quantile(risks, 0.95)
    score_p10 = _quantile(scores, 0.10)
    robust_score = (
        0.50 * score_mean
        + 0.25 * score_p10
        + 0.25 * success_rate
        - 0.30 * block_rate
        - 0.15 * warn_rate
    )
    robust_score = max(0.0, min(1.0, robust_score))
    if block_rate > 0.0 or robust_score < 0.40:
        decision = "reject"
    elif warn_rate > 0.0 or robust_score < 0.65:
        decision = "review"
    else:
        decision = "accept"
    worst = min(rollouts, key=lambda item: (float(item.get("sandbox_score") or 0.0), -float(item.get("critic_risk_score") or 0.0)))
    return {
        "num_rollouts": len(rollouts),
        "success_rate": _round4(success_rate),
        "critic_block_rate": _round4(block_rate),
        "critic_warn_rate": _round4(warn_rate),
        "risk_score_mean": _round4(risk_mean),
        "risk_score_p95": _round4(risk_p95),
        "score_mean": _round4(score_mean),
        "score_p10": _round4(score_p10),
        "robust_sandbox_score": _round4(robust_score),
        "robust_decision": decision,
        "worst_case_failure_reason": str(worst.get("failure_reason") or ""),
        "worst_case_rollout_index": int(worst.get("rollout_index", -1)),
    }
