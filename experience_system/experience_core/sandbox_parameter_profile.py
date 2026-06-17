"""Convert sim-real gap memories into sandbox parameter-sweep profiles."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from typing import Any

from .calibration import calibration_group_key, group_gap_entries
from .schema import ExperienceEntry, utc_now


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(float(value), hi))


def _num(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _profile_id(group_key: tuple[str, str, str, str], gap_ids: list[str]) -> str:
    payload = "|".join(group_key) + "|" + "|".join(sorted(gap_ids))
    return "sandbox_profile_" + hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]


@dataclass
class SandboxParameterProfile:
    profile_id: str = ""
    schema_version: str = "sandbox_parameter_profile_v1"
    created_at: str = ""
    group_key: dict[str, str] = field(default_factory=dict)
    source_gap_ids: list[str] = field(default_factory=list)
    parameter_ranges: dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    expected_failure_modes: list[str] = field(default_factory=list)
    evidence: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SandboxParameterProfile":
        fields = getattr(cls, "__dataclass_fields__", {})
        return cls(**{key: value for key, value in payload.items() if key in fields})


def build_sandbox_parameter_profile(
    group_entries: list[ExperienceEntry],
    *,
    group_key: tuple[str, str, str, str] | None = None,
) -> SandboxParameterProfile:
    if not group_entries:
        return SandboxParameterProfile()
    key = group_key or calibration_group_key(group_entries[0])
    source_gap_ids: list[str] = []
    evidence: list[dict[str, Any]] = []
    pose_errors: list[float] = []
    contact_mismatch_count = 0
    sim_success_real_fail_count = 0
    total_weight = 0.0
    expected_modes: set[str] = set()

    for entry in group_entries:
        gap = entry.sim_real_gap
        if not gap.gap_id or gap.gap_id in source_gap_ids:
            continue
        source_gap_ids.append(gap.gap_id)
        gap_score = _clamp(gap.gap_score, 0.0, 1.0)
        uncertainty = _clamp(gap.uncertainty, 0.0, 1.0)
        weight = _clamp(0.65 * gap_score + 0.35 * (1.0 - uncertainty), 0.0, 1.0)
        total_weight += weight
        pose_gap = gap.pose_gap if isinstance(gap.pose_gap, dict) else {}
        contact_gap = gap.contact_gap if isinstance(gap.contact_gap, dict) else {}
        outcome_gap = gap.outcome_gap if isinstance(gap.outcome_gap, dict) else {}
        pose_error = _num(pose_gap.get("object_pose_error"), 0.0)
        pose_errors.append(pose_error)
        contact_mismatch = bool(contact_gap.get("contact_mismatch"))
        if contact_mismatch:
            contact_mismatch_count += 1
            expected_modes.add("contact_mismatch")
            expected_modes.add("slip_or_grasp_instability")
        outcome_type = str(outcome_gap.get("type") or "")
        if outcome_type == "sim_success_real_fail":
            sim_success_real_fail_count += 1
            expected_modes.add("sim_success_real_fail")
            expected_modes.add("overoptimistic_sandbox_success")
        if pose_error > 0.05:
            expected_modes.add("object_pose_uncertainty")
        evidence.append({
            "experience_id": entry.experience_id,
            "gap_id": gap.gap_id,
            "gap_score": gap.gap_score,
            "uncertainty": gap.uncertainty,
            "weight": round(weight, 4),
            "object_pose_error": round(pose_error, 6),
            "contact_mismatch": contact_mismatch,
            "outcome_gap_type": outcome_type,
        })

    if not source_gap_ids:
        return SandboxParameterProfile(group_key={
            "robot_type": key[0],
            "scenario_id": key[1],
            "condition_id": key[2],
            "object_class": key[3],
        })

    avg_pose_error = sum(pose_errors) / max(len(pose_errors), 1)
    max_pose_error = max(pose_errors) if pose_errors else 0.0
    contact_mismatch_rate = contact_mismatch_count / len(source_gap_ids)
    sim_fail_rate = sim_success_real_fail_count / len(source_gap_ids)
    pose_noise = _clamp(max(0.015, min(max_pose_error * 0.18, 0.08)), 0.005, 0.08)
    friction_low = _clamp(1.0 - 0.45 * contact_mismatch_rate - 0.25 * sim_fail_rate, 0.35, 1.0)
    mass_low = _clamp(1.0 - 0.20 * sim_fail_rate, 0.60, 1.0)
    mass_high = _clamp(1.0 + 0.25 * sim_fail_rate + 0.10 * contact_mismatch_rate, 1.0, 1.5)
    delay_steps = int(round(8 * sim_fail_rate + 4 * contact_mismatch_rate))
    controller_gain_low = _clamp(1.0 - 0.20 * sim_fail_rate, 0.70, 1.0)
    contact_solref_high = _clamp(1.0 + 0.40 * contact_mismatch_rate + 0.25 * sim_fail_rate, 1.0, 1.8)
    contact_solimp_margin_high = _clamp(1.0 + 0.50 * contact_mismatch_rate + 0.25 * sim_fail_rate, 1.0, 2.0)
    confidence = _clamp((total_weight / max(len(source_gap_ids), 1)) * min(len(source_gap_ids), 5) / 5.0, 0.0, 1.0)

    return SandboxParameterProfile(
        profile_id=_profile_id(key, source_gap_ids),
        created_at=utc_now(),
        group_key={
            "robot_type": key[0],
            "scenario_id": key[1],
            "condition_id": key[2],
            "object_class": key[3],
        },
        source_gap_ids=source_gap_ids,
        parameter_ranges={
            "object_pose_noise_xyz": [-round(pose_noise, 6), round(pose_noise, 6)],
            "object_yaw_noise_deg": [-5.0, 5.0] if avg_pose_error > 0.05 else [0.0, 0.0],
            "friction_scale": [round(friction_low, 4), 1.05],
            "mass_scale": [round(mass_low, 4), round(mass_high, 4)],
            "actuation_delay_steps": [0, delay_steps],
            "controller_gain_scale": [round(controller_gain_low, 4), 1.0],
            "contact_solref_time_scale": [1.0, round(contact_solref_high, 4)],
            "contact_solimp_margin_scale": [1.0, round(contact_solimp_margin_high, 4)],
            "gripper_closure_bias": [-round(0.02 * contact_mismatch_rate, 4), 0.0],
            "contact_success_bias": round(_clamp(-0.8 * contact_mismatch_rate - 0.7 * sim_fail_rate, -1.0, 1.0), 4),
            "slip_risk_bias": round(_clamp(0.5 * contact_mismatch_rate + 0.7 * sim_fail_rate, 0.0, 1.0), 4),
        },
        confidence=round(confidence, 4),
        expected_failure_modes=sorted(expected_modes),
        evidence=evidence,
    )


def build_group_sandbox_parameter_profiles(entries: list[ExperienceEntry]) -> dict[tuple[str, str, str, str], SandboxParameterProfile]:
    return {
        key: build_sandbox_parameter_profile(group_entries, group_key=key)
        for key, group_entries in group_gap_entries(entries).items()
    }
