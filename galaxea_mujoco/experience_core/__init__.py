"""Compatibility package for the migrated experience-memory core.

The canonical implementation now lives at ``../experience_system/experience_core``.
This package keeps historical ``import experience_core`` callers working from
inside ``galaxea_mujoco``.
"""

from __future__ import annotations

from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CANONICAL = _REPO_ROOT / "experience_system" / "experience_core"

if _CANONICAL.exists():
    __path__ = [str(_CANONICAL), *[path for path in __path__ if path != str(_CANONICAL)]]

from .calibration import apply_calibration_to_position, apply_sandbox_calibration, compute_group_calibrations, compute_sandbox_calibration, group_gap_entries
from .consolidation import consolidate_experiences, consolidation_key, should_consolidate
from .critic import apply_critic, build_critic_result, critique_experience
from .failure_taxonomy import STANDARD_FAILURE_TYPES, infer_standard_failure_type, is_actionable_failure_type, normalize_failure_type, standardize_failure_taxonomy
from .gating import compute_memory_gate
from .library import ExperienceLibrary
from .lifecycle import consolidate_memory_lifecycle, increment_retrieval_count, initialize_memory_lifecycle, memory_tier, retrieval_count, set_memory_tier, should_promote_to_ltm
from .lessons import adjust_candidate_with_lessons, load_lesson_library, normalize_lesson
from .policy_calibration import build_policy_risk_calibration, find_policy_group, load_policy_risk_calibration
from .quality import missing_entry_fields, validate_experience_entry, validate_experience_library, validate_raw_real_episode
from .retrieval import RetrievalMatch, RetrievalQuery, matches_to_tuples, retrieve_experiences
from .robot_plan_executor import DryRunSkillExecutor, RobotPlanExecutionReport, SkillExecutor, default_r1pro_skill_registry, execute_validated_robot_plan, validate_robot_plan_for_executor
from .runtime_scene import RuntimeObjectSpec, RuntimePlaceZoneSpec, RuntimeSandboxScene, render_runtime_scene_xml, write_runtime_scene
from .sandbox_parameter_profile import SandboxParameterProfile, build_group_sandbox_parameter_profiles, build_sandbox_parameter_profile
from .sandbox_state import SandboxInitialState, build_sandbox_initial_state, choose_sandbox_state_entry, coerce_sandbox_initial_state
from .sandbox_uncertainty import SandboxPerturbation, apply_perturbation_to_state, generate_sandbox_perturbations, robust_sandbox_summary
from .dual_source import apply_pair_and_gap, compute_sim_real_gap, pair_score, pair_sim_real_experiences
from .scoring import entry_risk_adjustment, score_candidate_plan, sensor_evidence_bonus
from .sensor_quality import enrich_memory_gate_with_sensor_quality, sensor_quality_report
from .sensor_gap import apply_sensor_sim_real_gaps, attach_sensor_sim_real_gap, derive_sensor_sim_real_gap
from .visual_retrieval import VisualRetrievalIndex, detect_visual_device, image_paths_from_entry
from .write_policy import apply_write_decision, should_write_entry
from .stage_retrieval import apply_stage_score_adjustment, run_stage_retrieval, summarize_stage_retrieval
from .stage_prompt import build_stage_planner_context, render_stage_prompt_text, summarize_stage_planner_contexts
from .schema import (
    CriticResult,
    ExperienceEntry,
    MemoryGate,
    ObjectState,
    RobotState,
    SandboxCalibration,
    SensorEvidence,
    SensorSummary,
    SimRealGap,
    SkillTraceItem,
)

__all__ = [
    "CriticResult",
    "DryRunSkillExecutor",
    "ExperienceEntry",
    "ExperienceLibrary",
    "MemoryGate",
    "ObjectState",
    "RetrievalMatch",
    "RetrievalQuery",
    "RobotPlanExecutionReport",
    "RobotState",
    "RuntimeObjectSpec",
    "RuntimePlaceZoneSpec",
    "RuntimeSandboxScene",
    "SandboxCalibration",
    "SandboxInitialState",
    "SandboxParameterProfile",
    "SandboxPerturbation",
    "SensorEvidence",
    "SensorSummary",
    "SimRealGap",
    "SkillTraceItem",
    "SkillExecutor",
    "STANDARD_FAILURE_TYPES",
    "VisualRetrievalIndex",
    "apply_calibration_to_position",
    "apply_pair_and_gap",
    "apply_sandbox_calibration",
    "apply_critic",
    "apply_write_decision",
    "apply_stage_score_adjustment",
    "apply_perturbation_to_state",
    "adjust_candidate_with_lessons",
    "build_critic_result",
    "build_policy_risk_calibration",
    "build_stage_planner_context",
    "build_sandbox_initial_state",
    "build_sandbox_parameter_profile",
    "build_group_sandbox_parameter_profiles",
    "compute_group_calibrations",
    "compute_memory_gate",
    "compute_sandbox_calibration",
    "compute_sim_real_gap",
    "coerce_sandbox_initial_state",
    "consolidate_experiences",
    "consolidate_memory_lifecycle",
    "consolidation_key",
    "critique_experience",
    "choose_sandbox_state_entry",
    "detect_visual_device",
    "default_r1pro_skill_registry",
    "entry_risk_adjustment",
    "execute_validated_robot_plan",
    "group_gap_entries",
    "generate_sandbox_perturbations",
    "find_policy_group",
    "infer_standard_failure_type",
    "is_actionable_failure_type",
    "image_paths_from_entry",
    "increment_retrieval_count",
    "initialize_memory_lifecycle",
    "load_policy_risk_calibration",
    "load_lesson_library",
    "matches_to_tuples",
    "memory_tier",
    "missing_entry_fields",
    "normalize_failure_type",
    "normalize_lesson",
    "pair_score",
    "pair_sim_real_experiences",
    "retrieve_experiences",
    "render_runtime_scene_xml",
    "render_stage_prompt_text",
    "retrieval_count",
    "run_stage_retrieval",
    "robust_sandbox_summary",
    "score_candidate_plan",
    "sensor_evidence_bonus",
    "apply_sensor_sim_real_gaps",
    "attach_sensor_sim_real_gap",
    "derive_sensor_sim_real_gap",
    "sensor_quality_report",
    "enrich_memory_gate_with_sensor_quality",
    "should_consolidate",
    "set_memory_tier",
    "should_promote_to_ltm",
    "should_write_entry",
    "summarize_stage_retrieval",
    "summarize_stage_planner_contexts",
    "standardize_failure_taxonomy",
    "validate_experience_entry",
    "validate_experience_library",
    "validate_raw_real_episode",
    "validate_robot_plan_for_executor",
    "write_runtime_scene",
]
