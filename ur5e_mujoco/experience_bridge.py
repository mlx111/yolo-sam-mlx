"""Bridge between wrapper1 experiments and experience_system-style memory APIs."""

from __future__ import annotations

from typing import Any

from experience_system.memory.v3 import canonical_action_signature_from_steps
from skills import registry


class Wrapper1ExperienceBridge:
    """Thin adapter that keeps memory interactions out of skill code."""

    def __init__(self, experiment: Any):
        self.experiment = experiment

    def anomaly_state(self) -> dict[str, Any]:
        exp = self.experiment
        return {
            "condition_id": exp.condition_id,
            "scenario_id": exp.scenario_id,
            "available_actions": sorted(registry.allowed_actions(exp.scenario_id)),
            "grasp_orientation_policy": exp.metrics.get("grasp_orientation_policy", "fixed_downward_no_graspnet"),
        }

    def retrieval_key(self, anomaly_state: dict[str, Any] | None = None) -> dict[str, Any]:
        exp = self.experiment
        plan_signature = canonical_action_signature_from_steps(
            exp.metrics.get("executed_recovery_steps")
            or exp.metrics.get("llm_recovery_steps")
            or (exp.recovery_plan or {}).get("steps", [])
        )
        return {
            "condition_id": exp.condition_id,
            "scenario_id": exp.scenario_id,
            "plan_signature": plan_signature,
            "grasp_orientation_policy": exp.metrics.get("grasp_orientation_policy", "fixed_downward_no_graspnet"),
        }

    def query_recovery_experiences(
        self,
        *,
        top_k: int,
        diversity_lambda: float,
    ) -> list[tuple[object, float]]:
        exp = self.experiment
        library = getattr(exp, "experience_library", None)
        if library is None or len(library) <= 0:
            return []
        anomaly_state = self.anomaly_state()
        return library.query(
            scenario_id=exp.scenario_id,
            condition_id=exp.condition_id,
            available_actions=registry.allowed_actions(exp.scenario_id),
            anomaly_state=anomaly_state,
            retrieval_key=self.retrieval_key(anomaly_state),
            task_stage=exp.condition_spec.task_stage if exp.condition_spec else "",
            text_summary=f"condition_id={exp.condition_id}; scenario_id={exp.scenario_id}; fixed_vertical_grasp=true",
            top_k=top_k,
            diversity_lambda=diversity_lambda,
        )

    def save_entry(self, entry: object) -> object:
        exp = self.experiment
        library = getattr(exp, "experience_library", None)
        if library is None:
            return entry
        library.upsert(entry)
        try:
            library.consolidate()
        except Exception as exc:  # noqa: BLE001
            print(f"  [WARN] STM/LTM consolidation 失败: {exc}")
        if exp.experience_lib_path is not None:
            exp.experience_lib_path.parent.mkdir(parents=True, exist_ok=True)
            library.save(exp.experience_lib_path)
        return entry
