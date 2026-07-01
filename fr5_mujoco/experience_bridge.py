"""FR5 bridge between runtime experiments and MemoryV3 retrieval."""

from __future__ import annotations

from typing import Any

from experience_system.memory.v3 import canonical_action_signature_from_steps
from skills import registry


class FR5ExperienceBridge:
    def __init__(self, experiment: Any) -> None:
        self.experiment = experiment

    def anomaly_state(self) -> dict[str, Any]:
        exp = self.experiment
        return {
            "condition_id": exp.condition_id,
            "scenario_id": exp.scenario_id,
            "available_actions": sorted(registry.allowed_actions(exp.scenario_id)),
            "robot": "fr5",
            "target_class": exp.target_class,
            "place_target": exp.place_target,
            "grasp_orientation_policy": "fixed_vertical_down",
        }

    def retrieval_key(self, anomaly_state: dict[str, Any] | None = None) -> dict[str, Any]:
        exp = self.experiment
        steps = (
            exp.metrics.get("executed_recovery_steps")
            or exp.metrics.get("llm_recovery_steps")
            or (exp.recovery_plan or {}).get("steps", [])
        )
        return {
            "condition_id": exp.condition_id,
            "scenario_id": exp.scenario_id,
            "plan_signature": canonical_action_signature_from_steps(steps),
            "robot": "fr5",
            "target_class": exp.target_class,
            "place_target": exp.place_target,
            "grasp_orientation_policy": "fixed_vertical_down",
            **(anomaly_state or {}),
        }

    def query_recovery_experiences(
        self,
        *,
        top_k: int,
        diversity_lambda: float,
        include_failed: bool = True,
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
            task_stage="direct_recovery",
            text_summary=(
                f"robot=fr5; target={exp.target_class}; place_target={exp.place_target}; "
                "fixed_vertical_grasp=true"
            ),
            top_k=top_k,
            include_failed=include_failed,
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
            print(f"  [WARN] memory consolidation failed: {exc}")
        if exp.experience_lib_path is not None:
            exp.experience_lib_path.parent.mkdir(parents=True, exist_ok=True)
            library.save(exp.experience_lib_path)
        return entry
