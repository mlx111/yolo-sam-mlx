"""UR5e-specific structured retrieval helpers."""

from __future__ import annotations

from typing import Any

try:
    from experience_system.experience_core import ExperienceLibrary, RetrievalQuery
except ModuleNotFoundError:  # pragma: no cover - script-root fallback
    from experience_core import ExperienceLibrary, RetrievalQuery

from .schema import UR5E_NAMESPACE, UR5E_ROBOT_TYPE


def build_ur5e_retrieval_query(**kwargs: Any) -> RetrievalQuery:
    payload = {
        "robot_type": UR5E_ROBOT_TYPE,
        "backend": "mujoco",
        "skill_namespace": UR5E_NAMESPACE,
        "include_failed": True,
        "top_k": 5,
    }
    payload.update(kwargs)
    return RetrievalQuery(**payload)


def query_ur5e_experiences(library: ExperienceLibrary, **kwargs: Any):
    query = build_ur5e_retrieval_query(**kwargs)
    return library.query_structured(query)
