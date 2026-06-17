"""Lightweight text-semantic retrieval over structured experience fields."""

from __future__ import annotations

import re
from collections import Counter
from typing import Any

from .schema import ExperienceEntry


TOKEN_RE = re.compile(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]+")


def tokens(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(text) if token.strip()]


def nonempty(value: Any) -> bool:
    if value in (None, "", [], {}):
        return False
    if isinstance(value, dict):
        return any(nonempty(item) for item in value.values())
    if isinstance(value, list):
        return any(nonempty(item) for item in value)
    return True


def normalize_text(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, dict):
        items = []
        for key in sorted(value):
            item = value[key]
            if nonempty(item):
                items.append(f"{key}:{normalize_text(item)}")
        return " ".join(items)
    if isinstance(value, list):
        return " ".join(normalize_text(item) for item in value if nonempty(item))
    return str(value or "")


def _compact_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def semantic_summary(entry: ExperienceEntry) -> str:
    """Build a compact text summary from explicit experience fields."""

    segments = [
        normalize_text(entry.source),
        normalize_text(entry.backend),
        normalize_text(entry.scenario_id),
        normalize_text(entry.condition_id),
        normalize_text(entry.task.get("name", "")),
        normalize_text(entry.task.get("stage", "")),
        normalize_text(entry.anomaly.get("type", "")),
        normalize_text(entry.anomaly.get("description", "")),
        normalize_text(entry.object_state.object_class),
        normalize_text(entry.object_state.target_object),
        normalize_text(entry.failure_taxonomy.get("failure_type", "")),
        normalize_text(entry.failure_taxonomy.get("standard_failure_type", "")),
        normalize_text(entry.critic_result.overall_status),
        normalize_text(entry.critic_result.feedback_for_rewrite),
        normalize_text(entry.sim_real_gap.outcome_gap.get("type", "")),
        normalize_text(entry.sim_real_gap.gap_id),
        normalize_text(entry.memory_tags.get("memory_role", "")),
        normalize_text(entry.memory_tags.get("memory_type", "")),
        normalize_text(entry.memory_tags.get("memory_scope", "")),
        normalize_text(entry.memory_gate.write_decision),
        normalize_text(entry.retrieval_key.get("plan_signature", "")),
        normalize_text(_compact_dict(entry.execution_feedback)),
        normalize_text(_compact_dict(entry.result)),
    ]
    segments.extend(normalize_text(flag) for flag in entry.critic_result.rule_flags if nonempty(flag))
    segments.extend(normalize_text(skill.name) for skill in entry.skill_sequence if skill.name)
    return " ".join(segment for segment in segments if segment)


def semantic_query_text(
    *,
    scenario: str = "",
    condition: str = "",
    object_class: str = "",
    candidate_id: str = "",
    candidate_description: str = "",
    candidate_steps: list[str] | None = None,
    task_stage: str = "task_chain",
    failure_type: str = "",
    critic_status: str = "",
    memory_role: str = "",
) -> str:
    parts = [
        scenario,
        condition,
        object_class,
        candidate_id,
        candidate_description,
        task_stage,
        failure_type,
        critic_status,
        memory_role,
        " ".join(candidate_steps or []),
    ]
    return " ".join(normalize_text(part) for part in parts if nonempty(part))


def faiss_tfidf_available() -> tuple[bool, str]:
    try:
        import faiss  # noqa: F401
        import numpy  # noqa: F401
        import sklearn  # noqa: F401
    except Exception as exc:  # pragma: no cover - environment-dependent fallback
        return False, f"{type(exc).__name__}: {exc}"
    return True, ""


def token_overlap_scores(entries: list[ExperienceEntry], query_text: str, *, top_k: int = 10) -> dict[str, float]:
    query_tokens = set(tokens(query_text))
    if not query_tokens:
        return {}
    scored: list[tuple[float, str]] = []
    for entry in entries:
        entry_tokens = set(tokens(semantic_summary(entry)))
        if not entry_tokens:
            continue
        overlap = len(query_tokens & entry_tokens)
        if overlap <= 0:
            continue
        score = overlap / max(len(query_tokens | entry_tokens), 1)
        scored.append((round(float(score), 4), entry.experience_id))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return {experience_id: score for score, experience_id in scored[: max(int(top_k), 0)]}


class TextSemanticRetrievalIndex:
    """In-memory TF-IDF + FAISS index for auxiliary semantic retrieval."""

    def __init__(self, entries: list[ExperienceEntry], *, backend: str = "auto") -> None:
        self.entries = list(entries)
        self.backend = "token_overlap"
        self.fallback_reason = ""
        self.faiss_metadata: dict[str, Any] = {}
        self._summaries = {entry.experience_id: semantic_summary(entry) for entry in self.entries}
        self._vectorizer: Any = None
        self._index: Any = None
        self._row_to_entry: dict[int, ExperienceEntry] = {}
        self._build(backend)

    def _build(self, backend: str) -> None:
        if backend == "token_overlap":
            self.backend = "token_overlap"
            return
        ok, reason = faiss_tfidf_available()
        if backend == "faiss" and not ok:
            raise RuntimeError(f"FAISS TF-IDF backend requested but unavailable: {reason}")
        if not ok:
            self.backend = "token_overlap"
            self.fallback_reason = reason
            return
        if not self.entries:
            self.backend = "faiss_tfidf"
            self.faiss_metadata = {"dimension": 0, "indexed_entry_count": 0}
            return
        import faiss
        import numpy as np
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.preprocessing import normalize

        texts = [self._summaries[entry.experience_id] for entry in self.entries]
        vectorizer = TfidfVectorizer(token_pattern=r"(?u)\b[\w\-]+\b", lowercase=True, norm=None)
        matrix = vectorizer.fit_transform(texts)
        matrix = normalize(matrix, norm="l2", copy=False)
        dense = matrix.astype("float32").toarray()
        if dense.shape[1] == 0:
            self.backend = "token_overlap"
            self.fallback_reason = "empty TF-IDF vocabulary"
            return
        index = faiss.IndexFlatIP(int(dense.shape[1]))
        index.add(np.ascontiguousarray(dense, dtype=np.float32))
        self.backend = "faiss_tfidf"
        self._vectorizer = vectorizer
        self._index = index
        self._row_to_entry = {index: entry for index, entry in enumerate(self.entries)}
        self.faiss_metadata = {"dimension": int(dense.shape[1]), "indexed_entry_count": int(dense.shape[0])}

    def search_scores(self, query_text: str, *, top_k: int = 10) -> dict[str, float]:
        if self.backend != "faiss_tfidf" or self._vectorizer is None or self._index is None:
            return token_overlap_scores(self.entries, query_text, top_k=top_k)
        import numpy as np
        from sklearn.preprocessing import normalize

        query_matrix = self._vectorizer.transform([query_text])
        query_matrix = normalize(query_matrix, norm="l2", copy=False)
        query_vector = np.ascontiguousarray(query_matrix.astype("float32").toarray(), dtype=np.float32)
        search_k = min(max(int(top_k), 1), len(self._row_to_entry))
        distances, indexes = self._index.search(query_vector, search_k)
        scored: list[tuple[float, str]] = []
        for score, row in zip(distances[0], indexes[0]):
            if int(row) < 0 or float(score) <= 0.0:
                continue
            scored.append((round(float(score), 4), self._row_to_entry[int(row)].experience_id))
        scored.sort(key=lambda item: (-item[0], item[1]))
        return {experience_id: score for score, experience_id in scored[: max(int(top_k), 0)]}

    def statistics(self, *, query_count: int = 0, semantic_hit_count: int = 0) -> dict[str, Any]:
        token_counts = [len(tokens(text)) for text in self._summaries.values()]
        return {
            "retrieval_backend": self.backend,
            "backend_fallback_reason": self.fallback_reason,
            "faiss_index": self.faiss_metadata,
            "entry_count": len(self.entries),
            "semantic_summary_nonempty_count": sum(1 for text in self._summaries.values() if text.strip()),
            "avg_token_count": round(sum(token_counts) / len(token_counts), 4) if token_counts else 0.0,
            "semantic_signal_rate": round(semantic_hit_count / query_count, 4) if query_count else 0.0,
            "source_distribution": dict(Counter(entry.source for entry in self.entries)),
            "scenario_distribution": dict(Counter(entry.scenario_id for entry in self.entries)),
            "condition_distribution": dict(Counter(entry.condition_id for entry in self.entries)),
        }


def build_semantic_scores(
    entries: list[ExperienceEntry],
    *,
    query_text: str,
    top_k: int = 10,
    backend: str = "auto",
) -> tuple[dict[str, float], dict[str, Any]]:
    index = TextSemanticRetrievalIndex(entries, backend=backend)
    scores = index.search_scores(query_text, top_k=top_k)
    return scores, index.statistics(query_count=1, semantic_hit_count=int(bool(scores)))
