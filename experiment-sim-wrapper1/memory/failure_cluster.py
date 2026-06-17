"""Failure clustering for experience library.

Provides embedding-based clustering of failure entries so that ``_critic_prefilter``
can deduplicate by semantic cluster rather than exact ``failure_type`` string.

Device detection: CUDA > MPS > CPU (auto).
"""

from __future__ import annotations

import json
import os
import pickle
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.cluster import DBSCAN

_CLUSTER_MODEL_NAME = os.getenv(
    "FAILURE_CLUSTER_MODEL",
    "paraphrase-multilingual-MiniLM-L12-v2",
)
_EPS = float(os.getenv("FAILURE_CLUSTER_EPS", "0.25"))
_MIN_SAMPLES = int(os.getenv("FAILURE_CLUSTER_MIN_SAMPLES", "2"))


def _detect_device() -> str:
    import torch
    if torch.cuda.is_available():
        return "cuda"
    try:
        if torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


class FailureClusterer:
    """Cluster failure entries by semantic similarity of their failure descriptions.

    Usage:
        clusterer = FailureClusterer()
        clusterer.cluster(entries)           # batch cluster all entries
        clusterer.assign_new(entry, entries) # or incremental for a single new entry
    """

    def __init__(self, model_name: str | None = None) -> None:
        self._model_name = model_name or _CLUSTER_MODEL_NAME
        self._device = _detect_device()
        self._model: Any = None  # lazy load
        self._centroids: dict[str, np.ndarray] = {}  # cluster_id -> centroid embedding
        print(f"  [FailureClusterer] device={self._device} model={self._model_name}")

    # ── model lazy load ──────────────────────────────────────────────

    def _get_model(self) -> Any:
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(
                self._model_name,
                device=self._device,
            )
        return self._model

    # ── embedding ────────────────────────────────────────────────────

    def embed(self, texts: list[str]) -> np.ndarray:
        """Return (N, D) float32 array."""
        if not texts:
            return np.zeros((0, 384), dtype=np.float32)
        model = self._get_model()
        return model.encode(texts, convert_to_numpy=True, show_progress_bar=False)

    # ── clustering ───────────────────────────────────────────────────

    def cluster(self, entries: list[Any]) -> dict[str, list[Any]]:
        """Run DBSCAN on failure entries and assign ``cluster_id``.

        Returns a dict mapping cluster_id → list of entries.
        Noise entries get cluster_id ``"noise"``.
        """
        texts = [self._entry_text(e) for e in entries]
        emb = self.embed(texts)
        clustering = DBSCAN(eps=_EPS, min_samples=_MIN_SAMPLES, metric="cosine").fit(emb)

        groups: dict[str, list[Any]] = {}
        for i, label in enumerate(clustering.labels_):
            cid = f"fc{label}" if label >= 0 else "noise"
            groups.setdefault(cid, []).append(entries[i])
            if hasattr(entries[i], "cluster_id") or isinstance(entries[i], dict):
                self._set_cluster_id(entries[i], cid)

        # record centroids for incremental assignment
        self._centroids = {}
        for label in set(clustering.labels_):
            if label < 0:
                continue
            mask = clustering.labels_ == label
            self._centroids[f"fc{label}"] = emb[mask].mean(axis=0)

        return groups

    def assign_new(self, entry: Any, existing_entries: list[Any]) -> str:
        """Assign a single new entry to the nearest existing cluster (or noise)."""
        if not existing_entries or not self._centroids:
            self._set_cluster_id(entry, "noise")
            return "noise"

        text = self._entry_text(entry)
        emb = self.embed([text])[0]

        best_cid = "noise"
        best_sim = -1.0
        for cid, centroid in self._centroids.items():
            sim = float(np.dot(emb, centroid) / (np.linalg.norm(emb) * np.linalg.norm(centroid) + 1e-10))
            if sim > best_sim:
                best_sim = sim
                best_cid = cid

        if best_sim < 0.5:  # too far from any cluster → noise
            best_cid = "noise"

        self._set_cluster_id(entry, best_cid)
        return best_cid

    # ── persistence ──────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        data = {
            "centroids": {k: v.tolist() for k, v in self._centroids.items()},
            "model_name": self._model_name,
            "device": self._device,
        }
        with open(path, "w") as f:
            json.dump(data, f)

    def load(self, path: str | Path) -> None:
        with open(path) as f:
            data = json.load(f)
        self._centroids = {k: np.array(v, dtype=np.float32) for k, v in data.get("centroids", {}).items()}
        self._model_name = data.get("model_name", _CLUSTER_MODEL_NAME)
        self._device = data.get("device", self._device)

    # ── helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _entry_text(entry: Any) -> str:
        """Get the text used for clustering from an entry."""
        ft = getattr(entry, "failure_taxonomy", {}) or {}
        return str(
            ft.get("corrective_direction")
            or ft.get("failure_type")
            or getattr(getattr(entry, "result", None), "failure_reason", "")
            or ""
        )

    @staticmethod
    def _set_cluster_id(entry: Any, cid: str) -> None:
        if isinstance(entry, dict):
            ft = entry.setdefault("failure_taxonomy", {})
            ft["cluster_id"] = cid
        else:
            ft = getattr(entry, "failure_taxonomy", {}) or {}
            ft["cluster_id"] = cid
            entry.failure_taxonomy = ft
