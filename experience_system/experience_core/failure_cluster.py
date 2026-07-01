"""Failure clustering for universal/Galaxea experience libraries.

Shared failure clustering helpers used by the experience system.
interface-compatible with wrapper1 ``FailureClusterer``.
"""

from __future__ import annotations

import json
import os
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
    """Cluster failure entries by semantic similarity of failure descriptions."""

    def __init__(self, model_name: str | None = None) -> None:
        self._model_name = model_name or _CLUSTER_MODEL_NAME
        self._device = _detect_device()
        self._model: Any = None
        self._centroids: dict[str, np.ndarray] = {}
        print(f"  [FailureClusterer] device={self._device} model={self._model_name}")

    def _get_model(self) -> Any:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self._model_name, device=self._device)
        return self._model

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 384), dtype=np.float32)
        model = self._get_model()
        return model.encode(texts, convert_to_numpy=True, show_progress_bar=False)

    def cluster(self, entries: list[Any]) -> dict[str, list[Any]]:
        texts = [self._entry_text(entry) for entry in entries]
        emb = self.embed(texts)
        clustering = DBSCAN(eps=_EPS, min_samples=_MIN_SAMPLES, metric="cosine").fit(emb)

        groups: dict[str, list[Any]] = {}
        for index, label in enumerate(clustering.labels_):
            cid = f"fc{label}" if label >= 0 else "noise"
            groups.setdefault(cid, []).append(entries[index])
            self._set_cluster_id(entries[index], cid)

        self._centroids = {}
        for label in set(clustering.labels_):
            if label < 0:
                continue
            mask = clustering.labels_ == label
            self._centroids[f"fc{label}"] = emb[mask].mean(axis=0)
        return groups

    def assign_new(self, entry: Any, existing_entries: list[Any]) -> str:
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
        if best_sim < 0.5:
            best_cid = "noise"
        self._set_cluster_id(entry, best_cid)
        return best_cid

    def save(self, path: str | Path) -> None:
        data = {
            "centroids": {key: value.tolist() for key, value in self._centroids.items()},
            "model_name": self._model_name,
            "device": self._device,
        }
        Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    def load(self, path: str | Path) -> None:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        self._centroids = {key: np.array(value, dtype=np.float32) for key, value in data.get("centroids", {}).items()}
        self._model_name = data.get("model_name", _CLUSTER_MODEL_NAME)
        self._device = data.get("device", self._device)

    @staticmethod
    def _entry_text(entry: Any) -> str:
        ft = getattr(entry, "failure_taxonomy", {}) or {}
        result = getattr(entry, "result", {}) or {}
        failure_reason = result.get("failure_reason", "") if isinstance(result, dict) else getattr(result, "failure_reason", "")
        return str(
            ft.get("corrective_direction")
            or ft.get("critic_root_cause")
            or ft.get("failure_type")
            or failure_reason
            or ""
        )

    @staticmethod
    def _set_cluster_id(entry: Any, cid: str) -> None:
        if isinstance(entry, dict):
            ft = entry.setdefault("failure_taxonomy", {})
            ft["cluster_id"] = cid
            return
        ft = getattr(entry, "failure_taxonomy", {}) or {}
        ft["cluster_id"] = cid
        entry.failure_taxonomy = ft
