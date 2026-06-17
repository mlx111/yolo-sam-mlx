"""Visual keyframe retrieval for universal experience memory."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_VISUAL_MODEL = os.getenv("VISUAL_RETRIEVAL_MODEL", "openai/clip-vit-base-patch32")


def detect_visual_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    try:
        if torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def image_paths_from_entry(entry: Any, base_dir: str | Path | None = None) -> list[str]:
    keyframes = getattr(entry, "keyframes", None) or []
    if not keyframes and hasattr(entry, "metadata"):
        keyframes = (entry.metadata or {}).get("keyframes") or []

    paths: list[str] = []
    for frame in keyframes:
        raw = (frame.get("image_path") if isinstance(frame, dict) else getattr(frame, "image_path", None)) or ""
        if not raw:
            continue
        path = Path(str(raw))
        if path.is_absolute() and path.exists():
            paths.append(str(path.resolve()))
            continue
        if base_dir is not None:
            resolved = Path(base_dir).resolve() / path
            if resolved.exists():
                paths.append(str(resolved))
    return paths


class VisualRetrievalIndex:
    def __init__(self, model_name: str | None = None, dim: int = 512, device: str | None = None) -> None:
        self.model_name = model_name or DEFAULT_VISUAL_MODEL
        self.dim = int(dim)
        self.device = device or detect_visual_device()
        self._model: Any = None
        self._processor: Any = None
        self._index: Any = None
        self._id_to_eid: dict[int, str] = {}
        self._eid_to_ids: dict[str, list[int]] = {}
        self._next_id = 0

    @property
    def size(self) -> int:
        return int(self._index.ntotal) if self._index is not None else 0

    def _get_model(self) -> Any:
        if self._model is None:
            from transformers import CLIPModel

            self._model = CLIPModel.from_pretrained(self.model_name).to(self.device).eval()
        return self._model

    def _get_processor(self) -> Any:
        if self._processor is None:
            from transformers import CLIPProcessor

            self._processor = CLIPProcessor.from_pretrained(self.model_name)
        return self._processor

    def _ensure_index(self) -> None:
        if self._index is not None:
            return
        import faiss

        self._index = faiss.IndexIDMap(faiss.IndexFlatIP(self.dim))

    def embed(self, image_paths: list[str]) -> np.ndarray:
        import torch
        from PIL import Image

        if not image_paths:
            return np.zeros((0, self.dim), dtype=np.float32)

        images = []
        for path in image_paths:
            try:
                images.append(Image.open(path).convert("RGB"))
            except Exception:
                continue
        if not images:
            return np.zeros((0, self.dim), dtype=np.float32)

        processor = self._get_processor()
        model = self._get_model()
        inputs = processor(images=images, return_tensors="pt").to(self.device)
        with torch.no_grad():
            embedding = model.get_image_features(**inputs)
        return embedding.detach().cpu().numpy().astype(np.float32)

    def add(self, experience_id: str, image_paths: list[str]) -> int:
        import faiss

        self._ensure_index()
        embedding = self.embed(image_paths)
        if embedding.shape[0] == 0:
            return 0
        faiss.normalize_L2(embedding)
        ids = np.arange(self._next_id, self._next_id + embedding.shape[0], dtype=np.int64)
        self._index.add_with_ids(embedding, ids)
        self._id_to_eid.update({int(index): experience_id for index in ids})
        self._eid_to_ids.setdefault(experience_id, []).extend(int(index) for index in ids)
        self._next_id += int(embedding.shape[0])
        return int(embedding.shape[0])

    def search(self, query_image_paths: list[str], top_k: int = 5) -> list[tuple[str, float]]:
        import faiss

        if self._index is None or self._index.ntotal <= 0:
            return []
        query_embedding = self.embed(query_image_paths)
        if query_embedding.shape[0] == 0:
            return []
        faiss.normalize_L2(query_embedding)
        distances, index_ids = self._index.search(query_embedding, max(int(top_k), 1))

        scores: dict[str, float] = {}
        for row_distances, row_ids in zip(distances, index_ids):
            for distance, index_id in zip(row_distances, row_ids):
                if int(index_id) < 0:
                    continue
                experience_id = self._id_to_eid.get(int(index_id), "")
                if not experience_id:
                    continue
                scores[experience_id] = max(scores.get(experience_id, 0.0), float(distance))
        return sorted(
            [(experience_id, max(0.0, min(1.0, score))) for experience_id, score in scores.items()],
            key=lambda item: (-item[1], item[0]),
        )[: max(int(top_k), 0)]

    def save(self, index_dir: str | Path) -> None:
        import faiss

        output = Path(index_dir)
        output.mkdir(parents=True, exist_ok=True)
        if self._index is not None:
            faiss.write_index(self._index, str(output / "visual_index.faiss"))
        mapping = {
            "model_name": self.model_name,
            "dim": self.dim,
            "next_id": self._next_id,
            "id_to_eid": {str(key): value for key, value in self._id_to_eid.items()},
            "eid_to_ids": self._eid_to_ids,
        }
        (output / "visual_mapping.json").write_text(json.dumps(mapping, indent=2, ensure_ascii=False), encoding="utf-8")

    def load(self, index_dir: str | Path) -> None:
        import faiss

        source = Path(index_dir)
        faiss_path = source / "visual_index.faiss"
        mapping_path = source / "visual_mapping.json"
        if faiss_path.exists():
            self._index = faiss.read_index(str(faiss_path))
        if mapping_path.exists():
            mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
            self.model_name = str(mapping.get("model_name") or self.model_name)
            self.dim = int(mapping.get("dim") or self.dim)
            self._next_id = int(mapping.get("next_id") or 0)
            self._id_to_eid = {int(key): str(value) for key, value in dict(mapping.get("id_to_eid") or {}).items()}
            self._eid_to_ids = {
                str(key): [int(item) for item in value]
                for key, value in dict(mapping.get("eid_to_ids") or {}).items()
                if isinstance(value, list)
            }
