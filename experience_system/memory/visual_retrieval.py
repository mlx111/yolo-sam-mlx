"""Visual retrieval for experience library using CLIP + FAISS.

Provides embedding-based similarity search over keyframe images so that
the memory system can retrieve experiences with visually similar scenes
(e.g. similar object layout, lighting, occlusion pattern).

Device detection: CUDA > MPS > CPU (auto).
"""

from __future__ import annotations

import json
import os
import pickle
from pathlib import Path
from typing import Any

import numpy as np

_CLIP_MODEL_NAME = os.getenv("VISUAL_RETRIEVAL_MODEL", "openai/clip-vit-base-patch32")
_FAISS_INDEX_KEY = os.getenv("VISUAL_FAISS_INDEX_KEY", "Flat")


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


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _load_image(path: str | Path):
    """Return a PIL image from *path*."""
    from PIL import Image
    return Image.open(str(path)).convert("RGB")


def _image_paths_from_entry(entry: Any, base_dir: str | Path | None = None) -> list[str]:
    """Extract absolute keyframe image paths from an entry.

    ``entry.keyframes`` is a list of dicts with ``image_path`` (relative to
    the experience JSON).  If *base_dir* is given, paths are resolved
    relative to it.  Otherwise the entry's own metadata is consulted.
    """
    keyframes = getattr(entry, "keyframes", None) or []
    if not keyframes and hasattr(entry, "metadata"):
        keyframes = (entry.metadata or {}).get("keyframes") or []

    paths: list[str] = []
    for kf in keyframes:
        raw = (kf.get("image_path") if isinstance(kf, dict) else getattr(kf, "image_path", None)) or ""
        if not raw:
            continue
        p = Path(raw)
        if p.is_absolute() and p.exists():
            paths.append(str(p.resolve()))
        elif base_dir is not None:
            resolved = Path(base_dir).resolve() / p
            if resolved.exists():
                paths.append(str(resolved))
    return paths


# ---------------------------------------------------------------------------
# VisualRetrievalIndex
# ---------------------------------------------------------------------------

class VisualRetrievalIndex:
    """FAISS-based visual index for keyframe images.

    Usage:
        index = VisualRetrievalIndex()
        index.add("exp_001", ["/path/to/kf1.jpg", "/path/to/kf2.jpg"])
        results = index.search(["/path/to/query.jpg"], top_k=5)
        index.save("/path/to/faiss_index")
        index.load("/path/to/faiss_index")
    """

    def __init__(
        self,
        model_name: str | None = None,
        index_key: str | None = None,
        dim: int = 512,
    ) -> None:
        self._model_name = model_name or _CLIP_MODEL_NAME
        self._index_key = index_key or _FAISS_INDEX_KEY
        self._device = _detect_device()
        self._dim = dim
        self._model: Any = None  # lazy load
        self._processor: Any = None  # lazy load
        self._index: Any = None  # FAISS index
        self._id_to_eid: dict[int, str] = {}  # FAISS internal id -> experience_id
        self._eid_to_ids: dict[str, list[int]] = {}  # experience_id -> [FAISS ids]
        self._next_id: int = 0
        print(f"  [VisualRetrievalIndex] device={self._device} model={self._model_name}")

    # ── model lazy load ──────────────────────────────────────────────

    def _get_model(self) -> Any:
        if self._model is None:
            from transformers import CLIPModel
            self._model = CLIPModel.from_pretrained(self._model_name).to(self._device).eval()
        return self._model

    def _get_processor(self) -> Any:
        if self._processor is None:
            from transformers import CLIPProcessor
            self._processor = CLIPProcessor.from_pretrained(self._model_name)
        return self._processor

    # ── index lazy init ──────────────────────────────────────────────

    def _ensure_index(self) -> None:
        if self._index is not None:
            return
        import faiss
        self._index = faiss.IndexIDMap(faiss.IndexFlatIP(self._dim))

    # ── embedding ────────────────────────────────────────────────────

    def embed(self, image_paths: list[str]) -> np.ndarray:
        """Encode images to (N, D) float32 CLIP embeddings."""
        import torch
        if not image_paths:
            return np.zeros((0, self._dim), dtype=np.float32)
        model = self._get_model()
        processor = self._get_processor()
        images = []
        valid_indices: list[int] = []
        for i, p in enumerate(image_paths):
            try:
                images.append(_load_image(p))
                valid_indices.append(i)
            except Exception as exc:
                print(f"  [WARN] 加载图像失败 {p}: {exc}")
        if not images:
            return np.zeros((0, self._dim), dtype=np.float32)
        inputs = processor(images=images, return_tensors="pt").to(self._device)
        with torch.no_grad():
            emb = model.get_image_features(**inputs)
        return emb.cpu().numpy().astype(np.float32)

    # ── add / remove ────────────────────────────────────────────────

    def add(self, experience_id: str, image_paths: list[str]) -> None:
        """Encode images and add them to the FAISS index."""
        import faiss
        self._ensure_index()
        if not image_paths:
            return
        emb = self.embed(image_paths)
        if emb.shape[0] == 0:
            return
        n = emb.shape[0]
        ids = np.arange(self._next_id, self._next_id + n, dtype=np.int64)
        self._index.add_with_ids(emb, ids)
        self._id_to_eid.update({int(faid): experience_id for faid in ids})
        self._eid_to_ids.setdefault(experience_id, []).extend(int(faid) for faid in ids)
        self._next_id += n

    def remove(self, experience_id: str) -> None:
        """Remove all vectors for a given experience_id (idempotent)."""
        import faiss
        faids = self._eid_to_ids.pop(experience_id, [])
        if not faids:
            return
        if hasattr(self._index, "remove_ids"):
            sel = faiss.IDSelectorArray(np.array(faids, dtype=np.int64))
            self._index.remove_ids(sel)
        for faid in faids:
            self._id_to_eid.pop(faid, None)

    # ── search ───────────────────────────────────────────────────────

    def search(
        self,
        query_image_paths: list[str],
        top_k: int = 5,
    ) -> list[tuple[str, float]]:
        """Search for most visually similar experiences.

        Returns [(experience_id, similarity_score)] where similarity_score
        is in [0, 1] (cosine similarity).
        """
        if self._index is None or self._index.ntotal == 0:
            return []
        q_emb = self.embed(query_image_paths)
        if q_emb.shape[0] == 0:
            return []
        import faiss
        # L2 normalize for cosine similarity via inner product
        faiss.normalize_L2(q_emb)
        distances, faiss_ids = self._index.search(q_emb, top_k)
        # Aggregate by experience_id, take max similarity
        score_map: dict[str, float] = {}
        for row_d, row_id in zip(distances, faiss_ids):
            for d, fid in zip(row_d, row_id):
                if fid == -1:
                    continue
                eid = self._id_to_eid.get(int(fid), "")
                if not eid:
                    continue
                # FAISS inner product on normalized vectors ~ cosine in [0, 1]
                score_map[eid] = max(score_map.get(eid, 0.0), float(d))
        # Convert inner product -> cosine (clamped to [0, 1])
        results = sorted(
            [(eid, max(0.0, min(1.0, s))) for eid, s in score_map.items()],
            key=lambda x: -x[1],
        )
        return results[:top_k]

    # ── persistence ──────────────────────────────────────────────────

    def save(self, index_dir: str | Path) -> None:
        import faiss
        d = Path(index_dir)
        d.mkdir(parents=True, exist_ok=True)
        if self._index is not None:
            faiss.write_index(self._index, str(d / "visual_index.faiss"))
        mapping = {
            "model_name": self._model_name,
            "index_key": self._index_key,
            "dim": self._dim,
            "next_id": self._next_id,
            "id_to_eid": {str(k): v for k, v in self._id_to_eid.items()},
            "eid_to_ids": self._eid_to_ids,
        }
        (d / "visual_mapping.json").write_text(json.dumps(mapping, indent=2))

    def load(self, index_dir: str | Path) -> None:
        import faiss
        d = Path(index_dir)
        faiss_path = d / "visual_index.faiss"
        mapping_path = d / "visual_mapping.json"
        if faiss_path.exists():
            self._index = faiss.read_index(str(faiss_path))
        if mapping_path.exists():
            mapping = json.loads(mapping_path.read_text())
            self._model_name = mapping.get("model_name", _CLIP_MODEL_NAME)
            self._index_key = mapping.get("index_key", _FAISS_INDEX_KEY)
            self._dim = int(mapping.get("dim", 512))
            self._next_id = int(mapping.get("next_id", 0))
            self._id_to_eid = {int(k): v for k, v in mapping.get("id_to_eid", {}).items()}
            self._eid_to_ids = mapping.get("eid_to_ids", {})

    @property
    def size(self) -> int:
        return self._index.ntotal if self._index is not None else 0
