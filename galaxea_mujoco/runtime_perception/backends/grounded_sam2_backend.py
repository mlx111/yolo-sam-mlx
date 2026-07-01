from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
import warnings
from pathlib import Path
import re
import sys
import tempfile
from typing import Any

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from runtime_perception.warnings_filter import suppress_grounded_sam2_warnings

suppress_grounded_sam2_warnings()


@dataclass
class GroundedSAM2Result:
    mask: Any
    mask_path: str | None
    annotated_path: str | None
    candidate: dict | None
    candidates: list[dict]
    image_shape: list[int]


class GroundedSAM2Segmenter:
    def __init__(
        self,
        grounded_sam2_root: str | None = None,
        sam2_checkpoint: str | None = None,
        sam2_model_config: str = "configs/sam2.1/sam2.1_hiera_l.yaml",
        grounding_dino_config: str | None = None,
        grounding_dino_checkpoint: str | None = None,
        bert_path: str | None = None,
        box_threshold: float = 0.2,
        text_threshold: float = 0.2,
        device: str | None = None,
        multimask_output: bool = True,
    ):
        self.root = self._resolve_root(grounded_sam2_root)
        self.sam2_checkpoint = str(self.root / "checkpoints" / "sam2.1_hiera_large.pt") if sam2_checkpoint is None else sam2_checkpoint
        self.sam2_model_config = sam2_model_config
        self.grounding_dino_config = (
            str(self.root / "grounding_dino" / "groundingdino" / "config" / "GroundingDINO_SwinT_OGC.py")
            if grounding_dino_config is None
            else grounding_dino_config
        )
        self.grounding_dino_checkpoint = (
            str(self.root / "gdino_checkpoints" / "groundingdino_swint_ogc.pth")
            if grounding_dino_checkpoint is None
            else grounding_dino_checkpoint
        )
        self.bert_path = str(self.root / "bert-base-uncased") if bert_path is None else bert_path
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.device = device
        self.multimask_output = multimask_output
        self._loaded = False

    @staticmethod
    def _resolve_root(root: str | None) -> Path:
        if root:
            path = Path(root)
            return path if path.is_absolute() else (Path.cwd() / path).resolve()
        package_root = Path(__file__).resolve().parents[2]
        internal = package_root / "models" / "Grounded-SAM-2"
        if internal.exists():
            return internal
        return (package_root.parent / "Grounded-SAM-2").resolve()

    def _validate_resources(self) -> None:
        required_paths = {
            "Grounded-SAM2 root": self.root,
            "SAM2 package": self.root / "sam2",
            "GroundingDINO package": self.root / "grounding_dino",
            "SAM2 checkpoint": self.sam2_checkpoint,
            "GroundingDINO config": self.grounding_dino_config,
            "GroundingDINO checkpoint": self.grounding_dino_checkpoint,
            "BERT local model": self.bert_path,
        }
        missing = [f"{name}: {path}" for name, path in required_paths.items() if not Path(path).exists()]
        if missing:
            raise FileNotFoundError("Missing Grounded-SAM2 resources:\n" + "\n".join(missing))

    def _prepare_grounding_dino_config(self) -> str:
        source_config = Path(self.grounding_dino_config)
        config_text = source_config.read_text(encoding="utf-8")
        replacement = f'text_encoder_type = r"{Path(self.bert_path)}"'
        if re.search(r"^text_encoder_type\s*=", config_text, flags=re.MULTILINE):
            config_text = re.sub(r"^text_encoder_type\s*=.*$", replacement, config_text, flags=re.MULTILINE)
        else:
            config_text = config_text.rstrip() + "\n" + replacement + "\n"

        digest = hashlib.sha1((str(source_config.resolve()) + str(self.bert_path)).encode("utf-8")).hexdigest()[:12]
        config_dir = Path(tempfile.gettempdir()) / "galaxea_grounded_sam2_configs"
        config_dir.mkdir(parents=True, exist_ok=True)
        local_config = config_dir / f"{source_config.stem}_{digest}.py"
        local_config.write_text(config_text, encoding="utf-8")
        return str(local_config)

    def _load(self):
        if self._loaded:
            return
        self._validate_resources()
        if str(self.root) not in sys.path:
            sys.path.insert(0, str(self.root))
        grounding_dino_root = self.root / "grounding_dino"
        if str(grounding_dino_root) not in sys.path:
            sys.path.insert(0, str(grounding_dino_root))
        os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        os.environ.setdefault("HF_HUB_OFFLINE", "1")

        import torch
        from grounding_dino.groundingdino.util.inference import load_image, load_model, predict
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        from torchvision.ops import box_convert
        from transformers import AutoModel, AutoTokenizer

        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        AutoTokenizer.from_pretrained(self.bert_path, local_files_only=True, trust_remote_code=False)
        AutoModel.from_pretrained(self.bert_path, local_files_only=True, trust_remote_code=False).to(self.device)

        sam2_model = build_sam2(self.sam2_model_config, self.sam2_checkpoint, device=self.device)
        self.sam2_predictor = SAM2ImagePredictor(sam2_model)
        self.grounding_model = load_model(
            model_config_path=self._prepare_grounding_dino_config(),
            model_checkpoint_path=self.grounding_dino_checkpoint,
            device=self.device,
        )
        self.torch = torch
        self.load_image = load_image
        self.predict = predict
        self.box_convert = box_convert
        self._loaded = True

    @staticmethod
    def _normalize_target_class(target_class: str) -> str:
        return str(target_class).strip().lower().rstrip(".")

    @staticmethod
    def _prompt_variants(target_class: str) -> list[str]:
        target = GroundedSAM2Segmenter._normalize_target_class(target_class)
        if not target:
            return []
        aliases = [target]
        aliases.extend({
            "pear": ("pear", "green pear", "yellow pear", "fruit pear"),
            "apple": ("apple", "red apple", "fruit apple"),
        }.get(target, ()))
        templates = ("{target}.", "a {target}.", "one {target}.", "single {target} object.")
        variants = []
        seen = set()
        for alias in aliases:
            for template in templates:
                prompt = template.format(target=alias)
                if prompt not in seen:
                    seen.add(prompt)
                    variants.append(prompt)
        return variants

    @staticmethod
    def _box_iou(box_a, box_b) -> float:
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b
        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)
        inter_area = max(0.0, inter_x2 - inter_x1) * max(0.0, inter_y2 - inter_y1)
        area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        union = area_a + area_b - inter_area
        return 0.0 if union <= 1e-9 else float(inter_area / union)

    def _dedupe_candidates(self, candidates: list[dict], iou_thresh: float = 0.7) -> list[dict]:
        ordered = sorted(candidates, key=lambda item: item["score"], reverse=True)
        kept = []
        for candidate in ordered:
            box = candidate["xyxy"]
            if any(self._box_iou(box, existing["xyxy"]) >= iou_thresh for existing in kept):
                continue
            kept.append(candidate)
        return kept

    def _run_grounding_dino_candidates(self, target_class: str, image_source, image) -> list[dict]:
        h, w, _ = image_source.shape
        candidates = []
        for prompt in self._prompt_variants(target_class):
            boxes, confidences, labels = self.predict(
                model=self.grounding_model,
                image=image,
                caption=prompt,
                box_threshold=self.box_threshold,
                text_threshold=self.text_threshold,
                device=self.device,
            )
            if len(boxes) == 0:
                continue
            boxes = boxes * self.torch.tensor([w, h, w, h], dtype=boxes.dtype, device=boxes.device)
            xyxy = self.box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").cpu().numpy()
            scores = confidences.detach().cpu().numpy().tolist()
            for box, score, label in zip(xyxy, scores, labels):
                candidates.append({
                    "detector": "grounding_dino",
                    "prompt": prompt,
                    "label": str(label),
                    "score": float(score),
                    "xyxy": [float(v) for v in box.tolist()],
                })
        return candidates

    @staticmethod
    def _foreground_prior(image_bgr):
        import cv2
        import numpy as np

        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
        foreground = ((hsv[:, :, 1] > 35) & (hsv[:, :, 2] > 35)).astype(np.uint8) * 255
        kernel = np.ones((5, 5), dtype=np.uint8)
        foreground = cv2.morphologyEx(foreground, cv2.MORPH_OPEN, kernel)
        foreground = cv2.morphologyEx(foreground, cv2.MORPH_CLOSE, kernel)
        return foreground

    @staticmethod
    def _mask_quality(mask, candidate: dict, foreground_mask) -> tuple[float, dict]:
        import cv2
        import numpy as np

        mask_bool = mask.astype(bool)
        area = int(np.count_nonzero(mask_bool))
        if area == 0:
            return -1e9, {"area": 0, "foreground_iou": 0.0, "bbox_fill": 0.0, "component_count": 0}
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        x1, y1, x2, y2 = [int(round(v)) for v in candidate["xyxy"]]
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(mask.shape[1], x2)
        y2 = min(mask.shape[0], y2)
        bbox_area = max(1, (x2 - x1) * (y2 - y1))
        bbox_fill = float(area / bbox_area)
        fg_bool = foreground_mask.astype(bool)
        intersection = int(np.count_nonzero(mask_bool & fg_bool))
        union = int(np.count_nonzero(mask_bool | fg_bool))
        foreground_iou = 0.0 if union == 0 else float(intersection / union)
        score = 1.5 * float(candidate["score"]) + 2.0 * foreground_iou + 0.35 * min(bbox_fill, 1.0) - 0.15 * max(len(contours) - 1, 0)
        return score, {"area": area, "foreground_iou": foreground_iou, "bbox_fill": bbox_fill, "component_count": len(contours)}

    def _predict_masks_for_candidate(self, image_source, candidate: dict):
        import numpy as np

        h, w, _ = image_source.shape
        box = np.asarray(candidate["xyxy"], dtype=np.float32).reshape(1, 4)
        box[:, 0::2] = np.clip(box[:, 0::2], 0, w - 1)
        box[:, 1::2] = np.clip(box[:, 1::2], 0, h - 1)
        masks, _, _ = self.sam2_predictor.predict(point_coords=None, point_labels=None, box=box, multimask_output=self.multimask_output)
        masks = np.asarray(masks)
        if masks.ndim == 4:
            masks = masks[:, 0]
        if masks.ndim == 2:
            masks = masks[None, :, :]
        return (masks > 0).astype(np.uint8)

    @staticmethod
    def _foreground_component_fallback(foreground_mask):
        import cv2
        import numpy as np

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(foreground_mask, connectivity=8)
        if num_labels <= 1:
            return None
        best_idx = None
        best_area = 0
        for idx in range(1, num_labels):
            area = int(stats[idx, cv2.CC_STAT_AREA])
            if area > best_area:
                best_idx = idx
                best_area = area
        if best_idx is None or best_area < 64:
            return None
        return (labels == best_idx).astype(np.uint8) * 255

    def _save_annotated_image(self, image_bgr, mask, candidate: dict | None, output_annotated_path: str) -> str:
        import cv2

        annotated = image_bgr.copy()
        if mask is not None:
            mask_bool = mask.astype(bool)
            overlay = annotated.copy()
            overlay[mask_bool] = (0, 180, 0)
            annotated = cv2.addWeighted(overlay, 0.35, annotated, 0.65, 0)
        if candidate and candidate.get("xyxy"):
            x1, y1, x2, y2 = [int(round(v)) for v in candidate["xyxy"]]
            h, w = annotated.shape[:2]
            x1 = max(0, min(w - 1, x1))
            x2 = max(0, min(w - 1, x2))
            y1 = max(0, min(h - 1, y1))
            y2 = max(0, min(h - 1, y2))
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        Path(output_annotated_path).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(output_annotated_path, annotated)
        return output_annotated_path

    def segment_image(self, image_path: str, target_class: str, output_mask_path: str | None = None, output_annotated_path: str | None = None):
        self._load()
        import cv2
        import numpy as np

        image_bgr = cv2.imread(image_path)
        if image_bgr is None:
            raise FileNotFoundError(f"Failed to read image: {image_path}")
        image_source, image = self.load_image(image_path)
        self.sam2_predictor.set_image(image_source)
        foreground_mask = self._foreground_prior(image_bgr)
        candidates = self._dedupe_candidates(self._run_grounding_dino_candidates(target_class, image_source, image))

        best_mask = None
        best_candidate = None
        best_score = -1e9
        for candidate in candidates[:12]:
            for mask in self._predict_masks_for_candidate(image_source, candidate):
                mask_uint8 = (mask > 0).astype(np.uint8) * 255
                score, stats = self._mask_quality(mask_uint8, candidate, foreground_mask)
                candidate_with_mask = {**candidate, "mask_score": float(score), "mask_stats": stats}
                if score > best_score:
                    best_score = score
                    best_mask = mask_uint8
                    best_candidate = candidate_with_mask

        if best_mask is None:
            best_mask = self._foreground_component_fallback(foreground_mask)
            if best_mask is None:
                return None
            best_candidate = {"detector": "foreground_fallback", "prompt": target_class, "label": target_class, "score": 0.0, "xyxy": None}

        mask_path = None
        if output_mask_path:
            Path(output_mask_path).parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(output_mask_path, best_mask)
            mask_path = output_mask_path
        annotated_path = None
        if output_annotated_path:
            annotated_path = self._save_annotated_image(image_bgr, best_mask, best_candidate, output_annotated_path)
        return GroundedSAM2Result(best_mask, mask_path, annotated_path, best_candidate, candidates, list(image_bgr.shape))
