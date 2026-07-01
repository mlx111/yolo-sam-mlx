from __future__ import annotations

from pathlib import Path
from typing import Any

import mujoco
from PIL import Image


class KeyframeRecorder:
    def __init__(self, output_dir: Path | None, *, width: int = 640, height: int = 480, camera: str = "") -> None:
        self.output_dir = output_dir
        self.width = int(width)
        self.height = int(height)
        self.camera = str(camera or "")
        self.keyframes: list[dict[str, Any]] = []
        self.errors: list[dict[str, str]] = []
        self._renderer: mujoco.Renderer | None = None

    def capture(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        stage: str,
        *,
        description: str = "",
        action: str = "",
        index: int | None = None,
    ) -> None:
        if self.output_dir is None:
            return
        self.output_dir.mkdir(parents=True, exist_ok=True)
        safe_stage = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in str(stage))
        path = self.output_dir / f"{safe_stage}.png"
        try:
            if self._renderer is None:
                self._renderer = mujoco.Renderer(model, height=self.height, width=self.width)
            mujoco.mj_forward(model, data)
            if self.camera:
                self._renderer.update_scene(data, camera=self.camera)
            else:
                self._renderer.update_scene(data)
            image = self._renderer.render()
            Image.fromarray(image).save(path)
        except Exception as exc:
            self.errors.append({
                "stage": str(stage),
                "error": str(exc),
                "hint": "MuJoCo rendering needs a working GL backend, for example MUJOCO_GL=egl or a display server.",
            })
            return
        frame = {
            "stage": str(stage),
            "image_path": str(path.resolve()),
            "description": description or str(stage),
            "used_for_retrieval": True,
        }
        if action:
            frame["action"] = action
        if index is not None:
            frame["index"] = int(index)
        self.keyframes.append(frame)

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
