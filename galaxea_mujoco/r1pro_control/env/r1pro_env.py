from __future__ import annotations

import time
from pathlib import Path
from typing import Callable

import mujoco
import numpy as np


class R1ProEnv:
    """Small MuJoCo environment wrapper matching the UR5e env control style."""

    def __init__(self, model_path: str | Path = "r1pro_grasp_scene.xml"):
        self.model_path = Path(model_path)
        self.model: mujoco.MjModel | None = None
        self.data: mujoco.MjData | None = None
        self.viewer = None

    def reset(self) -> tuple[mujoco.MjModel, mujoco.MjData]:
        self.model = mujoco.MjModel.from_xml_path(str(self.model_path))
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)
        return self.model, self.data

    def launch_viewer(self):
        if self.model is None or self.data is None:
            self.reset()
        from mujoco import viewer as mj_viewer

        self.viewer = mj_viewer.launch_passive(self.model, self.data)
        return self.viewer

    def step(self, action: np.ndarray | None = None) -> None:
        if self.model is None or self.data is None:
            raise RuntimeError("Call reset() before step().")
        if action is not None:
            self.data.ctrl[:] = action
        mujoco.mj_step(self.model, self.data)
        if self.viewer is not None and self.viewer.is_running():
            self.viewer.sync()

    def spin(self, callback: Callable[[], None] | None = None) -> None:
        if self.model is None or self.data is None:
            raise RuntimeError("Call reset() before spin().")
        while self.viewer is not None and self.viewer.is_running():
            if callback is not None:
                callback()
            else:
                mujoco.mj_forward(self.model, self.data)
            self.viewer.sync()
            time.sleep(self.model.opt.timestep)

    def close(self) -> None:
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

