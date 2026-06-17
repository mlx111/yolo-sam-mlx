from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import mujoco
import numpy as np


ArmSide = Literal["left", "right"]


@dataclass(frozen=True)
class WristContact:
    geom1: str
    geom2: str
    force_world: np.ndarray
    torque_world: np.ndarray
    distance: float


@dataclass(frozen=True)
class WristForceReading:
    side: ArmSide
    force_world: np.ndarray
    torque_world: np.ndarray
    force_norm: float
    torque_norm: float
    contact_count: int
    contacts: tuple[WristContact, ...] = field(default_factory=tuple)


class R1ProWristForceSkill:
    """Estimate wrist external wrench by summing MuJoCo contacts on the terminal hand subtree."""

    terminal_roots: dict[ArmSide, tuple[str, ...]] = {
        "left": ("left_arm_link7", "left_gripper_link"),
        "right": ("right_arm_link7", "right_gripper_link"),
    }

    def _body_id(self, model: mujoco.MjModel, body_name: str) -> int:
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if body_id < 0:
            raise ValueError(f"MuJoCo body not found: {body_name}")
        return int(body_id)

    def _site_pos(self, model: mujoco.MjModel, data: mujoco.MjData, site_name: str) -> np.ndarray:
        site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        if site_id < 0:
            raise ValueError(f"MuJoCo site not found: {site_name}")
        return data.site_xpos[site_id].copy()

    def _geom_name(self, model: mujoco.MjModel, geom_id: int) -> str:
        return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or f"geom:{geom_id}"

    def _descendant_bodies(self, model: mujoco.MjModel, roots: tuple[str, ...]) -> set[int]:
        root_ids = {self._body_id(model, name) for name in roots}
        body_ids: set[int] = set()
        for body_id in range(model.nbody):
            current = body_id
            while current >= 0:
                if current in root_ids:
                    body_ids.add(body_id)
                    break
                parent = int(model.body_parentid[current])
                if parent == current:
                    break
                current = parent
        return body_ids

    def _terminal_geom_ids(self, model: mujoco.MjModel, side: ArmSide) -> set[int]:
        body_ids = self._descendant_bodies(model, self.terminal_roots[side])
        return {geom_id for geom_id in range(model.ngeom) if int(model.geom_bodyid[geom_id]) in body_ids}

    def read(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        side: ArmSide,
        *,
        include_wrist_link: bool = True,
    ) -> WristForceReading:
        if side not in self.terminal_roots:
            raise ValueError(f"Unsupported side: {side!r}")

        mujoco.mj_forward(model, data)
        terminal_geoms = self._terminal_geom_ids(model, side)
        if not include_wrist_link:
            wrist_body = self._body_id(model, f"{side}_arm_link7")
            terminal_geoms = {
                geom_id
                for geom_id in terminal_geoms
                if int(model.geom_bodyid[geom_id]) != wrist_body
            }

        tcp_pos = self._site_pos(model, data, f"{side}_hand_tcp")
        total_force = np.zeros(3, dtype=np.float64)
        total_torque = np.zeros(3, dtype=np.float64)
        contacts: list[WristContact] = []

        for contact_index in range(data.ncon):
            contact = data.contact[contact_index]
            geom1 = int(contact.geom1)
            geom2 = int(contact.geom2)
            geom1_terminal = geom1 in terminal_geoms
            geom2_terminal = geom2 in terminal_geoms
            if geom1_terminal == geom2_terminal:
                continue

            local_wrench = np.zeros(6, dtype=np.float64)
            mujoco.mj_contactForce(model, data, contact_index, local_wrench)
            frame = contact.frame.reshape(3, 3)
            force_world = frame.T @ local_wrench[:3]
            torque_world = frame.T @ local_wrench[3:]
            if geom2_terminal:
                force_world = -force_world
                torque_world = -torque_world

            contact_pos = np.asarray(contact.pos, dtype=np.float64)
            torque_at_tcp = torque_world + np.cross(contact_pos - tcp_pos, force_world)
            total_force += force_world
            total_torque += torque_at_tcp
            contacts.append(
                WristContact(
                    geom1=self._geom_name(model, geom1),
                    geom2=self._geom_name(model, geom2),
                    force_world=force_world.copy(),
                    torque_world=torque_at_tcp.copy(),
                    distance=float(contact.dist),
                )
            )

        return WristForceReading(
            side=side,
            force_world=total_force,
            torque_world=total_torque,
            force_norm=float(np.linalg.norm(total_force)),
            torque_norm=float(np.linalg.norm(total_torque)),
            contact_count=len(contacts),
            contacts=tuple(contacts),
        )

    def execute_recovery_action(self, model: mujoco.MjModel, data: mujoco.MjData, params: dict) -> WristForceReading:
        side = str(params.get("side", "left"))
        if side not in self.terminal_roots:
            raise ValueError(f"Unsupported side: {side!r}")
        return self.read(
            model,
            data,
            side,  # type: ignore[arg-type]
            include_wrist_link=bool(params.get("include_wrist_link", True)),
        )


def load_skill() -> R1ProWristForceSkill:
    return R1ProWristForceSkill()
