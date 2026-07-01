from typing import List

import numpy as np
import roboticstoolbox as rtb
import modern_robotics as mr
from spatialmath import SE3

from arm.geometry import Geometry3D
from .robot import Robot


class FR5(Robot):
    """
    FR5 kinematics aligned to manipulator_grasp/assets/fr5/mjcf/fr5.xml.
    """

    def __init__(self) -> None:
        super().__init__()
        self._dof = 6
        self.q0 = [0.0 for _ in range(self._dof)]

        # Keep a valid RTB object for base/tool assignment compatibility.
        links = [rtb.DHLink(d=0.0, alpha=0.0, a=0.0, offset=0.0, mdh=True) for _ in range(self._dof)]
        self.robot = rtb.DHRobot(links)

        self._q_min = np.array([-3.0543, -4.6251, -2.8274, -4.6251, -3.0543, -3.0543], dtype=float)
        self._q_max = np.array([3.0543, 1.4835, 2.8274, 1.4835, 3.0543, 3.0543], dtype=float)

    def _fkine_local(self, q: np.ndarray) -> SE3:
        q = np.asarray(q, dtype=float)
        T = SE3()
        T = T * SE3.Rz(q[0])
        T = T * SE3.Trans(0.0, 0.0, 0.152) * SE3.Rx(np.pi / 2)
        T = T * SE3.Rz(q[1])
        T = T * SE3.Trans(-0.425, 0.0, 0.0)
        T = T * SE3.Rz(q[2])
        T = T * SE3.Trans(-0.39501, 0.0, 0.0)
        T = T * SE3.Rz(q[3])
        T = T * SE3.Trans(0.0, 0.0, 0.1021) * SE3.Rx(np.pi / 2)
        T = T * SE3.Rz(q[4])
        T = T * SE3.Trans(0.0, 0.0, 0.102) * SE3.Rx(-np.pi / 2)
        T = T * SE3.Rz(q[5])
        # fr5.xml flange: pos="0 0.1 0" quat="1 -1 0 0" -> Rx(-pi/2)
        T = T * SE3.Trans(0.0, 0.1, 0.0) * SE3.Rx(-np.pi / 2)
        return T

    def fkine(self, q) -> SE3:
        return self._base * self._fkine_local(np.asarray(q, dtype=float)) * self._tool

    @staticmethod
    def _pose_error(T_cur: SE3, T_des: SE3) -> np.ndarray:
        dp = T_des.t - T_cur.t
        R_err = T_des.R @ T_cur.R.T
        w = mr.so3ToVec(mr.MatrixLog3(R_err))
        return np.hstack((dp, w))

    def _numerical_jacobian(self, q: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        T0 = self._fkine_local(q)
        J = np.zeros((6, self._dof), dtype=float)
        for i in range(self._dof):
            qd = q.copy()
            qd[i] += eps
            Td = self._fkine_local(qd)
            dp = (Td.t - T0.t) / eps
            R_delta = Td.R @ T0.R.T
            dw = mr.so3ToVec(mr.MatrixLog3(R_delta)) / eps
            J[:, i] = np.hstack((dp, dw))
        return J

    def _ikine_dls(self, T_goal: SE3, q_init: np.ndarray) -> np.ndarray:
        q = np.asarray(q_init, dtype=float).copy()
        best_q = q.copy()
        best_cost = np.inf

        for _ in range(120):
            T_cur = self._fkine_local(q)
            e = self._pose_error(T_cur, T_goal)
            # Prioritize position to improve grasp approach stability.
            e[3:] *= 0.35
            cost = np.linalg.norm(e)
            if cost < best_cost:
                best_cost = cost
                best_q = q.copy()
            if np.linalg.norm(e[:3]) < 1e-4 and np.linalg.norm(e[3:]) < 1e-3:
                return q

            J = self._numerical_jacobian(q)
            J[3:, :] *= 0.35
            lam = 1e-3
            H = J @ J.T + lam * np.eye(6)
            dq = J.T @ np.linalg.solve(H, e)

            # Limit one-step jump to avoid diverging near singularities.
            dq_norm = np.linalg.norm(dq)
            if dq_norm > 0.25:
                dq *= 0.25 / dq_norm

            # Backtracking line-search for more stable convergence.
            accepted = False
            alpha = 1.0
            for _ in range(6):
                q_try = np.clip(q + alpha * dq, self._q_min, self._q_max)
                e_try = self._pose_error(self._fkine_local(q_try), T_goal)
                e_try[3:] *= 0.35
                if np.linalg.norm(e_try) < cost:
                    q = q_try
                    accepted = True
                    break
                alpha *= 0.5
            if not accepted:
                q = np.clip(q + 0.1 * dq, self._q_min, self._q_max)

            if np.linalg.norm(dq) < 1e-6:
                break

        T_cur = self._fkine_local(best_q)
        e = self._pose_error(T_cur, T_goal)
        if np.linalg.norm(e[:3]) < 8e-3 and np.linalg.norm(e[3:]) < 1.2e-1:
            return best_q
        return np.array([])

    def ikine(self, Tep: SE3) -> np.ndarray:
        T_goal = self._base.inv() * Tep * self._tool.inv()
        q_now = np.clip(np.asarray(self.q0, dtype=float), self._q_min, self._q_max)
        seeds = [
            q_now,
            np.zeros(self._dof, dtype=float),
            np.array([-1.2, -1.2, 1.2, -1.2, -1.2, 0.0], dtype=float),
            np.array([1.2, -1.0, 1.0, -1.0, 1.0, 0.0], dtype=float),
            np.clip(q_now + np.array([0.25, -0.2, 0.2, -0.2, 0.2, 0.0]), self._q_min, self._q_max),
            np.clip(q_now + np.array([-0.25, 0.2, -0.2, 0.2, -0.2, 0.0]), self._q_min, self._q_max),
        ]

        best_q = np.array([])
        best_score = np.inf
        for q_init in seeds:
            q_sol = self._ikine_dls(T_goal, q_init)
            if len(q_sol):
                err = self._pose_error(self._fkine_local(q_sol), T_goal)
                score = np.linalg.norm(err[:3]) + 0.35 * np.linalg.norm(err[3:])
                if score < 8e-3:
                    return q_sol
                if score < best_score:
                    best_score = score
                    best_q = q_sol

        if len(best_q):
            return best_q
        return np.array([])

    def set_robot_config(self, q):
        # Numerical IK does not need branch config selection.
        return

    def get_geometries(self) -> List[Geometry3D]:
        return []
