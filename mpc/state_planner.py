from __future__ import annotations

import numpy as np

from .robot_handler import QuadRobot


class StatePlanner:
    def __init__(self, robot: QuadRobot, horizon: int, timestep: float):
        self.robot_ = robot
        self.nq_ = robot.getModel().nq
        self.nv_ = robot.getModel().nv
        self.horizon_ = int(horizon)
        self.timestep_ = float(timestep)

        self.nominal_state_ = np.array(robot.getReferenceState(), dtype=float)
        self.nominal_joint_positions_ = self.nominal_state_[7 : self.nq_].copy()
        self.base_height_ = float(self.nominal_state_[2])
        print(f"StatePlanner default height: {self.base_height_:.3f} m")
        self.references_ = [self.nominal_state_.copy() for _ in range(self.horizon_)]

    def updateReference(self, x_current: np.ndarray, velocity_base: np.ndarray) -> None:
        x_current = np.asarray(x_current, dtype=float)
        velocity_base = np.asarray(velocity_base, dtype=float)
        if x_current.shape[0] != self.nq_ + self.nv_:
            raise RuntimeError("current state size does not match robot model")
        if velocity_base.shape[0] != 6:
            raise RuntimeError("velocity_base size should be 6")

        base_position = np.array(x_current[:3], dtype=float)
        yaw0 = self._yaw_from_quaternion(x_current[3:7])

        for t in range(self.horizon_):
            dt = t * self.timestep_
            x_ref = self.nominal_state_.copy()
            x_ref[0] = base_position[0] + velocity_base[0] * dt
            x_ref[1] = base_position[1] + velocity_base[1] * dt
            x_ref[2] = self.base_height_
            x_ref[3:7] = self._quaternion_from_yaw(yaw0 + velocity_base[5] * dt)
            x_ref[7: self.nq_] = self.nominal_joint_positions_
            x_ref[self.nq_: self.nq_ + 6] = 0.0
            x_ref[self.nq_] = velocity_base[0]
            x_ref[self.nq_ + 1] = velocity_base[1]
            x_ref[self.nq_ + 5] = velocity_base[5]
            self.references_[t] = x_ref

    def getReference(self, t: int) -> np.ndarray:
        return self.references_[t].copy()

    def _yaw_from_quaternion(self, quat: np.ndarray) -> float:
        qx, qy, qz, qw = np.asarray(quat, dtype=float)
        return float(
            np.arctan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
        )

    def _quaternion_from_yaw(self, yaw: float) -> np.ndarray:
        half_yaw = 0.5 * float(yaw)
        return np.array([0.0, 0.0, np.sin(half_yaw), np.cos(half_yaw)], dtype=float)
