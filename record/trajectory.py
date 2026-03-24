from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import RecordConfig


@dataclass(frozen=True)
class CommandJob:
    command: np.ndarray
    run_name: str


class TrajectoryGenerator:
    def __init__(self, config: RecordConfig):
        self.config = config

    @staticmethod
    def _quaternion_xyzw_to_euler_xyz(quat_xyzw: np.ndarray):
        x, y, z, w = np.asarray(quat_xyzw, dtype=np.float64)

        sinr_cosp = 2.0 * (w * x + y * z)
        cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)

        sinp = 2.0 * (w * y - z * x)
        sinp = np.clip(sinp, -1.0, 1.0)
        pitch = np.arcsin(sinp)

        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)

        return roll, pitch, yaw

    @staticmethod
    def _euler_xyz_to_quaternion_xyzw(roll: float, pitch: float, yaw: float) -> np.ndarray:
        cr = np.cos(roll * 0.5)
        sr = np.sin(roll * 0.5)
        cp = np.cos(pitch * 0.5)
        sp = np.sin(pitch * 0.5)
        cy = np.cos(yaw * 0.5)
        sy = np.sin(yaw * 0.5)

        w = cr * cp * cy + sr * sp * sy
        x = sr * cp * cy - cr * sp * sy
        y = cr * sp * cy + sr * cp * sy
        z = cr * cp * sy - sr * sp * cy
        return np.array([x, y, z, w], dtype=np.float64)

    @staticmethod
    def format_scan_value(value: float) -> str:
        return f"{float(value):.2f}".replace(".", "d")

    def make_default_run_name(self, command: np.ndarray) -> str:
        vx, vy, vyaw, height = np.asarray(command, dtype=np.float64)
        prefix = f"{self.config.gait_command}{self.format_scan_value(self.config.gait_period)}"
        return (
            f"{prefix}"
            f"_x{self.format_scan_value(vx)}"
            f"_y{self.format_scan_value(vy)}"
            f"_yaw{self.format_scan_value(vyaw)}"
            f"_height{self.format_scan_value(height)}"
        )

    def build_commands(self) -> list[CommandJob]:
        commands = []
        prefix = f"{self.config.gait_command}{self.format_scan_value(self.config.gait_period)}"
        for vx in self.config.batch_scan_x_values:
            for vy in self.config.batch_scan_y_values:
                for vyaw in self.config.batch_scan_yaw_values:
                    command = np.array([vx, vy, vyaw, self.config.batch_scan_height], dtype=np.float64)
                    run_name = (
                        f"{prefix}"
                        f"_x{self.format_scan_value(vx)}"
                        f"_y{self.format_scan_value(vy)}"
                        f"_yaw{self.format_scan_value(vyaw)}"
                        f"_height{self.format_scan_value(self.config.batch_scan_height)}"
                    )
                    commands.append(CommandJob(command=command, run_name=run_name))
        return commands

    @staticmethod
    def _trapezoidal_velocity_profile(
        t: float,
        vmax: float,
        acc_time: float,
        total_duration: float,
    ) -> float:
        t = float(t)
        vmax = float(vmax)
        total_duration = float(total_duration)
        if abs(vmax) < 1e-12 or total_duration <= 0.0:
            return 0.0

        acc_time = float(np.clip(acc_time, 0.0, 0.5 * total_duration))
        if acc_time < 1e-12:
            return vmax if 0.0 <= t <= total_duration else 0.0

        cruise_end = total_duration - acc_time
        if t <= 0.0:
            return 0.0
        if t < acc_time:
            return vmax * (t / acc_time)
        if t <= cruise_end:
            return vmax
        if t < total_duration:
            return vmax * ((total_duration - t) / acc_time)
        return 0.0

    def build_target_trajectory(self, command: np.ndarray, q0: np.ndarray) -> dict:
        command = np.asarray(command, dtype=np.float64)
        vx_max = float(command[0])
        vy_max = float(command[1])
        vyaw_max = float(command[2])
        height = float(command[3])

        reference_dt = float(self.config.dt_mpc)
        if reference_dt <= 0.0:
            raise ValueError("reference dt must be positive")

        desired_time = max(float(self.config.target_duration), reference_dt)
        step_count = max(1, int(np.ceil(desired_time / reference_dt)))
        time_trajectory = np.arange(step_count + 1, dtype=np.float64) * reference_dt
        total_duration = float(time_trajectory[-1])

        acceleration_magnitude = max(float(self.config.acceleration_magnitude), 1e-6)
        linear_speed = float(np.hypot(vx_max, vy_max))
        linear_acc_time = linear_speed / acceleration_magnitude if linear_speed > 1e-12 else 0.0
        yaw_acc_time = abs(vyaw_max) / acceleration_magnitude if abs(vyaw_max) > 1e-12 else 0.0
        acc_time = max(linear_acc_time, yaw_acc_time, reference_dt)
        acc_time = min(acc_time, 0.5 * total_duration)

        state_trajectory = []

        q0 = np.asarray(q0, dtype=np.float64).reshape(-1)
        base_position = q0[:3].copy()
        base_position[2] = height
        joint_state = q0[7:19].copy()

        roll, pitch, yaw = self._quaternion_xyzw_to_euler_xyz(q0[3:7])
        initial_configuration = q0.copy()
        initial_configuration[:3] = base_position
        first_state = np.concatenate((initial_configuration, np.zeros(6, dtype=np.float64), np.zeros(12, dtype=np.float64)))
        state_trajectory.append(first_state)

        vx_prev = 0.0
        vy_prev = 0.0
        vyaw_prev = 0.0
        for i in range(1, len(time_trajectory)):
            t = float(time_trajectory[i])
            dt_step = float(time_trajectory[i] - time_trajectory[i - 1])

            vx_next = self._trapezoidal_velocity_profile(t, vx_max, acc_time, total_duration)
            vy_next = self._trapezoidal_velocity_profile(t, vy_max, acc_time, total_duration)
            vyaw_next = self._trapezoidal_velocity_profile(t, vyaw_max, acc_time, total_duration)

            vx_mid = 0.5 * (vx_prev + vx_next)
            vy_mid = 0.5 * (vy_prev + vy_next)
            vyaw_mid = 0.5 * (vyaw_prev + vyaw_next)
            yaw_mid = yaw + 0.5 * vyaw_mid * dt_step

            dx = (vx_mid * np.cos(yaw_mid) - vy_mid * np.sin(yaw_mid)) * dt_step
            dy = (vx_mid * np.sin(yaw_mid) + vy_mid * np.cos(yaw_mid)) * dt_step

            base_position[0] += dx
            base_position[1] += dy
            base_position[2] = height
            yaw += vyaw_mid * dt_step

            quat_xyzw = self._euler_xyz_to_quaternion_xyzw(roll, pitch, yaw)
            base_twist = np.array([vx_next, vy_next, 0.0, 0.0, 0.0, vyaw_next], dtype=np.float64)
            state = np.concatenate((base_position.copy(), quat_xyzw, joint_state, base_twist, np.zeros(12)))
            state_trajectory.append(state)

            vx_prev = vx_next
            vy_prev = vy_next
            vyaw_prev = vyaw_next

        return {
            "command": {
                "vx": vx_max,
                "vy": vy_max,
                "vyaw": vyaw_max,
                "height": height,
            },
            "time_trajectory": time_trajectory,
            "state_trajectory": state_trajectory,
            "dt": reference_dt,
            "duration": total_duration,
            "num_points": len(time_trajectory),
            "acceleration_time": acc_time,
        }

    def build_mpc_reference_window(
        self,
        target_trajectory: dict,
        start_index: int,
        horizon: int,
    ) -> np.ndarray:
        state_trajectory = np.asarray(target_trajectory["state_trajectory"], dtype=np.float64)
        start_index = max(0, int(start_index))
        horizon = int(horizon)
        if horizon <= 0:
            raise ValueError("horizon must be positive")

        if start_index >= len(state_trajectory):
            last_state = state_trajectory[-1]
            return np.repeat(last_state[None, :], horizon, axis=0)

        end_index = start_index + horizon
        window = state_trajectory[start_index:min(end_index, len(state_trajectory))]
        if len(window) < horizon:
            pad = np.repeat(state_trajectory[-1][None, :], horizon - len(window), axis=0)
            window = np.vstack((window, pad))
        return np.asarray(window, dtype=np.float64)
