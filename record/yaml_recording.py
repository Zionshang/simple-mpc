from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pinocchio as pin


class YamlRecorder:
    TARGET_DT = 0.005
    TARGET_WEIGHT = 1.0
    FOOT_PREFIXES = ["FL", "FR", "RL", "RR"]
    FOOT_FRAME_NAMES = {
        "FL": "FL_foot",
        "FR": "FR_foot",
        "RL": "RL_foot",
        "RR": "RR_foot",
    }
    JOINT_FIELD_NAMES = [
        "FL_hip_joint_q",
        "FL_thigh_joint_q",
        "FL_calf_joint_q",
        "FR_hip_joint_q",
        "FR_thigh_joint_q",
        "FR_calf_joint_q",
        "RL_hip_joint_q",
        "RL_thigh_joint_q",
        "RL_calf_joint_q",
        "RR_hip_joint_q",
        "RR_thigh_joint_q",
        "RR_calf_joint_q",
    ]
    JOINT_VEL_FIELD_NAMES = [name.replace("_q", "_dq") for name in JOINT_FIELD_NAMES]

    def __init__(self, target_dt: float | None = None, weight: float | None = None):
        self.target_dt = float(self.TARGET_DT if target_dt is None else target_dt)
        self.weight = float(self.TARGET_WEIGHT if weight is None else weight)

    @staticmethod
    def _normalize_quaternion_xyzw(quaternion_xyzw: np.ndarray) -> np.ndarray:
        quaternion_xyzw = np.asarray(quaternion_xyzw, dtype=np.float64)
        return quaternion_xyzw / np.linalg.norm(quaternion_xyzw)

    @classmethod
    def _nlerp_quaternion_xyzw(cls, q0: np.ndarray, q1: np.ndarray, alpha: float) -> np.ndarray:
        q0 = cls._normalize_quaternion_xyzw(q0)
        q1 = cls._normalize_quaternion_xyzw(q1)
        if np.dot(q0, q1) < 0.0:
            q1 = -q1
        q = (1.0 - alpha) * q0 + alpha * q1
        return cls._normalize_quaternion_xyzw(q)

    @classmethod
    def _interpolate_state(cls, q0: np.ndarray, q1: np.ndarray, v0: np.ndarray, v1: np.ndarray, alpha: float):
        q = np.empty_like(q0, dtype=np.float64)
        q[:3] = (1.0 - alpha) * q0[:3] + alpha * q1[:3]
        q[3:7] = cls._nlerp_quaternion_xyzw(q0[3:7], q1[3:7], alpha)
        q[7:] = (1.0 - alpha) * q0[7:] + alpha * q1[7:]
        v = (1.0 - alpha) * v0 + alpha * v1
        return q, v

    def resample_trajectory(self, q_traj, v_traj, source_dt: float):
        q_traj = [np.asarray(q, dtype=np.float64).reshape(-1) for q in q_traj]
        v_traj = [np.asarray(v, dtype=np.float64).reshape(-1) for v in v_traj]

        if len(q_traj) == 0:
            return [], []
        if len(q_traj) == 1:
            return q_traj, v_traj

        source_times = np.arange(len(q_traj), dtype=np.float64) * float(source_dt)
        target_times = np.arange(0.0, source_times[-1] + 1e-12, self.target_dt)

        q_resampled = []
        v_resampled = []
        for t in target_times:
            idx = np.searchsorted(source_times, t, side="right") - 1
            idx = int(np.clip(idx, 0, len(source_times) - 2))
            t0 = source_times[idx]
            t1 = source_times[idx + 1]
            alpha = 0.0 if t1 <= t0 else float((t - t0) / (t1 - t0))
            q, v = self._interpolate_state(q_traj[idx], q_traj[idx + 1], v_traj[idx], v_traj[idx + 1], alpha)
            q_resampled.append(q)
            v_resampled.append(v)

        if not np.allclose(q_resampled[-1], q_traj[-1]) or not np.allclose(v_resampled[-1], v_traj[-1]):
            q_resampled.append(q_traj[-1])
            v_resampled.append(v_traj[-1])

        return q_resampled, v_resampled

    @staticmethod
    def _xyzw_to_wxyz(quaternion_xyzw: np.ndarray) -> list[float]:
        qx, qy, qz, qw = np.asarray(quaternion_xyzw, dtype=np.float64)
        return [float(qw), float(qx), float(qy), float(qz)]

    @classmethod
    def _quaternion_xyzw_to_rotation_matrix(cls, quaternion_xyzw: np.ndarray) -> np.ndarray:
        qx, qy, qz, qw = cls._normalize_quaternion_xyzw(quaternion_xyzw)
        return np.array(
            [
                [1.0 - 2.0 * (qy * qy + qz * qz), 2.0 * (qx * qy - qz * qw), 2.0 * (qx * qz + qy * qw)],
                [2.0 * (qx * qy + qz * qw), 1.0 - 2.0 * (qx * qx + qz * qz), 2.0 * (qy * qz - qx * qw)],
                [2.0 * (qx * qz - qy * qw), 2.0 * (qy * qz + qx * qw), 1.0 - 2.0 * (qx * qx + qy * qy)],
            ],
            dtype=np.float64,
        )

    @staticmethod
    def _get_robot_model(robot):
        if hasattr(robot, "model"):
            return robot.model
        if hasattr(robot, "getModel"):
            return robot.getModel()
        raise AttributeError("robot must expose either a model attribute or getModel()")

    def build_dataset(self, robot, q_traj, v_traj, source_dt: float) -> dict:
        q_resampled, v_resampled = self.resample_trajectory(q_traj, v_traj, source_dt=source_dt)

        model = self._get_robot_model(robot)
        data = model.createData()
        dataset = {
            "dt": float(self.target_dt),
            "weight": float(self.weight),
            "length": len(q_resampled),
            "root_position_world": [],
            "root_quaternion_wxyz": [],
            "root_linear_velocity_base": [],
            "root_angular_velocity_base": [],
            "FL_foot_position_base": [],
            "FR_foot_position_base": [],
            "RL_foot_position_base": [],
            "RR_foot_position_base": [],
            "FL_foot_velocity_base": [],
            "FR_foot_velocity_base": [],
            "RL_foot_velocity_base": [],
            "RR_foot_velocity_base": [],
        }
        for field_name in self.JOINT_FIELD_NAMES + self.JOINT_VEL_FIELD_NAMES:
            dataset[field_name] = []

        frame_ids = {
            prefix: model.getFrameId(frame_name) for prefix, frame_name in self.FOOT_FRAME_NAMES.items()
        }

        for q, v in zip(q_resampled, v_resampled):
            q = np.asarray(q, dtype=np.float64)
            v = np.asarray(v, dtype=np.float64)
            base_position_world = q[:3]
            base_quaternion_xyzw = self._normalize_quaternion_xyzw(q[3:7])
            base_rotation = self._quaternion_xyzw_to_rotation_matrix(base_quaternion_xyzw)
            base_linear_velocity_base = v[:3]
            base_angular_velocity_base = v[3:6]
            base_linear_velocity_world = base_rotation @ base_linear_velocity_base
            base_angular_velocity_world = base_rotation @ base_angular_velocity_base

            pin.forwardKinematics(model, data, q, v)
            pin.updateFramePlacements(model, data)

            dataset["root_position_world"].append(base_position_world.tolist())
            dataset["root_quaternion_wxyz"].append(self._xyzw_to_wxyz(base_quaternion_xyzw))
            dataset["root_linear_velocity_base"].append(base_linear_velocity_base.tolist())
            dataset["root_angular_velocity_base"].append(base_angular_velocity_base.tolist())

            for prefix in self.FOOT_PREFIXES:
                frame_id = frame_ids[prefix]
                foot_position_world = data.oMf[frame_id].translation.copy()
                relative_position_world = foot_position_world - base_position_world
                foot_position_base = base_rotation.T @ relative_position_world

                foot_velocity_world = np.asarray(
                    pin.getFrameVelocity(model, data, frame_id, pin.LOCAL_WORLD_ALIGNED).linear,
                    dtype=np.float64,
                )
                base_point_velocity_world = base_linear_velocity_world + np.cross(
                    base_angular_velocity_world, relative_position_world
                )
                foot_velocity_base = base_rotation.T @ (foot_velocity_world - base_point_velocity_world)

                dataset[f"{prefix}_foot_position_base"].append(foot_position_base.tolist())
                dataset[f"{prefix}_foot_velocity_base"].append(foot_velocity_base.tolist())

            joint_positions = q[7:19]
            joint_velocities = v[6:18]
            for idx, field_name in enumerate(self.JOINT_FIELD_NAMES):
                dataset[field_name].append(float(joint_positions[idx]))
            for idx, field_name in enumerate(self.JOINT_VEL_FIELD_NAMES):
                dataset[field_name].append(float(joint_velocities[idx]))

        dataset["length"] = len(dataset["root_position_world"])
        return dataset

    @staticmethod
    def _yaml_scalar(value) -> str:
        if isinstance(value, float):
            return repr(float(value))
        if isinstance(value, (int, np.integer)):
            return str(int(value))
        return str(value)

    @classmethod
    def _write_yaml_block(cls, lines: list[str], key: str, value, indent: int = 0) -> None:
        prefix = "  " * indent
        if isinstance(value, dict):
            lines.append(f"{prefix}{key}:")
            for child_key, child_value in value.items():
                cls._write_yaml_block(lines, child_key, child_value, indent + 1)
            return

        if isinstance(value, list):
            lines.append(f"{prefix}{key}:")
            if len(value) == 0:
                lines[-1] += " []"
                return
            for item in value:
                if isinstance(item, list):
                    scalar_values = ", ".join(cls._yaml_scalar(x) for x in item)
                    lines.append(f"{prefix}  - [{scalar_values}]")
                else:
                    lines.append(f"{prefix}  - {cls._yaml_scalar(item)}")
            return

        lines.append(f"{prefix}{key}: {cls._yaml_scalar(value)}")

    def save_dataset(self, dataset: dict, output_dir: str, run_name: str | None = None) -> str:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        if run_name is None:
            run_name = "b2_trot_" + datetime.now().strftime("%Y%m%d_%H%M%S")

        file_path = Path(output_dir) / f"{run_name}.yaml"
        lines = []
        for key, value in dataset.items():
            self._write_yaml_block(lines, key, value)
        file_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return str(file_path)
