from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class RecordConfig:
    target_velocity_command: np.ndarray = field(
        default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.55], dtype=np.float64)
    )
    dt: float = 0.015
    target_duration: float = 4.0
    acceleration_magnitude: float = 1.0
    gait_command: str = "trot"
    gait_period: float = 0.8
    batch_scan_mode: bool = True
    batch_scan_height: float = 0.40
    batch_scan_x_values: list[float] = field(default_factory=lambda: [0.0, 0.3, -0.3, 0.6, -0.6, 0.9, -0.9])
    batch_scan_y_values: list[float] = field(default_factory=lambda: [0.0, 0.2, -0.2, 0.4, -0.4, 0.6, -0.6])
    batch_scan_yaw_values: list[float] = field(default_factory=lambda: [0.0, 0.5, -0.5, 1.0, -1.0, 1.5, -1.5])
    recording_dir: str = "results"

    @property
    def gait_type(self) -> str:
        return self.gait_command

    @property
    def mpc_loops(self) -> int:
        return int(np.ceil(self.target_duration / self.dt))
