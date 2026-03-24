from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class RecordConfig:
    robot_name: str = "go2"
    reference_configuration_name: str = "standing"
    base_joint_name: str = "root_joint"
    foot_names: list[str] = field(default_factory=lambda: ["FL_foot", "FR_foot", "RL_foot", "RR_foot"])
    viewer_name: str = "simple_mpc_record"

    dt_mpc: float = 0.01
    dt_sim: float = 0.005
    horizon: int = 50
    gait_command: str = "trot"
    gait_period: float = 0.6

    target_duration: float = 4.0
    recording_dir: str = "results"
    target_weight: float = 1.0
    batch_scan_height: float | None = None
    acceleration_magnitude: float = 1.0

    batch_scan_x_values: list[float] = field(
        default_factory=lambda: [0.0, 0.3, -0.3, 0.6, -0.6, 0.9, -0.9]
    )
    batch_scan_y_values: list[float] = field(
        default_factory=lambda: [0.0, 0.2, -0.2, 0.4, -0.4, 0.6, -0.6]
    )
    batch_scan_yaw_values: list[float] = field(
        default_factory=lambda: [0.0, 0.5, -0.5, 1.0, -1.0, 1.5, -1.5]
    )

    enable_visualization: bool = True
    visualization_sleep: bool = False
    reset_each_round: bool = False
    mpc_print_every: int = 25

    mpc_tol: float = 1e-4
    mpc_mu_init: float = 1e-8
    mpc_max_iters: int = 2
    mpc_num_threads: int = 8
    mpc_swing_apex: float = 0.30

    def apply_reference_height(self, reference_height: float) -> None:
        if self.batch_scan_height is None:
            self.batch_scan_height = float(reference_height)

    @property
    def mpc_loops(self) -> int:
        return int(np.ceil(self.target_duration / self.dt_mpc))
