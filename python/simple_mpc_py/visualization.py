from __future__ import annotations

import time

import meshcat.geometry as mg
import numpy as np
import pinocchio as pin


class MPCMeshcatVisualizer:
    def __init__(
        self,
        robot,
        model_handler,
        foot_names,
        dt,
        force_scale=0.0025,
        open_viewer=True,
        viewer_name="simple_mpc_py",
    ):
        self.robot = robot
        self.model_handler = model_handler
        self.model = model_handler.getModel()
        self.data = self.model.createData()
        self.foot_names = list(foot_names)
        self.dt = dt
        self.force_scale = force_scale
        self.foot_ids = {foot_name: model_handler.getFootNb(foot_name) for foot_name in self.foot_names}

        self.viz = pin.visualize.MeshcatVisualizer(
            robot.model,
            robot.collision_model,
            robot.visual_model,
        )
        self.viz.initViewer(open=open_viewer)
        self.viz.loadViewerModel(viewer_name)
        self.viewer = self.viz.viewer

        self.reference_color = 0x3A86FF
        self.optimized_color = 0xE63946
        self.force_color = 0xF77F00

    def _update_kinematics(self, x: np.ndarray) -> None:
        x = np.asarray(x, dtype=float)
        q = x[: self.model.nq]
        v = x[self.model.nq :]
        pin.forwardKinematics(self.model, self.data, q, v)
        pin.updateFramePlacements(self.model, self.data)

    def _get_foot_translation(self, foot_name: str) -> np.ndarray:
        foot_frame_id = self.model_handler.getFootFrameId(self.foot_ids[foot_name])
        return np.array(self.data.oMf[foot_frame_id].translation)

    def capture_horizon(self, mpc):
        horizon = len(mpc.us)
        force_size = 3
        optimized = {foot_name: [] for foot_name in self.foot_names}
        reference = {foot_name: [] for foot_name in self.foot_names}
        forces_stage0 = {}
        positions_stage0 = {}

        for t in range(horizon):
            self._update_kinematics(np.array(mpc.xs[t]))
            control_t = np.array(mpc.us[t])
            force_t = control_t[: len(self.foot_names) * force_size].reshape(len(self.foot_names), force_size)
            for foot_index, foot_name in enumerate(self.foot_names):
                optimized_position = self._get_foot_translation(foot_name)
                optimized[foot_name].append(optimized_position)
                reference[foot_name].append(np.array(mpc.getReferencePose(t, foot_name).translation))
                if t == 0:
                    positions_stage0[foot_name] = optimized_position
                    forces_stage0[foot_name] = np.array(force_t[foot_index])

        return {
            "optimized": {name: np.asarray(points) for name, points in optimized.items()},
            "reference": {name: np.asarray(points) for name, points in reference.items()},
            "force_positions": positions_stage0,
            "forces": forces_stage0,
        }

    def display_configuration(self, q):
        self.viz.display(q)

    def display_horizon(self, horizon_data):
        for foot_name in self.foot_names:
            self._set_polyline(
                f"horizon/reference/{foot_name}",
                horizon_data["reference"][foot_name],
                self.reference_color,
            )
            self._set_polyline(
                f"horizon/optimized/{foot_name}",
                horizon_data["optimized"][foot_name],
                self.optimized_color,
            )
            self._set_force_segment(
                f"forces/{foot_name}",
                horizon_data["force_positions"][foot_name],
                horizon_data["forces"][foot_name],
            )

    def play(self, q_traj, horizons, repeat=True):
        while True:
            for q, horizon in zip(q_traj, horizons):
                self.display_configuration(q)
                self.display_horizon(horizon)
                time.sleep(self.dt)
            if not repeat:
                break

    def _set_polyline(self, path, points, color):
        material = mg.LineBasicMaterial(color=color, linewidth=3)
        geometry = mg.PointsGeometry(np.asarray(points).T.astype(np.float32))
        self.viewer[path].set_object(mg.Line(geometry, material))

    def _set_force_segment(self, path, position, force):
        segment = np.vstack((position, position + self.force_scale * force[:3]))
        self._set_polyline(path, segment, self.force_color)
