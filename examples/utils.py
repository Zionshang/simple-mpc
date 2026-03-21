import os
import time

import example_robot_data
import meshcat.geometry as mg
import numpy as np
import pinocchio as pin

from simple_mpc import RobotDataHandler

CURRENT_DIRECTORY = os.getcwd()
DEFAULT_SAVE_DIR = CURRENT_DIRECTORY + "/tmp"


class MPCMeshcatVisualizer:
    def __init__(
        self,
        robot,
        model_handler,
        foot_names,
        dt,
        force_scale=0.0025,
        open_viewer=True,
        viewer_name="simple_mpc",
    ):
        self.robot = robot
        self.model_handler = model_handler
        self.foot_names = list(foot_names)
        self.dt = dt
        self.force_scale = force_scale
        self.data_handler = RobotDataHandler(model_handler)
        self.foot_ids = {
            foot_name: model_handler.getFootNb(foot_name) for foot_name in self.foot_names
        }

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

    def capture_horizon(self, mpc):
        horizon = len(mpc.us)
        force_size = 3
        optimized = {foot_name: [] for foot_name in self.foot_names}
        reference = {foot_name: [] for foot_name in self.foot_names}
        forces_stage0 = {}
        positions_stage0 = {}

        for t in range(horizon):
            self.data_handler.updateInternalData(np.array(mpc.xs[t]), True)
            control_t = np.array(mpc.us[t])
            force_t = control_t[: len(self.foot_names) * force_size].reshape(len(self.foot_names), force_size)
            for foot_index, foot_name in enumerate(self.foot_names):
                optimized_position = np.array(
                    self.data_handler.getFootPose(self.foot_ids[foot_name]).translation
                ).copy()
                optimized[foot_name].append(optimized_position)
                reference[foot_name].append(
                    np.array(mpc.getReferencePose(t, foot_name).translation).copy()
                )
                if t == 0:
                    positions_stage0[foot_name] = optimized_position
                    forces_stage0[foot_name] = force_t[foot_index].copy()

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
        print("Meshcat ready. Press Ctrl+C to stop playback.")
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
        segment = np.vstack((position, position + self.force_scale * force))
        self._set_polyline(path, segment, self.force_color)


def loadTalos():
    robotComplete = example_robot_data.load("talos")
    qComplete = robotComplete.model.referenceConfigurations["half_sitting"]

    locked_joints_names = [
        "arm_left_5_joint",
        "arm_left_6_joint",
        "arm_left_7_joint",
        "gripper_left_joint",
        "arm_right_5_joint",
        "arm_right_6_joint",
        "arm_right_7_joint",
        "gripper_right_joint",
        "head_1_joint",
        "head_2_joint",
    ]
    locked_joints = [robotComplete.model.getJointId(el) for el in locked_joints_names]
    robot = robotComplete.buildReducedRobot(locked_joints, qComplete)
    rmodel: pin.Model = robot.model
    q0 = rmodel.referenceConfigurations["half_sitting"]

    return robotComplete.model, rmodel, qComplete, q0


def save_trajectory(
    xs,
    us,
    com,
    FL_force,
    FR_force,
    RL_force,
    RR_force,
    solve_time,
    FL_trans,
    FR_trans,
    RL_trans,
    RR_trans,
    FL_trans_ref,
    FR_trans_ref,
    RL_trans_ref,
    RR_trans_ref,
    L_measured,
    save_name=None,
    save_dir=DEFAULT_SAVE_DIR,
):
    simu_data = {}
    simu_data["xs"] = xs
    simu_data["us"] = us
    simu_data["com"] = com
    simu_data["FL_force"] = FL_force
    simu_data["FR_force"] = FR_force
    simu_data["RL_force"] = RL_force
    simu_data["RR_force"] = RR_force
    simu_data["FL_trans"] = FL_trans
    simu_data["FR_trans"] = FR_trans
    simu_data["RL_trans"] = RL_trans
    simu_data["RR_trans"] = RR_trans
    simu_data["FL_trans_ref"] = FL_trans_ref
    simu_data["FR_trans_ref"] = FR_trans_ref
    simu_data["RL_trans_ref"] = RL_trans_ref
    simu_data["RR_trans_ref"] = RR_trans_ref
    simu_data["L_measured"] = L_measured
    simu_data["solve_time"] = solve_time
    print("Compressing & saving data...")
    if save_name is None:
        save_name = "sim_data_NO_NAME"
    if save_dir is None:
        save_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data"))
    save_path = save_dir + "/" + save_name + ".npz"
    np.savez_compressed(save_path, data=simu_data)
    print("Saved data to " + str(save_path) + " !")


def load_data(npz_file):
    d = np.load(npz_file, allow_pickle=True, encoding="latin1")
    return d["data"][()]


def extract_forces(problem, workspace, id):
    force_FL = np.zeros(3)
    force_FR = np.zeros(3)
    force_RL = np.zeros(3)
    force_RR = np.zeros(3)
    in_contact = problem.stages[
        id
    ].dynamics.differential_dynamics.constraint_models.__len__()
    for i in range(in_contact):
        if (
            problem.stages[id].dynamics.differential_dynamics.constraint_models[i].name
            == "FL_foot"
        ):
            force_FL = (
                workspace.problem_data.stage_data[id]
                .dynamics_data.continuous_data.constraint_datas[i]
                .contact_force.linear
            )
        elif (
            problem.stages[id].dynamics.differential_dynamics.constraint_models[i].name
            == "FR_foot"
        ):
            force_FR = (
                workspace.problem_data.stage_data[id]
                .dynamics_data.continuous_data.constraint_datas[i]
                .contact_force.linear
            )
        elif (
            problem.stages[id].dynamics.differential_dynamics.constraint_models[i].name
            == "RL_foot"
        ):
            force_RL = (
                workspace.problem_data.stage_data[id]
                .dynamics_data.continuous_data.constraint_datas[i]
                .contact_force.linear
            )
        elif (
            problem.stages[id].dynamics.differential_dynamics.constraint_models[i].name
            == "RR_foot"
        ):
            force_RR = (
                workspace.problem_data.stage_data[id]
                .dynamics_data.continuous_data.constraint_datas[i]
                .contact_force.linear
            )

    return force_FL, force_FR, force_RL, force_RR
