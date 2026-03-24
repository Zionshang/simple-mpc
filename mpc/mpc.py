from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pinocchio as pin

import aligator

from .foot_planner import FootPlanner


@dataclass
class MPCSettings:
    TOL: float = 1e-4
    mu_init: float = 1e-8
    max_iters: int = 1
    num_threads: int = 1
    swing_apex: float = 0.15
    T_fly: int = 80
    T_contact: int = 20
    timestep: float = 0.01
    rollout_timestep: float = 0.005

    @classmethod
    def from_dict(cls, settings: dict) -> "MPCSettings":
        return cls(**settings)


class MPC:
    WALKING = "walking"
    STANDING = "standing"

    def __init__(self, settings: dict, problem):
        self.settings_ = MPCSettings.from_dict(settings)
        self.ocp_ = problem

        robot = self.ocp_.getRobot()
        self.robot = robot
        self.model_ = robot.getModel()
        self.data_ = self.model_.createData()
        self.support_force_ = -self.robot.getMass() * self.ocp_.settings_["gravity"][2]
        self._update_kinematics(robot.getReferenceState())

        starting_poses = {}
        for foot_nb in range(robot.getFeetNb()):
            name = robot.getFootFrameName(foot_nb)
            foot_frame_id = self.robot.getFootFrameId(foot_nb)
            starting_poses[name] = np.array(self.data_.oMf[foot_frame_id].translation)

        self.foot_planner_ = FootPlanner(
            starting_poses,
            self.settings_.swing_apex,
            self.settings_.T_fly,
            self.settings_.T_contact,
            self.ocp_.getSize(),
            self.settings_.timestep,
        )
        self.x0_ = np.array(robot.getReferenceState(), dtype=float)

        self.solver_ = aligator.SolverProxDDP(
            self.settings_.TOL,
            self.settings_.mu_init,
            100,
            aligator.QUIET,
        )
        self.solver_.rollout_type = aligator.ROLLOUT_LINEAR
        if self.settings_.num_threads > 1:
            self.solver_.linear_solver_choice = aligator.LQ_SOLVER_PARALLEL
            self.solver_.setNumThreads(self.settings_.num_threads)
        else:
            self.solver_.linear_solver_choice = aligator.LQ_SOLVER_SERIAL
        self.solver_.force_initial_condition = True

        self.ee_names_ = robot.getFeetFrameNames()
        force_ref = self.ocp_.getReferenceForce(0, robot.getFootFrameName(0))

        contact_states = {}
        land_constraint = {}
        contact_poses = {}
        force_map = {}
        for name in self.ee_names_:
            foot_nb = robot.getFootNb(name)
            placement = self.data_.oMf[self.robot.getFootFrameId(foot_nb)]
            contact_states[name] = True
            land_constraint[name] = False
            contact_poses[name] = pin.SE3(np.array(placement.rotation), np.array(placement.translation))
            force_map[name] = force_ref

        self.xs_ = []
        self.us_ = []
        self.standing_horizon_ = []
        self.standing_horizon_data_ = []
        self.cycle_horizon_ = []
        self.cycle_horizon_data_ = []
        self.contact_states_: list[dict[str, bool]] = []
        self.foot_takeoff_times_ = {}
        self.foot_land_times_ = {}

        for _ in range(len(self.ocp_.getProblem().stages)):
            self.xs_.append(self.x0_.copy())
            self.us_.append(self.ocp_.getReferenceControl(0))
            stage = self.ocp_.createStage(contact_states, contact_poses, force_map, land_constraint)
            self.standing_horizon_.append(stage)
            self.standing_horizon_data_.append(stage.createData())
        self.xs_.append(self.x0_.copy())

        self.solver_.setup(self.ocp_.getProblem())
        self.solver_.run(self.ocp_.getProblem(), self.xs_, self.us_)

        self.xs_ = [np.array(x) for x in self.solver_.results.xs]
        self.us_ = [np.array(u) for u in self.solver_.results.us]

        self.solver_.max_iters = self.settings_.max_iters

        self.now_ = self.WALKING
        self.rollout_substeps_ = int(round(self.settings_.timestep / self.settings_.rollout_timestep))

    def _update_kinematics(self, x: np.ndarray) -> None:
        x = np.asarray(x, dtype=float)
        q = x[: self.model_.nq]
        v = x[self.model_.nq :]
        pin.forwardKinematics(self.model_, self.data_, q, v)
        pin.updateFramePlacements(self.model_, self.data_)

    @property
    def xs(self):
        return [x.copy() for x in self.xs_]

    @property
    def us(self):
        return [u.copy() for u in self.us_]

    def rollout(self, x: np.ndarray, u: np.ndarray, stage_index: int = 0) -> np.ndarray:
        stage = self.ocp_.getProblem().stages[stage_index]
        integrator = aligator.dynamics.IntegratorSemiImplEuler(
            stage.dynamics.differential_dynamics,
            self.settings_.rollout_timestep,
        )
        control = np.asarray(u, dtype=float)
        x_next = np.asarray(x, dtype=float)
        for _ in range(self.rollout_substeps_):
            data = integrator.createData()
            integrator.forward(x_next, control, data)
            x_next = np.array(data.xnext)
        return x_next

    def _rotate_left(self, values):
        if values:
            values.append(values.pop(0))

    def generateCycleHorizon(self, contact_states: list[dict[str, bool]]) -> None:
        self.contact_states_ = [dict(state) for state in contact_states]
        self.cycle_horizon_.clear()
        self.cycle_horizon_data_.clear()
        self.foot_takeoff_times_.clear()
        self.foot_land_times_.clear()

        m = self.ocp_.getSize() // len(contact_states)
        for _ in range(m):
            self.contact_states_.extend(dict(state) for state in contact_states)

        for name in self.ee_names_:
            self.foot_takeoff_times_[name] = []
            self.foot_land_times_[name] = []
            for i in range(1, len(self.contact_states_)):
                if (not self.contact_states_[i][name]) and self.contact_states_[i - 1][name]:
                    self.foot_takeoff_times_[name].append(i + self.ocp_.getSize())
                if self.contact_states_[i][name] and (not self.contact_states_[i - 1][name]):
                    self.foot_land_times_[name].append(i + self.ocp_.getSize())
            if self.contact_states_[-1][name] and (not self.contact_states_[0][name]):
                self.foot_takeoff_times_[name].append(len(self.contact_states_) - 1 + self.ocp_.getSize())
            if (not self.contact_states_[-1][name]) and self.contact_states_[0][name]:
                self.foot_land_times_[name].append(len(self.contact_states_) - 1 + self.ocp_.getSize())

        self.foot_planner_.reset(self.contact_states_[0])

        previous_contacts = {name: True for name in self.ee_names_}
        for state in self.contact_states_:
            active_contacts = sum(1 for active in state.values() if active)
            force_ref = self.ocp_.getReferenceForce(
                0,
                self.ocp_.getRobot().getFootFrameName(0),
            )
            force_ref = np.asarray(force_ref, dtype=float)
            force_zero = np.zeros_like(force_ref)
            force_ref[:] = 0.0
            if active_contacts > 0:
                force_ref[2] = self.support_force_ / float(active_contacts)

            contact_poses = {}
            force_map = {}
            for name in self.ee_names_:
                foot_nb = self.robot.getFootNb(name)
                placement = self.data_.oMf[self.robot.getFootFrameId(foot_nb)]
                contact_poses[name] = pin.SE3(np.array(placement.rotation), np.array(placement.translation))
                force_map[name] = force_ref if state[name] else force_zero

            land_contacts = {}
            for name in self.ee_names_:
                land_contacts[name] = (not previous_contacts[name]) and state[name]

            stage = self.ocp_.createStage(state, contact_poses, force_map, land_contacts)
            self.cycle_horizon_.append(stage)
            self.cycle_horizon_data_.append(stage.createData())
            previous_contacts = dict(state)

    def updateCycleTiming(self, update_only_horizon: bool) -> None:
        horizon = self.ocp_.getSize()
        for name in self.ee_names_:
            for i in range(len(self.foot_land_times_[name])):
                if (not update_only_horizon) or self.foot_land_times_[name][i] < horizon:
                    self.foot_land_times_[name][i] -= 1
            while self.foot_land_times_[name] and self.foot_land_times_[name][0] < 0:
                self.foot_land_times_[name].pop(0)

            for i in range(len(self.foot_takeoff_times_[name])):
                if (not update_only_horizon) or self.foot_takeoff_times_[name][i] < horizon:
                    self.foot_takeoff_times_[name][i] -= 1
            while self.foot_takeoff_times_[name] and self.foot_takeoff_times_[name][0] < 0:
                self.foot_takeoff_times_[name].pop(0)

    def getHorizonContactStates(self):
        return [self.ocp_.getContactState(t) for t in range(self.ocp_.getSize())]

    def updateStateReferences(self, state_ref: np.ndarray) -> None:
        for time in range(self.ocp_.getSize()):
            self.ocp_.setReferenceState(time, state_ref[time])

    def updateStepTrackerReferences(self, state_ref: np.ndarray) -> None:
        self.updateStateReferences(state_ref)
        horizon_contact_states = self.getHorizonContactStates()
        velocity_base = state_ref[0, self.model_.nq : self.model_.nq + 6]
        for name in self.ee_names_:
            foot_nb = self.robot.getFootNb(name)
            foot_ref_frame_id = self.robot.getFootRefFrameId(foot_nb)
            base_frame_id = self.robot.getBaseFrameId()
            foot_frame_id = self.robot.getFootFrameId(foot_nb)
            self.foot_planner_.updateFootReference(
                name,
                foot_nb,
                horizon_contact_states,
                self.foot_land_times_[name],
                self.data_.oMf[foot_ref_frame_id].translation,
                self.data_.oMf[base_frame_id].translation,
                self.data_.oMf[foot_frame_id].translation,
                velocity_base,
            )

            pose = pin.SE3.Identity()
            for time in range(self.ocp_.getSize()):
                pose.translation = self.foot_planner_.getReference(name)[time]
                self.ocp_.setReferencePose(time, name, pose)

    def getReferencePose(self, t: int, ee_name: str) -> pin.SE3:
        return self.ocp_.getReferencePose(t, ee_name)

    def recedeWithCycle(self) -> None:
        problem = self.ocp_.getProblem()
        if self.now_ == self.WALKING or self.ocp_.getContactSupport(self.ocp_.getSize() - 1) < len(
            self.ee_names_
        ):
            if not self.cycle_horizon_:
                return

            problem.replaceStageCircular(self.cycle_horizon_[0])
            self.solver_.cycleProblem(problem, self.cycle_horizon_data_[0])

            self._rotate_left(self.cycle_horizon_)
            self._rotate_left(self.cycle_horizon_data_)
            self._rotate_left(self.contact_states_)

            for name in self.ee_names_:
                if (not self.contact_states_[-1][name]) and self.contact_states_[-2][name]:
                    self.foot_takeoff_times_[name].append(len(self.contact_states_) + self.ocp_.getSize())
                if self.contact_states_[-1][name] and (not self.contact_states_[-2][name]):
                    self.foot_land_times_[name].append(len(self.contact_states_) + self.ocp_.getSize())
            self.updateCycleTiming(False)
        else:
            problem.replaceStageCircular(self.standing_horizon_[0])
            self.solver_.cycleProblem(problem, self.standing_horizon_data_[0])

            self._rotate_left(self.standing_horizon_)
            self._rotate_left(self.standing_horizon_data_)

            self.updateCycleTiming(True)

    def iterate(self, x: np.ndarray, state_ref: np.ndarray) -> None:
        self._update_kinematics(x)
        self.recedeWithCycle()
        self.updateStepTrackerReferences(state_ref)

        self.x0_ = np.array(x, dtype=float)
        self.xs_.pop(0)
        self.xs_[0] = self.x0_.copy()
        self.xs_.append(self.xs_[-1].copy())

        self.us_.pop(0)
        self.us_.append(self.us_[-1].copy())

        self.ocp_.getProblem().x0_init = self.x0_

        self.solver_.run(self.ocp_.getProblem(), self.xs_, self.us_)

        self.xs_ = [np.array(xi) for xi in self.solver_.results.xs]
        self.us_ = [np.array(ui) for ui in self.solver_.results.us]
