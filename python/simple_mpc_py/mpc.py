from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pinocchio as pin

import aligator

from .foot_planner import FootPlanner


@dataclass
class MPCSettings:
    support_force: float = 1000.0
    TOL: float = 1e-4
    mu_init: float = 1e-8
    max_iters: int = 1
    num_threads: int = 1
    swing_apex: float = 0.15
    T_fly: int = 80
    T_contact: int = 20
    timestep: float = 0.01

    @classmethod
    def from_dict(cls, settings: dict) -> "MPCSettings":
        return cls(**settings)


class MPC:
    WALKING = "walking"
    STANDING = "standing"

    def __init__(self, settings: dict, problem):
        self.settings_ = MPCSettings.from_dict(settings)
        self.ocp_handler_ = problem

        robot = self.ocp_handler_.getRobot()
        self.robot = robot
        self.model_ = robot.getModel()
        self.data_ = self.model_.createData()
        self._update_kinematics(robot.getReferenceState(), update_com=True)

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
            self.ocp_handler_.getSize(),
            self.settings_.timestep,
        )

        self.x0_ = np.array(robot.getReferenceState(), dtype=float)
        self.x_reference_ = self.ocp_handler_.getReferenceState(0)

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
        force_ref = self.ocp_handler_.getReferenceForce(0, robot.getFootFrameName(0))

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

        for _ in range(len(self.ocp_handler_.getProblem().stages)):
            self.xs_.append(self.x0_.copy())
            self.us_.append(self.ocp_handler_.getReferenceControl(0))
            stage = self.ocp_handler_.createStage(contact_states, contact_poses, force_map, land_constraint)
            self.standing_horizon_.append(stage)
            self.standing_horizon_data_.append(stage.createData())
        self.xs_.append(self.x0_.copy())

        self.solver_.setup(self.ocp_handler_.getProblem())
        self.solver_.run(self.ocp_handler_.getProblem(), self.xs_, self.us_)

        self.xs_ = [np.array(x) for x in self.solver_.results.xs]
        self.us_ = [np.array(u) for u in self.solver_.results.us]

        self.solver_.max_iters = self.settings_.max_iters

        self.com0_ = np.array(self.data_.com[0])
        self.now_ = self.WALKING
        self.velocity_base_ = np.zeros(6)

    def _update_kinematics(self, x: np.ndarray, update_com: bool = False) -> None:
        x = np.asarray(x, dtype=float)
        q = x[: self.model_.nq]
        v = x[self.model_.nq :]
        pin.forwardKinematics(self.model_, self.data_, q, v)
        pin.updateFramePlacements(self.model_, self.data_)
        if update_com:
            pin.centerOfMass(self.model_, self.data_, q, v)

    @property
    def velocity_base(self):
        return self.velocity_base_

    @velocity_base.setter
    def velocity_base(self, value):
        self.velocity_base_ = np.asarray(value, dtype=float).copy()

    @property
    def ocp_handler(self):
        return self.ocp_handler_

    @property
    def xs(self):
        return [x.copy() for x in self.xs_]

    @property
    def us(self):
        return [u.copy() for u in self.us_]

    def _rotate_left(self, values):
        if values:
            values.append(values.pop(0))

    def generateCycleHorizon(self, contact_states: list[dict[str, bool]]) -> None:
        self.contact_states_ = [dict(state) for state in contact_states]
        self.cycle_horizon_.clear()
        self.cycle_horizon_data_.clear()
        self.foot_takeoff_times_.clear()
        self.foot_land_times_.clear()

        m = self.ocp_handler_.getSize() // len(contact_states)
        for _ in range(m):
            self.contact_states_.extend(dict(state) for state in contact_states)

        for name in self.ee_names_:
            self.foot_takeoff_times_[name] = []
            self.foot_land_times_[name] = []
            for i in range(1, len(self.contact_states_)):
                if (not self.contact_states_[i][name]) and self.contact_states_[i - 1][name]:
                    self.foot_takeoff_times_[name].append(i + self.ocp_handler_.getSize())
                if self.contact_states_[i][name] and (not self.contact_states_[i - 1][name]):
                    self.foot_land_times_[name].append(i + self.ocp_handler_.getSize())
            if self.contact_states_[-1][name] and (not self.contact_states_[0][name]):
                self.foot_takeoff_times_[name].append(len(self.contact_states_) - 1 + self.ocp_handler_.getSize())
            if (not self.contact_states_[-1][name]) and self.contact_states_[0][name]:
                self.foot_land_times_[name].append(len(self.contact_states_) - 1 + self.ocp_handler_.getSize())

        self.foot_planner_.reset(self.contact_states_[0])

        previous_contacts = {name: True for name in self.ee_names_}
        for state in self.contact_states_:
            active_contacts = sum(1 for active in state.values() if active)
            force_ref = self.ocp_handler_.getReferenceForce(
                0,
                self.ocp_handler_.getRobot().getFootFrameName(0),
            )
            force_ref = np.asarray(force_ref, dtype=float)
            force_zero = np.zeros_like(force_ref)
            force_ref[:] = 0.0
            if active_contacts > 0:
                force_ref[2] = self.settings_.support_force / float(active_contacts)

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

            stage = self.ocp_handler_.createStage(state, contact_poses, force_map, land_contacts)
            self.cycle_horizon_.append(stage)
            self.cycle_horizon_data_.append(stage.createData())
            previous_contacts = dict(state)

    def updateCycleTiming(self, update_only_horizon: bool) -> None:
        horizon = self.ocp_handler_.getSize()
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
        return [self.ocp_handler_.getContactState(t) for t in range(self.ocp_handler_.getSize())]

    def updateTerminalReferences(self) -> None:
        horizon = self.ocp_handler_.getSize()
        self.ocp_handler_.setReferenceState(horizon - 1, self.x_reference_)
        self.ocp_handler_.setVelocityBase(horizon - 1, self.velocity_base_)

        com_ref = np.zeros(3)
        for name in self.ee_names_:
            com_ref += self.foot_planner_.getReference(name)[-1]
        com_ref /= float(len(self.ee_names_))
        com_ref[2] += self.com0_[2]
        self.ocp_handler_.updateTerminalConstraint(com_ref)

    def updateStepTrackerReferences(self) -> None:
        horizon_contact_states = self.getHorizonContactStates()
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
                self.velocity_base_,
            )

            pose = pin.SE3.Identity()
            for time in range(self.ocp_handler_.getSize()):
                pose.translation = self.foot_planner_.getReference(name)[time]
                self.ocp_handler_.setReferencePose(time, name, pose)

        self.updateTerminalReferences()

    def getReferencePose(self, t: int, ee_name: str) -> pin.SE3:
        return self.ocp_handler_.getReferencePose(t, ee_name)

    def recedeWithCycle(self) -> None:
        problem = self.ocp_handler_.getProblem()
        if self.now_ == self.WALKING or self.ocp_handler_.getContactSupport(self.ocp_handler_.getSize() - 1) < len(
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
                    self.foot_takeoff_times_[name].append(len(self.contact_states_) + self.ocp_handler_.getSize())
                if self.contact_states_[-1][name] and (not self.contact_states_[-2][name]):
                    self.foot_land_times_[name].append(len(self.contact_states_) + self.ocp_handler_.getSize())
            self.updateCycleTiming(False)
        else:
            problem.replaceStageCircular(self.standing_horizon_[0])
            self.solver_.cycleProblem(problem, self.standing_horizon_data_[0])

            self._rotate_left(self.standing_horizon_)
            self._rotate_left(self.standing_horizon_data_)

            self.updateCycleTiming(True)

    def iterate(self, x: np.ndarray) -> None:
        self._update_kinematics(x)
        self.recedeWithCycle()
        self.updateStepTrackerReferences()

        self.x0_ = np.array(x, dtype=float)
        self.xs_.pop(0)
        self.xs_[0] = self.x0_.copy()
        self.xs_.append(self.xs_[-1].copy())

        self.us_.pop(0)
        self.us_.append(self.us_[-1].copy())

        self.ocp_handler_.getProblem().x0_init = self.x0_

        self.solver_.run(self.ocp_handler_.getProblem(), self.xs_, self.us_)

        self.xs_ = [np.array(xi) for xi in self.solver_.results.xs]
        self.us_ = [np.array(ui) for ui in self.solver_.results.us]
