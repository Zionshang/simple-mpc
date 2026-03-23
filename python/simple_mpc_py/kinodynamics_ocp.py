from __future__ import annotations

import numpy as np
import pinocchio as pin

import aligator
import aligator.constraints as constraints
import aligator.dynamics as dynamics
import aligator.manifolds as manifolds

from .robot_handler import QuadRobot


FORCE_SIZE = 3


class QuadKinodynOcp:
    def __init__(self, settings: dict, robot: QuadRobot):
        self.settings_ = dict(settings)
        self.robot = robot
        self.model_ = robot.getModel()
        self.space_ = manifolds.MultibodyPhaseSpace(self.model_)
        self.nq_ = self.model_.nq
        self.nv_ = self.model_.nv
        self.ndx_ = self.space_.ndx
        self.nu_ = self.nv_ - 6 + FORCE_SIZE * self.robot.getFeetNb()
        self.control_ref_ = np.zeros(self.nu_)
        self.size_ = 0
        self.problem_: aligator.TrajOptProblem | None = None
        self.problem_initialized_ = False
        self.terminal_constraint_ = False
        self.terminal_dcm_residual_ = None

    def _foot_names(self) -> list[str]:
        return self.robot.getFeetFrameNames()

    def _identity_pose_map(self) -> dict[str, pin.SE3]:
        return {name: pin.SE3.Identity() for name in self._foot_names()}

    def _get_cost_stack(self, t: int):
        if self.problem_ is None:
            raise RuntimeError("Create problem first!")
        if t < 0 or t >= self.getSize():
            raise RuntimeError("Stage index exceeds stage vector size")
        return self.problem_.stages[t].cost

    def _get_stage_dynamics(self, t: int):
        if self.problem_ is None:
            raise RuntimeError("Create problem first!")
        return self.problem_.stages[t].dynamics.differential_dynamics

    def computeControlFromForces(self, force_refs: dict[str, np.ndarray]) -> None:
        for foot_nb, name in enumerate(self._foot_names()):
            force_ref = np.asarray(force_refs[name], dtype=float)
            if force_ref.shape[0] != FORCE_SIZE:
                raise RuntimeError("force size does not match 3D point-foot model")
            start = foot_nb * FORCE_SIZE
            self.control_ref_[start : start + FORCE_SIZE] = force_ref

    def createStage(
        self,
        contact_phase: dict[str, bool],
        contact_pose: dict[str, pin.SE3],
        contact_force: dict[str, np.ndarray],
        land_constraint: dict[str, bool],
    ):
        space = manifolds.MultibodyPhaseSpace(self.robot.getModel())
        rcost = aligator.CostStack(space, self.nu_)
        contact_states = [bool(contact_phase[name]) for name in self._foot_names()]

        self.computeControlFromForces(contact_force)

        rcost.addCost(
            "state_cost",
            aligator.QuadraticStateCost(
                space,
                self.nu_,
                self.robot.getReferenceState(),
                self.settings_["w_x"],
            ),
        )
        rcost.addCost(
            "control_cost",
            aligator.QuadraticControlCost(space, self.control_ref_, self.settings_["w_u"]),
        )

        if self.settings_.get("cent_cost", False):
            cent_mom = aligator.CentroidalMomentumResidual(
                space.ndx,
                self.nu_,
                self.robot.getModel(),
                np.zeros(6),
            )
            rcost.addCost(
                "centroidal_cost",
                aligator.QuadraticResidualCost(space, cent_mom, self.settings_["w_cent"]),
            )

        if self.settings_.get("centder_cost", False):
            centder_mom = aligator.CentroidalMomentumDerivativeResidual(
                space.ndx,
                self.robot.getModel(),
                np.asarray(self.settings_["gravity"], dtype=float),
                contact_states,
                self.robot.getFeetFrameIds(),
                FORCE_SIZE,
            )
            rcost.addCost(
                "centroidal_derivative_cost",
                aligator.QuadraticResidualCost(space, centder_mom, self.settings_["w_centder"]),
            )

        for foot_nb, name in enumerate(self._foot_names()):
            frame_id = self.robot.getFootFrameId(foot_nb)
            frame_residual = aligator.FrameTranslationResidual(
                space.ndx,
                self.nu_,
                self.robot.getModel(),
                np.array(contact_pose[name].translation),
                frame_id,
            )
            rcost.addCost(
                f"{name}_pose_cost",
                aligator.QuadraticResidualCost(space, frame_residual, self.settings_["w_frame"]),
            )

        ode = dynamics.KinodynamicsFwdDynamics(
            space,
            self.robot.getModel(),
            np.asarray(self.settings_["gravity"], dtype=float),
            contact_states,
            self.robot.getFeetFrameIds(),
            FORCE_SIZE,
        )
        dyn_model = dynamics.IntegratorSemiImplEuler(ode, float(self.settings_["timestep"]))
        stage = aligator.StageModel(rcost, dyn_model)

        if self.settings_.get("kinematics_limits", False):
            state_fn = aligator.StateErrorResidual(space, self.nu_, space.neutral())
            state_id = list(range(6, self.nv_))
            state_slice = aligator.StageFunctionSliceXpr(state_fn, state_id)
            stage.addConstraint(
                state_slice,
                constraints.BoxConstraint(
                    np.asarray(self.settings_["qmin"], dtype=float),
                    np.asarray(self.settings_["qmax"], dtype=float),
                ),
            )

        v_ref = pin.Motion.Zero()
        for foot_nb, name in enumerate(self._foot_names()):
            if contact_phase[name]:
                frame_vel = aligator.FrameVelocityResidual(
                    space.ndx,
                    self.nu_,
                    self.robot.getModel(),
                    v_ref,
                    self.robot.getFootFrameId(foot_nb),
                    pin.LOCAL,
                )
                friction_residual = aligator.CentroidalFrictionConeResidual(
                    space.ndx,
                    self.nu_,
                    foot_nb,
                    self.settings_["mu"],
                    1e-4,
                )
                stage.addConstraint(friction_residual, constraints.NegativeOrthant())
                vel_slice = aligator.StageFunctionSliceXpr(frame_vel, [0, 1, 2])
                stage.addConstraint(vel_slice, constraints.EqualityConstraintSet())

                if self.settings_.get("land_cstr", False) and land_constraint[name]:
                    frame_residual = aligator.FrameTranslationResidual(
                        space.ndx,
                        self.nu_,
                        self.robot.getModel(),
                        np.array(contact_pose[name].translation),
                        self.robot.getFootFrameId(foot_nb),
                    )
                    frame_slice = aligator.StageFunctionSliceXpr(frame_residual, [2])
                    stage.addConstraint(frame_slice, constraints.EqualityConstraintSet())

        return stage

    def createStages(
        self,
        contact_phases: list[dict[str, bool]],
        contact_poses: list[dict[str, pin.SE3]],
        contact_forces: list[dict[str, np.ndarray]],
    ):
        if len(contact_phases) != len(contact_poses):
            raise RuntimeError("Contact phases and poses sequences do not have the same size")
        if len(contact_phases) != len(contact_forces):
            raise RuntimeError("Contact phases and forces sequences do not have the same size")

        previous_phases = {name: True for name in self._foot_names()}
        stage_models = []
        for i in range(len(contact_phases)):
            land_constraint = {}
            for name in self._foot_names():
                land_constraint[name] = (not previous_phases[name]) and contact_phases[i][name]
            stage_models.append(
                self.createStage(
                    contact_phases[i],
                    contact_poses[i],
                    contact_forces[i],
                    land_constraint,
                )
            )
            previous_phases = dict(contact_phases[i])
        return stage_models

    def createTerminalCost(self):
        space = manifolds.MultibodyPhaseSpace(self.robot.getModel())
        term_cost = aligator.CostStack(space, self.nu_)
        term_cost.addCost(
            "state_cost",
            aligator.QuadraticStateCost(
                space,
                self.nu_,
                self.robot.getReferenceState(),
                self.settings_["w_x"],
            ),
        )

        if self.settings_.get("term_cent_cost", False):
            cent_mom = aligator.CentroidalMomentumResidual(
                space.ndx,
                self.nu_,
                self.robot.getModel(),
                np.zeros(6),
            )
            term_cost.addCost(
                "centroidal_cost",
                aligator.QuadraticResidualCost(space, cent_mom, self.settings_["w_cent"] * 10.0),
            )

        return term_cost

    def createTerminalConstraint(self, com_ref: np.ndarray) -> None:
        if not self.problem_initialized_ or self.problem_ is None:
            raise RuntimeError("Create problem first!")
        if not self.settings_.get("term_dcm_cstr", False):
            self.terminal_constraint_ = False
            self.terminal_dcm_residual_ = None
            return

        com_ref = np.asarray(com_ref, dtype=float)
        tau = float(np.sqrt(com_ref[2] / 9.81))
        self.terminal_dcm_residual_ = aligator.DCMPositionResidual(
            self.ndx_,
            self.nu_,
            self.robot.getModel(),
            com_ref,
            tau,
        )
        self.problem_.addTerminalConstraint(
            self.terminal_dcm_residual_,
            constraints.EqualityConstraintSet(),
        )
        self.terminal_constraint_ = True

    def updateTerminalConstraint(self, com_ref: np.ndarray) -> None:
        if self.settings_.get("term_dcm_cstr", False) and self.terminal_constraint_:
            self.terminal_dcm_residual_.setReference(np.asarray(com_ref, dtype=float))

    def createProblem(
        self,
        x0: np.ndarray,
        horizon: int,
        gravity: float,
        terminal_constraint: bool,
    ) -> None:
        self.size_ = int(horizon)

        contact_phases = []
        contact_poses = []
        contact_forces = []

        force_ref = np.zeros(FORCE_SIZE)
        force_ref[2] = -self.robot.getMass() * float(gravity) / float(self.robot.getFeetNb())

        contact_phase = {name: True for name in self._foot_names()}
        contact_pose = self._identity_pose_map()
        contact_force = {name: force_ref for name in self._foot_names()}

        for _ in range(self.size_):
            contact_phases.append(dict(contact_phase))
            contact_poses.append({name: pin.SE3(pose) for name, pose in contact_pose.items()})
            contact_forces.append(dict(contact_force))

        stage_models = self.createStages(contact_phases, contact_poses, contact_forces)
        self.problem_ = aligator.TrajOptProblem(x0, stage_models, self.createTerminalCost())
        self.problem_initialized_ = True
        self.terminal_constraint_ = False
        self.terminal_dcm_residual_ = None

        if terminal_constraint:
            self.createTerminalConstraint(x0[:3])

    def getProblem(self):
        if self.problem_ is None:
            raise RuntimeError("Create problem first!")
        return self.problem_

    def getSize(self) -> int:
        return self.size_

    def getRobot(self) -> QuadRobot:
        return self.robot

    def setReferencePose(self, t: int, ee_name: str, pose_ref: pin.SE3) -> None:
        qrc = self._get_cost_stack(t).getComponent(f"{ee_name}_pose_cost")
        qrc.residual.setReference(np.array(pose_ref.translation))

    def getReferenceForce(self, t: int, ee_name: str) -> np.ndarray:
        foot_id = self.robot.getFootNb(ee_name)
        start = foot_id * FORCE_SIZE
        return self.getReferenceControl(t)[start : start + FORCE_SIZE].copy()

    def getReferencePose(self, t: int, ee_name: str) -> pin.SE3:
        qrc = self._get_cost_stack(t).getComponent(f"{ee_name}_pose_cost")
        pose_ref = pin.SE3.Identity()
        pose_ref.translation = np.asarray(qrc.residual.getReference(), dtype=float)
        return pose_ref

    def setVelocityBase(self, t: int, velocity_base: np.ndarray) -> None:
        velocity_base = np.asarray(velocity_base, dtype=float)
        if velocity_base.shape[0] != 6:
            raise RuntimeError("velocity_base size should be 6")
        x_ref = self.getReferenceState(t)
        x_ref[self.nq_ : self.nq_ + 6] = velocity_base
        self.setReferenceState(t, x_ref)

    def setReferenceState(self, t: int, x_ref: np.ndarray) -> None:
        x_ref = np.asarray(x_ref, dtype=float)
        if x_ref.shape[0] != self.nq_ + self.nv_:
            raise AssertionError("x_ref not of the right size")
        self._get_cost_stack(t).getComponent("state_cost").target = x_ref.copy()

    def getReferenceState(self, t: int) -> np.ndarray:
        qc = self._get_cost_stack(t).getComponent("state_cost")
        return np.asarray(qc.target, dtype=float).copy()

    def getReferenceControl(self, t: int) -> np.ndarray:
        qc = self._get_cost_stack(t).getComponent("control_cost")
        return np.asarray(qc.target, dtype=float).copy()

    def getContactSupport(self, t: int) -> int:
        contact_states = self._get_stage_dynamics(t).contact_states
        return int(sum(bool(contact) for contact in contact_states))

    def getContactState(self, t: int) -> list[bool]:
        return [bool(contact) for contact in self._get_stage_dynamics(t).contact_states]
