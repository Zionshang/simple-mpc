from __future__ import annotations

import numpy as np
import pinocchio as pin

import aligator
import aligator.constraints as constraints
import aligator.dynamics as dynamics
import aligator.manifolds as manifolds

from .robot_handler import RobotDataHandler, RobotModelHandler


class KinodynamicsOCP:
    def __init__(self, settings: dict, model_handler: RobotModelHandler):
        self.settings_ = dict(settings)
        self.model_handler_ = model_handler
        self.model_ = model_handler.getModel()
        self.space_ = manifolds.MultibodyPhaseSpace(self.model_)
        self.nq_ = self.model_.nq
        self.nv_ = self.model_.nv
        self.ndx_ = self.space_.ndx
        self.force_size_ = int(self.settings_["force_size"])
        self.nu_ = self.nv_ - 6 + self.force_size_ * self.model_handler_.getFeetNb()
        self.x0_ = np.asarray(self.model_handler_.getReferenceState(), dtype=float).copy()
        self.control_ref_ = np.zeros(self.nu_)
        self.size_ = 0
        self.problem_: aligator.TrajOptProblem | None = None
        self.problem_initialized_ = False
        self.terminal_constraint_ = False
        self.terminal_dcm_residual_ = None

    def _foot_names(self) -> list[str]:
        return self.model_handler_.getFeetFrameNames()

    def _foot_ids(self) -> list[int]:
        return self.model_handler_.getFeetFrameIds()

    def _identity_pose_map(self) -> dict[str, pin.SE3]:
        return {name: pin.SE3.Identity() for name in self._foot_names()}

    def _get_cost_stack(self, t: int):
        if self.problem_ is None:
            raise RuntimeError("Create problem first!")
        if t < 0 or t >= self.getSize():
            raise RuntimeError("Stage index exceeds stage vector size")
        return self.problem_.stages[t].cost

    def _get_terminal_cost_stack(self):
        if self.problem_ is None:
            raise RuntimeError("Create problem first!")
        return self.problem_.term_cost

    def _get_stage_dynamics(self, t: int):
        if self.problem_ is None:
            raise RuntimeError("Create problem first!")
        return self.problem_.stages[t].dynamics.differential_dynamics

    def computeControlFromForces(self, force_refs: dict[str, np.ndarray]) -> None:
        for foot_nb, name in enumerate(self._foot_names()):
            force_ref = np.asarray(force_refs[name], dtype=float)
            if force_ref.shape[0] != self.force_size_:
                raise RuntimeError("force size in settings does not match reference force size")
            start = foot_nb * self.force_size_
            self.control_ref_[start : start + self.force_size_] = force_ref

    def createStage(
        self,
        contact_phase: dict[str, bool],
        contact_pose: dict[str, pin.SE3],
        contact_force: dict[str, np.ndarray],
        land_constraint: dict[str, bool],
    ):
        space = manifolds.MultibodyPhaseSpace(self.model_handler_.getModel())
        rcost = aligator.CostStack(space, self.nu_)
        contact_states = [bool(contact_phase[name]) for name in self._foot_names()]

        self.computeControlFromForces(contact_force)

        rcost.addCost(
            "state_cost",
            aligator.QuadraticStateCost(
                space,
                self.nu_,
                self.model_handler_.getReferenceState(),
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
                self.model_handler_.getModel(),
                np.zeros(6),
            )
            rcost.addCost(
                "centroidal_cost",
                aligator.QuadraticResidualCost(space, cent_mom, self.settings_["w_cent"]),
            )

        if self.settings_.get("centder_cost", False):
            centder_mom = aligator.CentroidalMomentumDerivativeResidual(
                space.ndx,
                self.model_handler_.getModel(),
                np.asarray(self.settings_["gravity"], dtype=float),
                contact_states,
                self.model_handler_.getFeetFrameIds(),
                self.force_size_,
            )
            rcost.addCost(
                "centroidal_derivative_cost",
                aligator.QuadraticResidualCost(space, centder_mom, self.settings_["w_centder"]),
            )

        for foot_nb, name in enumerate(self._foot_names()):
            frame_id = self.model_handler_.getFootFrameId(foot_nb)
            if self.force_size_ == 6:
                frame_residual = aligator.FramePlacementResidual(
                    space.ndx,
                    self.nu_,
                    self.model_handler_.getModel(),
                    contact_pose[name],
                    frame_id,
                )
            else:
                frame_residual = aligator.FrameTranslationResidual(
                    space.ndx,
                    self.nu_,
                    self.model_handler_.getModel(),
                    np.array(contact_pose[name].translation).copy(),
                    frame_id,
                )
            rcost.addCost(
                f"{name}_pose_cost",
                aligator.QuadraticResidualCost(space, frame_residual, self.settings_["w_frame"]),
            )

        ode = dynamics.KinodynamicsFwdDynamics(
            space,
            self.model_handler_.getModel(),
            np.asarray(self.settings_["gravity"], dtype=float),
            contact_states,
            self.model_handler_.getFeetFrameIds(),
            self.force_size_,
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
            if not contact_phase[name]:
                continue

            frame_vel = aligator.FrameVelocityResidual(
                space.ndx,
                self.nu_,
                self.model_handler_.getModel(),
                v_ref,
                self.model_handler_.getFootFrameId(foot_nb),
                pin.LOCAL,
            )
            if self.force_size_ == 6:
                if self.settings_.get("force_cone", False):
                    wrench_residual = aligator.CentroidalWrenchConeResidual(
                        space.ndx,
                        self.nu_,
                        foot_nb,
                        self.settings_["mu"],
                        self.settings_["Lfoot"],
                        self.settings_["Wfoot"],
                    )
                    stage.addConstraint(wrench_residual, constraints.NegativeOrthant())
                stage.addConstraint(frame_vel, constraints.EqualityConstraintSet())
            else:
                if self.settings_.get("force_cone", False):
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
                        self.model_handler_.getModel(),
                        np.array(contact_pose[name].translation).copy(),
                        self.model_handler_.getFootFrameId(foot_nb),
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
        space = manifolds.MultibodyPhaseSpace(self.model_handler_.getModel())
        term_cost = aligator.CostStack(space, self.nu_)
        term_cost.addCost(
            "state_cost",
            aligator.QuadraticStateCost(
                space,
                self.nu_,
                self.model_handler_.getReferenceState(),
                self.settings_["w_x"],
            ),
        )

        if self.settings_.get("term_cent_cost", False):
            cent_mom = aligator.CentroidalMomentumResidual(
                space.ndx,
                self.nu_,
                self.model_handler_.getModel(),
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
            self.model_handler_.getModel(),
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
        force_size: int,
        gravity: float,
        terminal_constraint: bool,
    ) -> None:
        if int(force_size) != self.force_size_:
            raise ValueError("force_size does not match settings['force_size']")

        self.x0_ = np.asarray(x0, dtype=float).copy()
        self.size_ = int(horizon)

        contact_phases = []
        contact_poses = []
        contact_forces = []

        force_ref = np.zeros(self.force_size_)
        force_ref[2] = -self.model_handler_.getMass() * float(gravity) / float(self.model_handler_.getFeetNb())

        contact_phase = {name: True for name in self._foot_names()}
        contact_pose = self._identity_pose_map()
        contact_force = {name: force_ref.copy() for name in self._foot_names()}

        for _ in range(self.size_):
            contact_phases.append(dict(contact_phase))
            contact_poses.append({name: pin.SE3(pose) for name, pose in contact_pose.items()})
            contact_forces.append({name: force.copy() for name, force in contact_force.items()})

        stage_models = self.createStages(contact_phases, contact_poses, contact_forces)
        self.problem_ = aligator.TrajOptProblem(self.x0_, stage_models, self.createTerminalCost())
        self.problem_initialized_ = True
        self.terminal_constraint_ = False
        self.terminal_dcm_residual_ = None

        if terminal_constraint:
            self.createTerminalConstraint(self.x0_[:3])

    def getSettings(self) -> dict:
        return dict(self.settings_)

    def getProblem(self):
        if self.problem_ is None:
            raise RuntimeError("Create problem first!")
        return self.problem_

    def getSize(self) -> int:
        return self.size_

    def getModelHandler(self) -> RobotModelHandler:
        return self.model_handler_

    def setReferencePose(self, t: int, ee_name: str, pose_ref: pin.SE3) -> None:
        qrc = self._get_cost_stack(t).getComponent(f"{ee_name}_pose_cost")
        residual = qrc.residual
        if self.force_size_ == 6:
            residual.setReference(pose_ref)
        else:
            residual.setReference(np.array(pose_ref.translation).copy())

    def setReferencePoses(self, t: int, pose_refs: dict[str, pin.SE3]) -> None:
        if len(pose_refs) != self.model_handler_.getFeetNb():
            raise RuntimeError("pose_refs size does not match number of end effectors")
        for ee_name in self._foot_names():
            self.setReferencePose(t, ee_name, pose_refs[ee_name])

    def setTerminalReferencePose(self, ee_name: str, pose_ref: pin.SE3) -> None:
        qrc = self._get_terminal_cost_stack().getComponent(f"{ee_name}_pose_cost")
        residual = qrc.residual
        if self.force_size_ == 6:
            residual.setReference(pose_ref)
        else:
            residual.setReference(np.array(pose_ref.translation).copy())

    def setReferenceForces(self, t: int, force_refs: dict[str, np.ndarray]) -> None:
        self.computeControlFromForces(force_refs)
        self.setReferenceControl(t, self.control_ref_)

    def setReferenceForce(self, t: int, ee_name: str, force_ref: np.ndarray) -> None:
        foot_id = self.model_handler_.getFootNb(ee_name)
        control_ref = self.getReferenceControl(t)
        start = foot_id * self.force_size_
        control_ref[start : start + self.force_size_] = np.asarray(force_ref, dtype=float)
        self.setReferenceControl(t, control_ref)

    def getReferenceForce(self, t: int, ee_name: str) -> np.ndarray:
        foot_id = self.model_handler_.getFootNb(ee_name)
        start = foot_id * self.force_size_
        return self.getReferenceControl(t)[start : start + self.force_size_].copy()

    def getReferencePose(self, t: int, ee_name: str) -> pin.SE3:
        qrc = self._get_cost_stack(t).getComponent(f"{ee_name}_pose_cost")
        residual = qrc.residual
        if self.force_size_ == 6:
            return pin.SE3(residual.getReference())
        pose_ref = pin.SE3.Identity()
        pose_ref.translation = np.asarray(residual.getReference(), dtype=float)
        return pose_ref

    def getVelocityBase(self, t: int) -> np.ndarray:
        return self.getReferenceState(t)[self.nq_ : self.nq_ + 6].copy()

    def setVelocityBase(self, t: int, velocity_base: np.ndarray) -> None:
        velocity_base = np.asarray(velocity_base, dtype=float)
        if velocity_base.shape[0] != 6:
            raise RuntimeError("velocity_base size should be 6")
        x_ref = self.getReferenceState(t)
        x_ref[self.nq_ : self.nq_ + 6] = velocity_base
        self.setReferenceState(t, x_ref)

    def getPoseBase(self, t: int) -> np.ndarray:
        return self.getReferenceState(t)[:7].copy()

    def setPoseBase(self, t: int, pose_base: np.ndarray) -> None:
        pose_base = np.asarray(pose_base, dtype=float)
        if pose_base.shape[0] != 7:
            raise RuntimeError("pose_base size should be 7")
        x_ref = self.getReferenceState(t)
        x_ref[:7] = pose_base
        self.setReferenceState(t, x_ref)

    def getProblemState(self, data_handler: RobotDataHandler) -> np.ndarray:
        return data_handler.getState()

    def setReferenceState(self, t: int, x_ref: np.ndarray) -> None:
        x_ref = np.asarray(x_ref, dtype=float)
        if x_ref.shape[0] != self.nq_ + self.nv_:
            raise AssertionError("x_ref not of the right size")
        qc = self._get_cost_stack(t).getComponent("state_cost")
        qc.target = x_ref.copy()

    def getReferenceState(self, t: int) -> np.ndarray:
        qc = self._get_cost_stack(t).getComponent("state_cost")
        return np.asarray(qc.target, dtype=float).copy()

    def setReferenceControl(self, t: int, u_ref: np.ndarray) -> None:
        qc = self._get_cost_stack(t).getComponent("control_cost")
        qc.target = np.asarray(u_ref, dtype=float).copy()

    def getReferenceControl(self, t: int) -> np.ndarray:
        qc = self._get_cost_stack(t).getComponent("control_cost")
        return np.asarray(qc.target, dtype=float).copy()

    def getContactSupport(self, t: int) -> int:
        contact_states = self._get_stage_dynamics(t).contact_states
        return int(sum(bool(contact) for contact in contact_states))

    def getContactState(self, t: int) -> list[bool]:
        return [bool(contact) for contact in self._get_stage_dynamics(t).contact_states]
