from __future__ import annotations

import copy

import numpy as np
import pinocchio as pin


def _copy_model(model: pin.Model) -> pin.Model:
    if hasattr(model, "copy"):
        return model.copy()
    return copy.deepcopy(model)


class RobotModelHandler:
    POINT = "point"
    QUAD = "quad"

    def __init__(self, model: pin.Model, reference_configuration_name: str, base_frame_name: str):
        self._model = _copy_model(model)
        self._base_frame_id = self._model.getFrameId(base_frame_name)
        qref = np.array(self._model.referenceConfigurations[reference_configuration_name]).copy()
        self._reference_state = np.concatenate((qref, np.zeros(self._model.nv)))
        self._mass = float(pin.computeTotalMass(self._model))
        self._feet_frame_names: list[str] = []
        self._feet_types: list[str] = []
        self._feet_frame_ids: list[int] = []
        self._feet_ref_frame_ids: list[int] = []
        self._feet_contact_points: dict[int, np.ndarray] = {}

    def _add_foot_frames(self, foot_name: str, reference_parent_frame_name: str) -> int:
        foot_nb = self.getFeetNb()
        foot_frame_id = self._model.getFrameId(foot_name)
        self._feet_frame_names.append(foot_name)
        self._feet_frame_ids.append(foot_frame_id)

        reference_parent_frame_id = self._model.getFrameId(reference_parent_frame_name)
        parent_joint = self._model.frames[reference_parent_frame_id].parentJoint
        frame = pin.Frame(
            f"{foot_name}_ref",
            parent_joint,
            reference_parent_frame_id,
            pin.SE3.Identity(),
            pin.OP_FRAME,
        )
        ref_frame_id = self._model.addFrame(frame)
        self._feet_ref_frame_ids.append(ref_frame_id)

        data = self._model.createData()
        qref = self._reference_state[: self._model.nq]
        pin.forwardKinematics(self._model, data, qref)
        pin.updateFramePlacements(self._model, data)
        default_placement = data.oMf[reference_parent_frame_id].actInv(data.oMf[foot_frame_id])
        self.setFootReferencePlacement(foot_nb, default_placement)
        return foot_nb

    def addPointFoot(self, foot_name: str, reference_parent_frame_name: str) -> int:
        foot_nb = self._add_foot_frames(foot_name, reference_parent_frame_name)
        self._feet_types.append(self.POINT)
        return foot_nb

    def addQuadFoot(
        self,
        foot_name: str,
        reference_parent_frame_name: str,
        contact_points: np.ndarray,
    ) -> int:
        foot_nb = self._add_foot_frames(foot_name, reference_parent_frame_name)
        self._feet_types.append(self.QUAD)
        self._feet_contact_points[foot_nb] = np.array(contact_points, dtype=float).copy()
        return foot_nb

    def setFootReferencePlacement(self, foot_nb: int, parentframeMfootref: pin.SE3) -> None:
        self._model.frames[self._feet_ref_frame_ids[foot_nb]].placement = parentframeMfootref

    def difference(self, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
        x1 = np.asarray(x1)
        x2 = np.asarray(x2)
        nq = self._model.nq
        nv = self._model.nv
        dx = np.zeros(2 * nv)
        dx[:nv] = pin.difference(self._model, x1[:nq], x2[:nq])
        dx[nv:] = x2[nq:] - x1[nq:]
        return dx

    def getReferenceState(self) -> np.ndarray:
        return self._reference_state

    def getFootFrameName(self, foot_nb: int) -> str:
        return self._feet_frame_names[foot_nb]

    def getFeetNb(self) -> int:
        return len(self._feet_frame_ids)

    def getFootNb(self, foot_frame_name: str) -> int:
        return self._feet_frame_names.index(foot_frame_name)

    def getFootType(self, foot_nb: int) -> str:
        return self._feet_types[foot_nb]

    def getQuadFootContactPoints(self, foot_nb: int) -> np.ndarray:
        return self._feet_contact_points[foot_nb]

    def getFeetFrameIds(self) -> list[int]:
        return list(self._feet_frame_ids)

    def getFeetFrameNames(self) -> list[str]:
        return list(self._feet_frame_names)

    def getBaseFrameId(self) -> int:
        return self._base_frame_id

    def getBaseFrameName(self) -> str:
        return self._model.frames[self._base_frame_id].name

    def getFootFrameId(self, foot_nb: int) -> int:
        return self._feet_frame_ids[foot_nb]

    def getFootRefFrameId(self, foot_nb: int) -> int:
        return self._feet_ref_frame_ids[foot_nb]

    def getMass(self) -> float:
        return self._mass

    def getModel(self) -> pin.Model:
        return self._model


class RobotDataHandler:
    def __init__(self, model_handler: RobotModelHandler):
        self._model_handler = model_handler
        self._data = model_handler.getModel().createData()
        self._x = np.zeros(model_handler.getReferenceState().shape[0])
        self.updateInternalData(model_handler.getReferenceState(), True)

    def updateInternalData(self, q_or_x, v_or_updateJacobians=None, updateJacobians: bool = False) -> None:
        model = self._model_handler.getModel()
        if v_or_updateJacobians is None or isinstance(v_or_updateJacobians, (bool, np.bool_)):
            x = np.asarray(q_or_x, dtype=float)
            updateJacobians = bool(False if v_or_updateJacobians is None else v_or_updateJacobians)
            q = x[: model.nq]
            v = x[model.nq :]
        else:
            q = np.asarray(q_or_x, dtype=float)
            v = np.asarray(v_or_updateJacobians, dtype=float)
        self._x = np.concatenate((q, v))

        pin.forwardKinematics(model, self._data, q, v)
        pin.updateFramePlacements(model, self._data)
        pin.centerOfMass(model, self._data, q, v)
        pin.computeCentroidalMomentum(model, self._data, q, v)
        if updateJacobians:
            self.updateJacobiansMassMatrix(self._x)

    def updateJacobiansMassMatrix(self, x: np.ndarray) -> None:
        model = self._model_handler.getModel()
        q = np.asarray(x[: model.nq], dtype=float)
        v = np.asarray(x[model.nq :], dtype=float)
        pin.computeJointJacobians(model, self._data, q)
        pin.computeJointJacobiansTimeVariation(model, self._data, q, v)
        pin.crba(model, self._data, q)
        self._data.M[np.tril_indices(model.nv, -1)] = self._data.M.T[np.tril_indices(model.nv, -1)]
        pin.nonLinearEffects(model, self._data, q, v)
        pin.computeCentroidalMapTimeVariation(model, self._data, q, v)

    def getFootRefPose(self, foot_nb: int) -> pin.SE3:
        return self._data.oMf[self._model_handler.getFootRefFrameId(foot_nb)]

    def getFootPose(self, foot_nb: int) -> pin.SE3:
        return self._data.oMf[self._model_handler.getFootFrameId(foot_nb)]

    def getBaseFramePose(self) -> pin.SE3:
        return self._data.oMf[self._model_handler.getBaseFrameId()]

    def getModelHandler(self) -> RobotModelHandler:
        return self._model_handler

    def getData(self):
        return self._data

    def getState(self) -> np.ndarray:
        return self._x.copy()

    def getCentroidalState(self) -> np.ndarray:
        x_centroidal = np.zeros(9)
        x_centroidal[:3] = np.array(self._data.com[0]).copy()
        x_centroidal[3:6] = np.array(self._data.hg.linear).copy()
        x_centroidal[6:9] = np.array(self._data.hg.angular).copy()
        return x_centroidal
