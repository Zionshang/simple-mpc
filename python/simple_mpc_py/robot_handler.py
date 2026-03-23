from __future__ import annotations

import numpy as np
import pinocchio as pin


class RobotModelHandler:
    def __init__(self, model: pin.Model, reference_configuration_name: str, base_frame_name: str):
        self._model = model.copy()
        self._base_frame_id = self._model.getFrameId(base_frame_name)
        qref = np.array(self._model.referenceConfigurations[reference_configuration_name])
        self._reference_state = np.concatenate((qref, np.zeros(self._model.nv)))
        self._mass = float(pin.computeTotalMass(self._model))
        self._feet_frame_names: list[str] = []
        self._feet_frame_ids: list[int] = []
        self._feet_ref_frame_ids: list[int] = []

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
        return self._add_foot_frames(foot_name, reference_parent_frame_name)

    def setFootReferencePlacement(self, foot_nb: int, parentframeMfootref: pin.SE3) -> None:
        self._model.frames[self._feet_ref_frame_ids[foot_nb]].placement = parentframeMfootref

    def getReferenceState(self) -> np.ndarray:
        return self._reference_state

    def getFootFrameName(self, foot_nb: int) -> str:
        return self._feet_frame_names[foot_nb]

    def getFeetNb(self) -> int:
        return len(self._feet_frame_ids)

    def getFootNb(self, foot_frame_name: str) -> int:
        return self._feet_frame_names.index(foot_frame_name)

    def getFeetFrameIds(self) -> list[int]:
        return list(self._feet_frame_ids)

    def getFeetFrameNames(self) -> list[str]:
        return list(self._feet_frame_names)

    def getBaseFrameId(self) -> int:
        return self._base_frame_id

    def getFootFrameId(self, foot_nb: int) -> int:
        return self._feet_frame_ids[foot_nb]

    def getFootRefFrameId(self, foot_nb: int) -> int:
        return self._feet_ref_frame_ids[foot_nb]

    def getMass(self) -> float:
        return self._mass

    def getModel(self) -> pin.Model:
        return self._model
