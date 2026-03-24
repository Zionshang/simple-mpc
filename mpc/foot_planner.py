from __future__ import annotations

import ndcurves
import numpy as np


class FootPlanner:
    def __init__(
        self,
        starting_poses: dict[str, np.ndarray],
        swing_apex: float,
        T_fly: int,
        T_contact: int,
        T: int,
        timestep: float,
    ):
        self.swing_apex_ = float(swing_apex)
        self.T_fly_ = int(T_fly)
        self.T_contact_ = int(T_contact)
        self.T_ = int(T)
        self.timestep_ = float(timestep)

        self.initial_poses_: dict[str, np.ndarray] = {}
        self.final_poses_: dict[str, np.ndarray] = {}
        self.swing_trajectories_: dict[str, object] = {}
        self.previous_in_contact_: dict[str, bool] = {}
        self.references_: dict[str, list[np.ndarray]] = {}
        self.foot_land_positions_: dict[str, list[np.ndarray]] = {}
        self.previous_contact_states_: dict[str, bool] = {}

        for name, pose in starting_poses.items():
            p = np.array(pose, dtype=float)
            self.references_[name] = [p.copy() for _ in range(self.T_)]
            self.initial_poses_[name] = p
            self.final_poses_[name] = p
            self.swing_trajectories_[name] = self.defineTranslationBezier(p, p)
            self.previous_in_contact_[name] = True

        initial_contact_states = {name: True for name in starting_poses}
        self.reset(initial_contact_states)

    def reset(self, initial_contact_states: dict[str, bool]) -> None:
        self.foot_land_positions_.clear()
        self.previous_contact_states_.clear()
        for name, contact in initial_contact_states.items():
            self.foot_land_positions_[name] = []
            self.previous_contact_states_[name] = bool(contact)
            self.previous_in_contact_[name] = bool(contact)

    def computeFootLandingPose(
        self,
        foot_ref_position: np.ndarray,
        base_position: np.ndarray,
        foot_position: np.ndarray,
        velocity_base: np.ndarray,
    ) -> np.ndarray:
        twist_vect = np.zeros(2)
        twist_vect[0] = -(foot_ref_position[1] - base_position[1])
        twist_vect[1] = foot_ref_position[0] - base_position[0]

        next_pose = np.zeros(3)
        next_pose[:2] = foot_ref_position[:2]
        next_pose[:2] += (np.asarray(velocity_base[:2], dtype=float) + velocity_base[5] * twist_vect) * (
            self.T_fly_ + self.T_contact_
        ) * self.timestep_
        next_pose[2] = foot_ref_position[2]
        return next_pose

    def extractPhaseTimings(self, horizon_contact_states, foot_nb: int, in_contact: bool):
        takeoff_times = []
        land_times = []

        previous_contact = bool(in_contact)
        for time in range(1, len(horizon_contact_states)):
            contact = bool(horizon_contact_states[time][foot_nb])
            if (not contact) and previous_contact:
                takeoff_times.append(int(time))
            if contact and (not previous_contact):
                land_times.append(int(time))
            previous_contact = contact

        return takeoff_times, land_times

    def appendPreviewLandTimes(
        self,
        in_contact: bool,
        future_land_times: list[int],
        takeoff_times: list[int],
        land_times: list[int],
    ) -> None:
        required_land_count = len(takeoff_times) + (0 if in_contact else 1)
        for i in range(len(future_land_times)):
            if len(land_times) >= required_land_count:
                break
            future_land_time = int(future_land_times[i])
            if future_land_time > 0 and (not land_times or future_land_time > land_times[-1]):
                land_times.append(future_land_time)

    def updateCommittedLandPositions(
        self,
        ee_name: str,
        in_contact: bool,
        land_times: list[int],
        foot_ref_position: np.ndarray,
        base_position: np.ndarray,
        foot_position: np.ndarray,
        velocity_base: np.ndarray,
    ) -> None:
        committed_land_positions = self.foot_land_positions_[ee_name]
        if in_contact and (not self.previous_contact_states_[ee_name]) and committed_land_positions:
            committed_land_positions.pop(0)
        if len(committed_land_positions) > len(land_times):
            del committed_land_positions[len(land_times) :]

        should_commit_next = len(committed_land_positions) < len(land_times)
        while should_commit_next:
            land_id = len(committed_land_positions)
            should_commit_next = ((not in_contact) and land_id == 0) or (
                land_times[land_id] <= self.T_fly_ + self.T_contact_
            )
            if should_commit_next:
                committed_land_positions.append(
                    self.computeFootLandingPose(
                        foot_ref_position,
                        base_position,
                        foot_position,
                        velocity_base,
                    )
                )
                should_commit_next = len(committed_land_positions) < len(land_times)

    def updateSwingTrajectory(
        self,
        ee_name: str,
        in_contact: bool,
        land_positions: list[np.ndarray],
        current_pose: np.ndarray,
    ) -> None:
        current_pose = np.array(current_pose, dtype=float)
        if land_positions and (in_contact or self.previous_in_contact_[ee_name]):
            self.initial_poses_[ee_name] = current_pose
            self.final_poses_[ee_name] = np.array(land_positions[0], dtype=float)
            self.swing_trajectories_[ee_name] = self.defineTranslationBezier(
                self.initial_poses_[ee_name],
                self.final_poses_[ee_name],
            )
            return

        if in_contact:
            self.initial_poses_[ee_name] = current_pose
            self.final_poses_[ee_name] = current_pose
            self.swing_trajectories_[ee_name] = self.defineTranslationBezier(
                self.initial_poses_[ee_name],
                self.final_poses_[ee_name],
            )

    def defineTranslationBezier(self, trans_init: np.ndarray, trans_final: np.ndarray):
        trans_init = np.array(trans_init, dtype=float)
        trans_final = np.array(trans_final, dtype=float)

        points = []
        for _ in range(4):
            points.append(trans_init)
        midpoint = trans_init * 3.0 / 4.0 + trans_final * 1.0 / 4.0
        midpoint[2] += self.swing_apex_
        points.append(midpoint)
        for _ in range(5, 9):
            points.append(trans_final)

        control_points = np.column_stack(points)
        bezier_curve = ndcurves.bezier(control_points, 0.0, 1.0)
        curve = ndcurves.piecewise()
        curve.append(bezier_curve)
        return curve

    def createTrajectory(
        self,
        ee_name: str,
        current_trans: np.ndarray,
        in_contact: bool,
        takeoff_times: list[int],
        land_times: list[int],
        land_poses: list[np.ndarray],
    ) -> list[np.ndarray]:
        if len(land_times) != len(land_poses):
            raise RuntimeError("land_times size does not match land_poses size")

        current_trans = np.array(current_trans, dtype=float)
        trajectory = [None] * self.T_
        stance_pose = current_trans
        takeoff_id = 0
        land_id = 0
        swing_start_time = 0
        swing_trajectory = self.defineTranslationBezier(current_trans, current_trans)

        if (not in_contact) and land_poses:
            swing_start_time = int(land_times[0] - self.T_fly_)
            swing_trajectory = self.swing_trajectories_[ee_name]

        for t in range(self.T_):
            time = int(t)
            sample = stance_pose

            while land_id < len(land_times) and land_times[land_id] < time:
                stance_pose = np.array(land_poses[land_id], dtype=float)
                in_contact = True
                land_id += 1

            if in_contact:
                takeoff_now = takeoff_id < len(takeoff_times) and takeoff_times[takeoff_id] <= time
                if takeoff_now:
                    swing_start_time = int(takeoff_times[takeoff_id])
                    in_contact = False
                    if land_id < len(land_poses):
                        target_pose = np.array(land_poses[land_id], dtype=float)
                    else:
                        target_pose = stance_pose
                    swing_trajectory = self.defineTranslationBezier(stance_pose, target_pose)
                    takeoff_id += 1

            if in_contact or land_id >= len(land_times):
                sample = stance_pose
            else:
                landing_time = int(land_times[land_id])
                landing_pose = np.array(land_poses[land_id], dtype=float)
                if time >= landing_time:
                    stance_pose = landing_pose
                    in_contact = True
                    land_id += 1
                    sample = stance_pose
                else:
                    swing_duration = landing_time - swing_start_time
                    if swing_duration <= 0:
                        sample = landing_pose
                    else:
                        u = float(time - swing_start_time) / float(swing_duration)
                        u = min(max(u, 0.0), 1.0)
                        sample = np.array(swing_trajectory(u), dtype=float)

            trajectory[t] = np.array(sample, dtype=float, copy=True)

        return trajectory

    def updateFootReference(
        self,
        ee_name: str,
        foot_nb: int,
        horizon_contact_states,
        future_land_times: list[int],
        foot_ref_position: np.ndarray,
        base_position: np.ndarray,
        foot_position: np.ndarray,
        velocity_base: np.ndarray,
    ) -> None:
        in_contact = bool(horizon_contact_states[0][foot_nb])
        takeoff_times, land_times = self.extractPhaseTimings(horizon_contact_states, foot_nb, in_contact)
        self.appendPreviewLandTimes(in_contact, future_land_times, takeoff_times, land_times)
        self.updateCommittedLandPositions(
            ee_name,
            in_contact,
            land_times,
            foot_ref_position,
            base_position,
            foot_position,
            np.asarray(velocity_base, dtype=float),
        )

        land_positions = list(self.foot_land_positions_[ee_name])
        for _ in range(len(land_positions), len(land_times)):
            land_positions.append(
                self.computeFootLandingPose(
                    foot_ref_position,
                    base_position,
                    foot_position,
                    velocity_base,
                )
            )

        self.updateSwingTrajectory(ee_name, in_contact, land_positions, foot_position)
        self.references_[ee_name] = self.createTrajectory(
            ee_name,
            foot_position,
            in_contact,
            takeoff_times,
            land_times,
            land_positions,
        )

        self.previous_contact_states_[ee_name] = in_contact
        self.previous_in_contact_[ee_name] = in_contact

    def getReference(self, ee_name: str) -> list[np.ndarray]:
        return self.references_[ee_name]
