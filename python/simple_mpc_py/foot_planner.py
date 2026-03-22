from __future__ import annotations

import math

import numpy as np


def _sample_bezier(control_points: np.ndarray, u: float) -> np.ndarray:
    u = float(np.clip(u, 0.0, 1.0))
    order = control_points.shape[0] - 1
    sample = np.zeros(3)
    for i, point in enumerate(control_points):
        coeff = math.comb(order, i) * (u**i) * ((1.0 - u) ** (order - i))
        sample += coeff * point
    return sample


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
        self.swing_trajectories_: dict[str, np.ndarray] = {}
        self.previous_in_contact_: dict[str, bool] = {}
        self.references_: dict[str, list[np.ndarray]] = {}
        self.foot_land_positions_: dict[str, list[np.ndarray]] = {}
        self.previous_contact_states_: dict[str, bool] = {}

        for name, pose in starting_poses.items():
            p = np.asarray(pose, dtype=float).copy()
            self.references_[name] = [p.copy() for _ in range(self.T_)]
            self.initial_poses_[name] = p.copy()
            self.final_poses_[name] = p.copy()
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

    def computeFootLandingPose(self, foot_nb, data_handler, velocity_base: np.ndarray) -> np.ndarray:
        foot_ref = np.array(data_handler.getFootRefPose(foot_nb).translation).copy()
        base_pose = np.array(data_handler.getBaseFramePose().translation).copy()
        foot_pose = np.array(data_handler.getFootPose(foot_nb).translation).copy()

        twist_vect = np.zeros(2)
        twist_vect[0] = -(foot_ref[1] - base_pose[1])
        twist_vect[1] = foot_ref[0] - base_pose[0]

        next_pose = np.zeros(3)
        next_pose[:2] = foot_ref[:2]
        next_pose[:2] += (np.asarray(velocity_base[:2], dtype=float) + velocity_base[5] * twist_vect) * (
            self.T_fly_ + self.T_contact_
        ) * self.timestep_
        next_pose[2] = foot_pose[2]
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
        foot_nb: int,
        in_contact: bool,
        land_times: list[int],
        data_handler,
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
                    self.computeFootLandingPose(foot_nb, data_handler, velocity_base).copy()
                )
                should_commit_next = len(committed_land_positions) < len(land_times)

    def updateSwingTrajectory(
        self,
        ee_name: str,
        foot_nb: int,
        in_contact: bool,
        land_positions: list[np.ndarray],
        data_handler,
    ) -> None:
        current_pose = np.array(data_handler.getFootPose(foot_nb).translation).copy()
        if land_positions and (in_contact or self.previous_in_contact_[ee_name]):
            self.initial_poses_[ee_name] = current_pose
            self.final_poses_[ee_name] = np.asarray(land_positions[0], dtype=float).copy()
            self.swing_trajectories_[ee_name] = self.defineTranslationBezier(
                self.initial_poses_[ee_name],
                self.final_poses_[ee_name],
            )
            return

        if in_contact:
            self.initial_poses_[ee_name] = current_pose
            self.final_poses_[ee_name] = current_pose.copy()
            self.swing_trajectories_[ee_name] = self.defineTranslationBezier(
                self.initial_poses_[ee_name],
                self.final_poses_[ee_name],
            )

    def defineTranslationBezier(self, trans_init: np.ndarray, trans_final: np.ndarray) -> np.ndarray:
        trans_init = np.asarray(trans_init, dtype=float).copy()
        trans_final = np.asarray(trans_final, dtype=float).copy()

        points = []
        for _ in range(4):
            points.append(trans_init.copy())
        midpoint = trans_init * 3.0 / 4.0 + trans_final * 1.0 / 4.0
        midpoint[2] += self.swing_apex_
        points.append(midpoint.copy())
        for _ in range(5, 9):
            points.append(trans_final.copy())
        return np.asarray(points, dtype=float)

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

        current_trans = np.asarray(current_trans, dtype=float).copy()
        trajectory = [current_trans.copy() for _ in range(self.T_)]
        stance_pose = current_trans.copy()
        takeoff_id = 0
        land_id = 0
        swing_start_time = 0
        swing_trajectory = self.defineTranslationBezier(current_trans, current_trans)

        if (not in_contact) and land_poses:
            swing_start_time = int(land_times[0] - self.T_fly_)
            swing_trajectory = self.swing_trajectories_[ee_name]

        for t in range(self.T_):
            time = int(t)
            sample = stance_pose.copy()

            while land_id < len(land_times) and land_times[land_id] < time:
                stance_pose = np.asarray(land_poses[land_id], dtype=float).copy()
                in_contact = True
                land_id += 1

            if in_contact:
                takeoff_now = takeoff_id < len(takeoff_times) and takeoff_times[takeoff_id] <= time
                if takeoff_now:
                    swing_start_time = int(takeoff_times[takeoff_id])
                    in_contact = False
                    target_pose = np.asarray(land_poses[land_id], dtype=float).copy() if land_id < len(land_poses) else stance_pose
                    swing_trajectory = self.defineTranslationBezier(stance_pose, target_pose)
                    takeoff_id += 1

            if in_contact or land_id >= len(land_times):
                sample = stance_pose.copy()
            else:
                landing_time = int(land_times[land_id])
                landing_pose = np.asarray(land_poses[land_id], dtype=float).copy()
                if time >= landing_time:
                    stance_pose = landing_pose.copy()
                    in_contact = True
                    land_id += 1
                    sample = stance_pose.copy()
                else:
                    swing_duration = landing_time - swing_start_time
                    if swing_duration <= 0:
                        sample = landing_pose.copy()
                    else:
                        u = float(time - swing_start_time) / float(swing_duration)
                        sample = _sample_bezier(swing_trajectory, u)

            trajectory[t] = np.asarray(sample, dtype=float).copy()

        return trajectory

    def updateFootReference(
        self,
        ee_name: str,
        foot_nb: int,
        horizon_contact_states,
        future_land_times: list[int],
        data_handler,
        velocity_base: np.ndarray,
    ) -> None:
        in_contact = bool(horizon_contact_states[0][foot_nb])
        takeoff_times, land_times = self.extractPhaseTimings(horizon_contact_states, foot_nb, in_contact)
        self.appendPreviewLandTimes(in_contact, future_land_times, takeoff_times, land_times)
        self.updateCommittedLandPositions(
            ee_name,
            foot_nb,
            in_contact,
            land_times,
            data_handler,
            np.asarray(velocity_base, dtype=float),
        )

        land_positions = [np.asarray(position, dtype=float).copy() for position in self.foot_land_positions_[ee_name]]
        for _ in range(len(land_positions), len(land_times)):
            land_positions.append(self.computeFootLandingPose(foot_nb, data_handler, velocity_base))

        self.updateSwingTrajectory(ee_name, foot_nb, in_contact, land_positions, data_handler)
        self.references_[ee_name] = self.createTrajectory(
            ee_name,
            np.array(data_handler.getFootPose(foot_nb).translation).copy(),
            in_contact,
            takeoff_times,
            land_times,
            land_positions,
        )

        self.previous_contact_states_[ee_name] = in_contact
        self.previous_in_contact_[ee_name] = in_contact

    def getReference(self, ee_name: str) -> list[np.ndarray]:
        return self.references_[ee_name]
