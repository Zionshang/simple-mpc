///////////////////////////////////////////////////////////////////////////////
// BSD 2-Clause License
//
// Copyright (C) 2024, INRIA
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "simple-mpc/foot-planner.hpp"
#include "simple-mpc/robot-handler.hpp"

#include <ndcurves/bezier_curve.h>
#include <stdexcept>

namespace simple_mpc
{
  using curve_translation = ndcurves::bezier_curve<float, double, false, point3_t>;

  FootPlanner::FootPlanner(
    const std::map<std::string, point3_t> & starting_poses,
    double swing_apex,
    int T_fly,
    int T_contact,
    size_t T,
    double timestep)
  : swing_apex_(swing_apex)
  , T_fly_(T_fly)
  , T_contact_(T_contact)
  , T_(T)
  , timestep_(timestep)
  {
    for (auto const & pose : starting_poses)
    {
      references_[pose.first] = std::vector<point3_t>(T, pose.second);
      initial_poses_[pose.first] = pose.second;
      final_poses_[pose.first] = pose.second;
      swing_trajectories_[pose.first] = defineTranslationBezier(pose.second, pose.second);
      previous_in_contact_[pose.first] = true;
    }

    std::map<std::string, bool> initial_contact_states;
    for (auto const & pose : starting_poses)
    {
      initial_contact_states[pose.first] = true;
    }
    reset(initial_contact_states);
  }

  void FootPlanner::reset(const std::map<std::string, bool> & initial_contact_states)
  {
    foot_land_positions_.clear();
    previous_contact_states_.clear();
    for (auto const & contact : initial_contact_states)
    {
      foot_land_positions_[contact.first] = std::vector<point3_t>();
      previous_contact_states_[contact.first] = contact.second;
      previous_in_contact_[contact.first] = contact.second;
    }
  }

  point3_t FootPlanner::computeFootLandingPose(
    std::size_t foot_nb,
    const RobotDataHandler & data_handler,
    const Vector6d & velocity_base) const
  {
    Eigen::Vector2d twist_vect;
    Eigen::Vector3d next_pose;
    twist_vect[0] =
      -(data_handler.getFootRefPose(foot_nb).translation()[1] - data_handler.getBaseFramePose().translation()[1]);
    twist_vect[1] =
      data_handler.getFootRefPose(foot_nb).translation()[0] - data_handler.getBaseFramePose().translation()[0];
    next_pose.head<2>() = data_handler.getFootRefPose(foot_nb).translation().head<2>();
    next_pose.head<2>() +=
      (velocity_base.head<2>() + velocity_base[5] * twist_vect) * (T_fly_ + T_contact_) * timestep_;
    next_pose[2] = data_handler.getFootPose(foot_nb).translation()[2];

    return next_pose;
  }

  void FootPlanner::extractPhaseTimings(
    const std::vector<std::vector<bool>> & horizon_contact_states,
    std::size_t foot_nb,
    bool in_contact,
    std::vector<int> & takeoff_times,
    std::vector<int> & land_times) const
  {
    takeoff_times.clear();
    land_times.clear();

    bool previous_contact = in_contact;
    for (unsigned long time = 1; time < horizon_contact_states.size(); time++)
    {
      const bool contact = horizon_contact_states[time].at(foot_nb);
      if (!contact && previous_contact)
      {
        takeoff_times.push_back(static_cast<int>(time));
      }
      if (contact && !previous_contact)
      {
        land_times.push_back(static_cast<int>(time));
      }
      previous_contact = contact;
    }
  }

  void FootPlanner::appendPreviewLandTimes(
    bool in_contact,
    const std::vector<int> & future_land_times,
    const std::vector<int> & takeoff_times,
    std::vector<int> & land_times) const
  {
    const std::size_t required_land_count = takeoff_times.size() + (in_contact ? 0ul : 1ul);
    for (std::size_t i = 0; i < future_land_times.size() && land_times.size() < required_land_count; i++)
    {
      const int future_land_time = future_land_times[i];
      if (future_land_time > 0 && (land_times.empty() || future_land_time > land_times.back()))
      {
        land_times.push_back(future_land_time);
      }
    }
  }

  void FootPlanner::updateCommittedLandPositions(
    const std::string & ee_name,
    std::size_t foot_nb,
    bool in_contact,
    const std::vector<int> & land_times,
    const RobotDataHandler & data_handler,
    const Vector6d & velocity_base)
  {
    auto & committed_land_positions = foot_land_positions_.at(ee_name);
    if (in_contact && !previous_contact_states_.at(ee_name) && !committed_land_positions.empty())
    {
      committed_land_positions.erase(committed_land_positions.begin());
    }
    if (committed_land_positions.size() > land_times.size())
    {
      committed_land_positions.resize(land_times.size());
    }

    bool should_commit_next = committed_land_positions.size() < land_times.size();
    while (should_commit_next)
    {
      const std::size_t land_id = committed_land_positions.size();
      should_commit_next = (!in_contact && land_id == 0)
                           || (land_times[land_id] <= T_fly_ + T_contact_);
      if (should_commit_next)
      {
        committed_land_positions.push_back(computeFootLandingPose(foot_nb, data_handler, velocity_base));
        should_commit_next = committed_land_positions.size() < land_times.size();
      }
    }
  }

  void FootPlanner::updateSwingTrajectory(
    const std::string & ee_name,
    std::size_t foot_nb,
    bool in_contact,
    const std::vector<point3_t> & land_positions,
    const RobotDataHandler & data_handler)
  {
    const point3_t current_pose = data_handler.getFootPose(foot_nb).translation();
    if (!land_positions.empty() && (in_contact || previous_in_contact_.at(ee_name)))
    {
      initial_poses_[ee_name] = current_pose;
      final_poses_[ee_name] = land_positions.front();
      swing_trajectories_[ee_name] = defineTranslationBezier(initial_poses_.at(ee_name), final_poses_.at(ee_name));
      return;
    }

    if (in_contact)
    {
      initial_poses_[ee_name] = current_pose;
      final_poses_[ee_name] = current_pose;
      swing_trajectories_[ee_name] = defineTranslationBezier(initial_poses_.at(ee_name), final_poses_.at(ee_name));
    }
  }

  piecewise_curve FootPlanner::defineTranslationBezier(
    const point3_t & trans_init,
    const point3_t & trans_final) const
  {
    std::vector<Eigen::Vector3d> points;
    for (long i = 0; i < 4; i++)
    {
      points.push_back(trans_init);
    }
    Eigen::Vector3d midpoint = trans_init * 3 / 4 + trans_final * 1 / 4;
    midpoint[2] += swing_apex_;
    points.push_back(midpoint);
    for (long i = 5; i < 9; i++)
    {
      points.push_back(trans_final);
    }
    curve_translation bezier_curve(curve_translation(points.begin(), points.end(), 0., 1.));

    piecewise_curve curve;
    curve.add_curve(bezier_curve);
    return curve;
  }

  std::vector<point3_t> FootPlanner::createTrajectory(
    const std::string & ee_name,
    const point3_t & current_trans,
    bool in_contact,
    const std::vector<int> & takeoff_times,
    const std::vector<int> & land_times,
    const std::vector<point3_t> & land_poses) const
  {
    if (land_times.size() != land_poses.size())
    {
      throw std::runtime_error("land_times size does not match land_poses size");
    }

    std::vector<point3_t> trajectory(T_, current_trans);
    point3_t stance_pose = current_trans;
    std::size_t takeoff_id = 0;
    std::size_t land_id = 0;
    int swing_start_time = 0;
    piecewise_curve swing_trajectory = defineTranslationBezier(current_trans, current_trans);

    if (!in_contact && !land_poses.empty())
    {
      swing_start_time = land_times.front() - T_fly_;
      swing_trajectory = swing_trajectories_.at(ee_name);
    }

    for (size_t t = 0; t < T_; t++)
    {
      const int time = static_cast<int>(t);
      point3_t sample = stance_pose;

      while (land_id < land_times.size() && land_times[land_id] < time)
      {
        stance_pose = land_poses[land_id];
        in_contact = true;
        land_id += 1;
      }

      if (in_contact)
      {
        const bool takeoff_now = takeoff_id < takeoff_times.size() && takeoff_times[takeoff_id] <= time;
        if (takeoff_now)
        {
          swing_start_time = takeoff_times[takeoff_id];
          in_contact = false;
          const point3_t & target_pose = land_id < land_poses.size() ? land_poses[land_id] : stance_pose;
          swing_trajectory = defineTranslationBezier(stance_pose, target_pose);
          takeoff_id += 1;
        }
      }

      if (in_contact || land_id >= land_times.size())
      {
        sample = stance_pose;
      }
      else
      {
        const int landing_time = land_times[land_id];
        const point3_t & landing_pose = land_poses[land_id];
        if (time >= landing_time)
        {
          stance_pose = landing_pose;
          in_contact = true;
          land_id += 1;
          sample = stance_pose;
        }
        else
        {
          const int swing_duration = landing_time - swing_start_time;
          if (swing_duration <= 0)
          {
            sample = landing_pose;
          }
          else
          {
            double u = static_cast<double>(time - swing_start_time) / static_cast<double>(swing_duration);
            if (u < 0.)
              u = 0.;
            if (u > 1.)
              u = 1.;
            sample = swing_trajectory(static_cast<float>(u));
          }
        }
      }

      trajectory[t] = sample;
    }

    return trajectory;
  }

  void FootPlanner::updateFootReference(
    const std::string & ee_name,
    std::size_t foot_nb,
    const std::vector<std::vector<bool>> & horizon_contact_states,
    const std::vector<int> & future_land_times,
    const RobotDataHandler & data_handler,
    const Vector6d & velocity_base)
  {
    const bool in_contact = horizon_contact_states[0].at(foot_nb);
    std::vector<int> takeoff_times;
    std::vector<int> land_times;

    extractPhaseTimings(horizon_contact_states, foot_nb, in_contact, takeoff_times, land_times);
    appendPreviewLandTimes(in_contact, future_land_times, takeoff_times, land_times);
    updateCommittedLandPositions(ee_name, foot_nb, in_contact, land_times, data_handler, velocity_base);
    std::vector<point3_t> land_positions = foot_land_positions_.at(ee_name);
    for (std::size_t land_id = land_positions.size(); land_id < land_times.size(); land_id++)
    {
      land_positions.push_back(computeFootLandingPose(foot_nb, data_handler, velocity_base));
    }
    updateSwingTrajectory(ee_name, foot_nb, in_contact, land_positions, data_handler);

    references_[ee_name] = createTrajectory(
      ee_name,
      data_handler.getFootPose(foot_nb).translation(),
      in_contact,
      takeoff_times,
      land_times,
      land_positions);

    previous_contact_states_[ee_name] = in_contact;
    previous_in_contact_[ee_name] = in_contact;
  }

} // namespace simple_mpc
