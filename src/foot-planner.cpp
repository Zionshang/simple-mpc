///////////////////////////////////////////////////////////////////////////////
// BSD 2-Clause License
//
// Copyright (C) 2024, INRIA
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "simple-mpc/foot-planner.hpp"
#include "simple-mpc/robot-handler.hpp"

namespace simple_mpc
{
  FootPlanner::FootPlanner(
    const std::map<std::string, point3_t> & starting_poses,
    double swing_apex,
    int T_fly,
    int T_contact,
    size_t T,
    double timestep)
  : foot_trajectories_(starting_poses, swing_apex, T_fly, T_contact, T)
  , T_fly_(T_fly)
  , T_contact_(T_contact)
  , timestep_(timestep)
  {
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
    std::vector<point3_t> land_positions;

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

    const std::size_t required_land_count = takeoff_times.size() + (in_contact ? 0ul : 1ul);
    for (const int future_land_time : future_land_times)
    {
      if (land_times.size() >= required_land_count)
      {
        break;
      }
      if (future_land_time > 0 && (land_times.empty() || future_land_time > land_times.back()))
      {
        land_times.push_back(future_land_time);
      }
    }

    auto & committed_land_positions = foot_land_positions_.at(ee_name);
    if (in_contact && !previous_contact_states_.at(ee_name) && !committed_land_positions.empty())
    {
      committed_land_positions.erase(committed_land_positions.begin());
    }
    if (committed_land_positions.size() > land_times.size())
    {
      committed_land_positions.resize(land_times.size());
    }
    while (committed_land_positions.size() < land_times.size())
    {
      const std::size_t land_id = committed_land_positions.size();
      const bool should_commit = (!in_contact && land_id == 0)
                                 || (land_times[land_id] <= T_fly_ + T_contact_);
      if (!should_commit)
      {
        break;
      }
      committed_land_positions.push_back(computeFootLandingPose(foot_nb, data_handler, velocity_base));
    }

    land_positions = committed_land_positions;
    for (std::size_t land_id = land_positions.size(); land_id < land_times.size(); land_id++)
    {
      land_positions.push_back(computeFootLandingPose(foot_nb, data_handler, velocity_base));
    }

    foot_trajectories_.updateTrajectory(
      data_handler.getFootPose(foot_nb).translation(),
      in_contact,
      takeoff_times,
      land_times,
      land_positions,
      ee_name);

    previous_contact_states_.at(ee_name) = in_contact;
  }

} // namespace simple_mpc
