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

      while (land_id < land_times.size() && land_times[land_id] < time)
      {
        stance_pose = land_poses[land_id];
        in_contact = true;
        land_id += 1;
      }

      if (in_contact)
      {
        if (takeoff_id < takeoff_times.size() && takeoff_times[takeoff_id] <= time)
        {
          swing_start_time = takeoff_times[takeoff_id];
          in_contact = false;
          const point3_t & target_pose = land_id < land_poses.size() ? land_poses[land_id] : stance_pose;
          swing_trajectory = defineTranslationBezier(stance_pose, target_pose);
          takeoff_id += 1;
        }
        else
        {
          trajectory[t] = stance_pose;
          continue;
        }
      }

      if (land_id >= land_times.size())
      {
        trajectory[t] = stance_pose;
        continue;
      }

      const int landing_time = land_times[land_id];
      const point3_t & landing_pose = land_poses[land_id];
      if (time >= landing_time)
      {
        stance_pose = landing_pose;
        trajectory[t] = stance_pose;
        in_contact = true;
        land_id += 1;
        continue;
      }

      const int swing_duration = landing_time - swing_start_time;
      if (swing_duration <= 0)
      {
        trajectory[t] = landing_pose;
        continue;
      }

      double u = static_cast<double>(time - swing_start_time) / static_cast<double>(swing_duration);
      if (u < 0.)
        u = 0.;
      if (u > 1.)
        u = 1.;
      trajectory[t] = swing_trajectory(static_cast<float>(u));
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

    if (!land_positions.empty() && (in_contact || previous_in_contact_.at(ee_name)))
    {
      initial_poses_[ee_name] = data_handler.getFootPose(foot_nb).translation();
      final_poses_[ee_name] = land_positions.front();
      swing_trajectories_[ee_name] = defineTranslationBezier(initial_poses_.at(ee_name), final_poses_.at(ee_name));
    }
    else if (in_contact)
    {
      initial_poses_[ee_name] = data_handler.getFootPose(foot_nb).translation();
      final_poses_[ee_name] = data_handler.getFootPose(foot_nb).translation();
      swing_trajectories_[ee_name] = defineTranslationBezier(initial_poses_.at(ee_name), final_poses_.at(ee_name));
    }

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
