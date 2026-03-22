///////////////////////////////////////////////////////////////////////////////
// BSD 2-Clause License
//
// Copyright (C) 2024, INRIA
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "simple-mpc/foot-trajectory.hpp"
#include <stdexcept>
#include <ndcurves/bezier_curve.h>
#include <ndcurves/exact_cubic.h>
#include <ndcurves/se3_curve.h>
#include <ndcurves/so3_linear.h>
#include <pinocchio/spatial/se3.hpp>

namespace simple_mpc
{
  using curve_translation = ndcurves::bezier_curve<float, double, false, point3_t>;

  FootTrajectory::FootTrajectory(
    const std::map<std::string, point3_t> & initial_poses, double swing_apex, int T_fly, int T_contact, size_t T)
  {
    swing_apex_ = swing_apex;
    T_fly_ = T_fly;
    T_contact_ = T_contact;
    T_ = T;

    for (auto it = initial_poses.begin(); it != initial_poses.end(); it++)
    {
      references_.insert({it->first, std::vector<point3_t>(T, it->second)});
      initial_poses_.insert({it->first, it->second});
      final_poses_.insert({it->first, it->second});
      swing_trajectories_.insert({it->first, defineTranslationBezier(it->second, it->second)});
      previous_in_contact_.insert({it->first, true});
    }
  }

  piecewise_curve FootTrajectory::defineTranslationBezier(const point3_t & trans_init, const point3_t & trans_final)
  {
    std::vector<Eigen::Vector3d> points;
    for (long i = 0; i < 4; i++)
    { // init position. init vel,acc and jerk == 0
      points.push_back(trans_init);
    }
    // compute mid point (average and offset along z)
    Eigen::Vector3d midpoint = trans_init * 3 / 4 + trans_final * 1 / 4;
    midpoint[2] += swing_apex_;
    points.push_back(midpoint);
    for (long i = 5; i < 9; i++)
    { // final position. final vel,acc and jerk == 0
      points.push_back(trans_final);
    }
    curve_translation bezier_curve(curve_translation(points.begin(), points.end(), 0., 1.));

    piecewise_curve se3curve;
    se3curve.add_curve(bezier_curve);

    return se3curve;
  }

  std::vector<point3_t> FootTrajectory::createTrajectory(
    const std::string & ee_name,
    const point3_t & current_trans,
    bool in_contact,
    const std::vector<int> & takeoff_times,
    const std::vector<int> & land_times,
    const std::vector<point3_t> & land_poses)
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

  void FootTrajectory::updateTrajectory(
    const point3_t & ee_trans,
    bool in_contact,
    const std::vector<int> & takeoff_times,
    const std::vector<int> & land_times,
    const std::vector<point3_t> & land_poses,
    const std::string & ee_name)
  {
    if (!land_poses.empty() && (in_contact || previous_in_contact_.at(ee_name)))
    {
      initial_poses_.at(ee_name) = ee_trans;
      final_poses_.at(ee_name) = land_poses.front();
      swing_trajectories_.at(ee_name) = defineTranslationBezier(initial_poses_.at(ee_name), final_poses_.at(ee_name));
    }
    else if (in_contact)
    {
      initial_poses_.at(ee_name) = ee_trans;
      final_poses_.at(ee_name) = ee_trans;
      swing_trajectories_.at(ee_name) = defineTranslationBezier(ee_trans, ee_trans);
    }

    references_.at(ee_name) = createTrajectory(ee_name, ee_trans, in_contact, takeoff_times, land_times, land_poses);
    previous_in_contact_.at(ee_name) = in_contact;
  }

} // namespace simple_mpc
