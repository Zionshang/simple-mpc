///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2024, INRIA
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////
#pragma once

#include "simple-mpc/foot-trajectory.hpp"

namespace simple_mpc
{
  class FootPlanner
  {
  protected:
    FootTrajectory foot_trajectories_;
    std::map<std::string, std::vector<point3_t>> foot_land_positions_;
    std::map<std::string, bool> previous_contact_states_;
    int T_fly_;
    int T_contact_;
    double timestep_;

    point3_t computeFootLandingPose(
      std::size_t foot_nb,
      const RobotDataHandler & data_handler,
      const Vector6d & velocity_base) const;

  public:
    explicit FootPlanner() {};
    FootPlanner(
      const std::map<std::string, point3_t> & starting_poses,
      double swing_apex,
      int T_fly,
      int T_contact,
      size_t T,
      double timestep);

    void reset(const std::map<std::string, bool> & initial_contact_states);

    void updateFootReference(
      const std::string & ee_name,
      std::size_t foot_nb,
      const std::vector<std::vector<bool>> & horizon_contact_states,
      const std::vector<int> & future_land_times,
      const RobotDataHandler & data_handler,
      const Vector6d & velocity_base);

    const std::vector<point3_t> & getReference(const std::string & ee_name) const
    {
      return foot_trajectories_.getReference(ee_name);
    }
  };

} // namespace simple_mpc
