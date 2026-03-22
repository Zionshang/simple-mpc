///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2024, INRIA
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////
#pragma once

#include "simple-mpc/fwd.hpp"
#include <ndcurves/fwd.h>
#include <ndcurves/piecewise_curve.h>

namespace simple_mpc
{
  using point3_t = Eigen::Vector3d;
  using piecewise_curve = ndcurves::piecewise_curve<float, double, false, point3_t>;

  class FootPlanner
  {
  protected:
    std::map<std::string, point3_t> initial_poses_;
    std::map<std::string, point3_t> final_poses_;
    std::map<std::string, piecewise_curve> swing_trajectories_;
    std::map<std::string, bool> previous_in_contact_;
    std::map<std::string, std::vector<point3_t>> references_;
    std::map<std::string, std::vector<point3_t>> foot_land_positions_;
    std::map<std::string, bool> previous_contact_states_;
    double swing_apex_;
    int T_fly_;
    int T_contact_;
    size_t T_;
    double timestep_;

    point3_t computeFootLandingPose(
      std::size_t foot_nb,
      const RobotDataHandler & data_handler,
      const Vector6d & velocity_base) const;
    piecewise_curve defineTranslationBezier(const point3_t & trans_init, const point3_t & trans_final) const;
    std::vector<point3_t> createTrajectory(
      const std::string & ee_name,
      const point3_t & current_trans,
      bool in_contact,
      const std::vector<int> & takeoff_times,
      const std::vector<int> & land_times,
      const std::vector<point3_t> & land_poses) const;

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
      return references_.at(ee_name);
    }
  };

} // namespace simple_mpc
