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
  /**
   * @brief Foot trajectory generation over the whole MPC horizon.
   */

  using point3_t = Eigen::Vector3d;
  using piecewise_curve = ndcurves::piecewise_curve<float, double, false, point3_t>;

  class FootTrajectory
  {
  protected:
    std::map<std::string, point3_t> initial_poses_;
    std::map<std::string, point3_t> final_poses_;
    std::map<std::string, piecewise_curve> swing_trajectories_;
    std::map<std::string, bool> previous_in_contact_;
    std::map<std::string, std::vector<point3_t>> references_;
    double swing_apex_;
    int T_fly_;
    int T_contact_;
    size_t T_;

  public:
    explicit FootTrajectory() {};
    FootTrajectory(
      const std::map<std::string, point3_t> & initial_poses, double swing_apex, int T_fly, int T_contact, size_t T);
    virtual ~FootTrajectory() {};

    void updateApex(double swing_apex)
    {
      swing_apex_ = swing_apex;
    }

    piecewise_curve defineTranslationBezier(const point3_t & trans_init, const point3_t & trans_final);

    std::vector<point3_t> createTrajectory(
      const std::string & ee_name,
      const point3_t & current_trans,
      bool in_contact,
      const std::vector<int> & takeoff_times,
      const std::vector<int> & land_times,
      const std::vector<point3_t> & land_poses);

    void updateTrajectory(
      const point3_t & ee_trans,
      bool in_contact,
      const std::vector<int> & takeoff_times,
      const std::vector<int> & land_times,
      const std::vector<point3_t> & land_poses,
      const std::string & ee_name);

    const std::vector<point3_t> & getReference(const std::string & ee_name) const
    {
      return references_.at(ee_name);
    }
  };

} // namespace simple_mpc
