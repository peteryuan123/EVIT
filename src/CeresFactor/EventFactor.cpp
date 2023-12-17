//
// Created by mpl on 23-11-12.
//

#include <utility>

#include "CeresFactor/EventFactor.h"

using namespace CannyEVIT;

EventFactor::EventFactor(const CannyEVIT::Point &p_w, const TimeSurface::Ptr &time_surface, int wx, int wy,
                         TimeSurface::PolarType polar, double weight)
    : polar_(polar), p_w_(p_w), time_surface_(time_surface), wx_(wx), wy_(wy), weight_(weight) {
  set_num_residuals(wx * wy);
  *mutable_parameter_block_sizes() = std::vector<int32_t>{7};
}

bool EventFactor::Evaluate(const double *const *parameters, double *residuals, double **jacobians) const {
  Eigen::Vector3d Pi(parameters[0][0], parameters[0][1], parameters[0][2]);
  Eigen::Quaterniond Qi(parameters[0][6], parameters[0][3], parameters[0][4], parameters[0][5]);

  Eigen::Map<Eigen::MatrixXd> residual(residuals, num_residuals(), 1);
  residual = weight_ * time_surface_->evaluate(p_w_, Qi, Pi, wx_, wy_, polar_);
  if (jacobians && jacobians[0]) {
    Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> jacobian_pose_i(
        jacobians[0], num_residuals(), 7);
    jacobian_pose_i.setZero();
    jacobian_pose_i.block(0, 0, num_residuals(), 6) = weight_ * time_surface_->df(p_w_, Qi, Pi, wx_, wy_, polar_);
  }
  return true;
}