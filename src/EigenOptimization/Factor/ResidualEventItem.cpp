//
// Created by mpl on 23-12-22.
//
#include "EigenOptimization/Factor/ResidualEventItem.h"

using namespace CannyEVIT;

void ResidualEventItem::computeResidual(const Eigen::Quaterniond &Qwb, const Eigen::Vector3d &twb) {
  residuals_ = time_surface_->evaluate(p_, Qwb, twb, patch_size_x_, patch_size_y_, polar_type_, field_type_);
}

void ResidualEventItem::computeJacobian(const Eigen::Quaterniond &Qwb, const Eigen::Vector3d &twb) {
  jacobian_ = time_surface_->df(p_, Qwb, twb, patch_size_x_, patch_size_y_, polar_type_, field_type_);
}