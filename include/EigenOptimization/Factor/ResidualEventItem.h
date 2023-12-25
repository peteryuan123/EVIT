//
// Created by mpl on 23-11-20.
//

#ifndef CANNYEVIT_RESIDUALEVENTITEM_H
#define CANNYEVIT_RESIDUALEVENTITEM_H

#include <Eigen/Eigen>
#include <memory>
#include <utility>

#include "Type.h"
#include "TimeSurface.h"

namespace CannyEVIT {

struct ResidualEventItem {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef std::shared_ptr<ResidualEventItem> Ptr;

  ResidualEventItem(const Point &point,
                    TimeSurface::Ptr time_surface,
                    size_t patch_size_x,
                    size_t patch_size_y,
                    TimeSurface::PolarType polar_type,
                    TimeSurface::FieldType field_type)
      : p_(point),
        time_surface_(std::move(time_surface)),
        patch_size_x_(patch_size_x),
        patch_size_y_(patch_size_y),
        residuals_(patch_size_x * patch_size_y),
        irls_weight_(patch_size_x * patch_size_y),
        polar_type_(polar_type),
        field_type_(field_type){};

  void computeResidual(const Eigen::Quaterniond &Qwb, const Eigen::Vector3d &twb);
  void computeJacobian(const Eigen::Quaterniond &Qwb, const Eigen::Vector3d &twb);
  Point p_;  // 3D coordinate in world frame
  TimeSurface::Ptr time_surface_;
  size_t patch_size_x_, patch_size_y_;
  Eigen::VectorXd residuals_;
  Eigen::MatrixXd jacobian_;
  Eigen::VectorXd irls_weight_;
  TimeSurface::PolarType polar_type_;
  TimeSurface::FieldType field_type_;
};

using ResidualEventItems = std::vector<ResidualEventItem, Eigen::aligned_allocator<ResidualEventItem>>;

}  // namespace CannyEVIT

#endif  // CANNYEVIT_RESIDUALEVENTITEM_H
