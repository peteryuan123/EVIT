//
// Created by mpl on 23-11-16.
//

#ifndef CANNYEVIT_EVENTAUTODIFFFACTOR_H
#define CANNYEVIT_EVENTAUTODIFFFACTOR_H

#include <ceres/ceres.h>
#include <ceres/cubic_interpolation.h>

#include "TimeSurface.h"
#include "Type.h"

// THIS FACTOR IS NOT WORKDING !!!!!
namespace CannyEVIT {
class EventAutoDiffFactor {
 public:
  EventAutoDiffFactor() = delete;
  EventAutoDiffFactor(const Point& p_w, const TimeSurface::Ptr& time_surface, int wx, int wy)
      : p_w_(p_w), time_surface_(time_surface), wx_(wx), wy_(wy) {
    int rows = time_surface_->inverse_time_surface_.rows();
    int cols = time_surface_->inverse_time_surface_.cols();
    grid_.reset(new ceres::Grid2D<double, 1, false>(time_surface_->inverse_time_surface_.data(), 0, rows, 0, cols));
    interpolator_.reset(new ceres::BiCubicInterpolator<ceres::Grid2D<double, 1, false>>(*grid_));
  };

  template <typename T>
  bool operator()(const T* parameters, T* residual) const {
    Eigen::Matrix<T, 3, 1> twb(parameters[0], parameters[1], parameters[2]);
    T Qbw_array[4] = {parameters[6], -parameters[3], -parameters[4], -parameters[5]};

    Eigen::Matrix<T, 3, 1> p_w;
    p_w << T(p_w_.x), T(p_w_.y), T(p_w_.z);

    Eigen::Matrix<T, 3, 1> p_temp = p_w - twb;
    Eigen::Matrix<T, 3, 1> p_body;
    ceres::QuaternionRotatePoint(Qbw_array, p_temp.data(), p_body.data());

    //            Eigen::Matrix<T, 3, 3> Rbw = Qwb.conjugate().toRotationMatrix();
    Eigen::Matrix<T, 3, 3> Rcb = TimeSurface::event_cam_->Rcb().template cast<T>();
    Eigen::Matrix<T, 3, 1> tcb = TimeSurface::event_cam_->tcb().template cast<T>();

    Eigen::Matrix<T, 3, 1> p_cam = Rcb * p_body + tcb;

    Eigen::Matrix<T, 3, 4> projection_matrix = TimeSurface::event_cam_->getProjectionMatrix().template cast<T>();
    Eigen::Matrix<T, 3, 1> uv_homo =
        projection_matrix.template block<3, 3>(0, 0) * p_cam + projection_matrix.template block<3, 1>(0, 3);

    T u = uv_homo(0) / uv_homo(2);
    T v = uv_homo(1) / uv_homo(2);

    T val_at_uv;
    interpolator_->Evaluate(u, v, &val_at_uv);
    residual[0] = val_at_uv;

    return true;
  }

  inline static ceres::AutoDiffCostFunction<EventAutoDiffFactor, 1, 7>* create(const Point& p_w,
                                                                               const TimeSurface::Ptr& time_surface,
                                                                               int wx, int wy) {
    return new ceres::AutoDiffCostFunction<EventAutoDiffFactor, 1, 7>(
        new EventAutoDiffFactor(p_w, time_surface, wx, wy));
  }

  Point p_w_;
  TimeSurface::Ptr time_surface_;
  int wx_, wy_;
  std::unique_ptr<ceres::Grid2D<double, 1, false>> grid_;
  std::unique_ptr<ceres::BiCubicInterpolator<ceres::Grid2D<double, 1, false>>> interpolator_;
};

}  // namespace CannyEVIT

#endif  // CANNYEVIT_EVENTAUTODIFFFACTOR_H
