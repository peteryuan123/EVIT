//
// Created by mpl on 23-11-7.
//

#ifndef EVIT_NEW_TIMESURFACE_H
#define EVIT_NEW_TIMESURFACE_H

#include <ceres/cubic_interpolation.h>
#include <ceres/rotation.h>

#include <Eigen/Geometry>
#include <opencv2/opencv.hpp>

#include "EventCamera.h"
#include "Type.h"

namespace CannyEVIT {

class TimeSurface {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  typedef std::shared_ptr<TimeSurface> Ptr;
  typedef std::shared_ptr<TimeSurface const> ConstPtr;

  enum PolarType { NEUTRAL, POSITIVE, NEGATIVE };

  TimeSurface(double time_stamp, double decay_fator);

  void processTimeSurface(const cv::Mat& history_event, double time_stamp, double decay_factor, cv::Mat& time_surface,
                          Eigen::MatrixXd& inverse_time_surface, Eigen::MatrixXd& inverse_gradX,
                          Eigen::MatrixXd& inverse_gradY);
  void drawCloud(pCloud cloud, const Eigen::Matrix4d& Twc, const std::string& window_name,
                 PolarType polarType = NEUTRAL, bool showGrad = false);

  // for compute residual
  bool isValidPatch(Eigen::Vector2d& patchCentreCoord, Eigen::MatrixXi& mask, size_t wx, size_t wy);
  bool patchInterpolation(const Eigen::MatrixXd& img, const Eigen::Vector2d& location, int wx, int wy,
                          Eigen::MatrixXd& patch, bool debug);

  Eigen::VectorXd evaluate(const Point& p_w, const Eigen::Quaterniond& Qwb, const Eigen::Vector3d& twb, int wx, int wy,
                           PolarType polarType = NEUTRAL);
  Eigen::MatrixXd df(const Point& p_w, const Eigen::Quaterniond& Qwb, const Eigen::Vector3d& twb, int wx, int wy,
                     PolarType polarType = NEUTRAL);

 public:
  double time_stamp_;
  double decay_factor_;

  cv::Mat time_surface_;
  cv::Mat time_surface_positive_;
  cv::Mat time_surface_negative_;

  Eigen::MatrixXd inverse_time_surface_;
  Eigen::MatrixXd inverse_time_surface_positive_;
  Eigen::MatrixXd inverse_time_surface_negative_;

  Eigen::MatrixXd gradX_inverse_time_surface_;
  Eigen::MatrixXd gradY_inverse_time_surface_;
  Eigen::MatrixXd gradX_inverse_time_surface_positive_;
  Eigen::MatrixXd gradY_inverse_time_surface_positive_;
  Eigen::MatrixXd gradX_inverse_time_surface_negative_;
  Eigen::MatrixXd gradY_inverse_time_surface_negative_;

 public:
  static std::tuple<TimeSurface::PolarType, double> determinePolarAndWeight(const Point& p_w,
                                                                            const Eigen::Matrix4d& T_last,
                                                                            const Eigen::Matrix4d& T_current);
  static void initTimeSurface(const EventCamera::Ptr& event_cam);
  static void updateHistoryEvent(EventMsg msg);
  static EventCamera::Ptr event_cam_;
  static cv::Mat history_event_;
  static cv::Mat history_positive_event_;
  static cv::Mat history_negative_event_;
};

}  // namespace CannyEVIT

#endif  // EVIT_NEW_TIMESURFACE_H
