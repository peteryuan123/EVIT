//
// Created by mpl on 23-11-7.
//

#ifndef EVIT_NEW_TIMESURFACE_H
#define EVIT_NEW_TIMESURFACE_H

#include <ceres/cubic_interpolation.h>
#include <ceres/rotation.h>
#include <Eigen/Geometry>
#include <opencv2/opencv.hpp>
#include <unordered_map>

#include "EventCamera.h"
#include "Type.h"

namespace CannyEVIT {

struct OptField {
  Eigen::MatrixXd field_;
  Eigen::MatrixXd gradX_;
  Eigen::MatrixXd gradY_;
};

class TimeSurface {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  typedef std::shared_ptr<TimeSurface> Ptr;
  typedef std::shared_ptr<TimeSurface const> ConstPtr;

  enum PolarType { NEUTRAL, POSITIVE, NEGATIVE };
  enum FieldType { INV_TIME_SURFACE, DISTANCE_FIELD };
  enum VisualizationType { TIME_SURFACE, CANNY };

  TimeSurface(double time_stamp, double decay_fator, double truncate_threshold);

  void processTimeSurface(const cv::Mat &history_event,
                          double time_stamp,
                          double decay_factor,
                          cv::Mat &time_surface,
                          OptField &inv_time_surface_field);

  void constructDistanceField(const cv::Mat &time_surface,
                              OptField &distance_field,
                              cv::Mat &visualization_img);

  void drawCloud(pCloud cloud,
                 const Eigen::Matrix4d &Twc,
                 const std::string &window_name,
                 PolarType polarType = NEUTRAL,
                 VisualizationType visType = TIME_SURFACE,
                 bool showGrad = false,
                 const std::set<size_t> &indices = std::set<size_t>(),
                 const Eigen::Matrix4d &T_predict = Eigen::Matrix4d::Zero());

  // for compute residual
  bool isValidPatch(Eigen::Vector2d &patchCentreCoord, Eigen::MatrixXi &mask, size_t wx, size_t wy);

  bool patchInterpolation(const Eigen::MatrixXd &img,
                          const Eigen::Vector2d &location,
                          int wx,
                          int wy,
                          Eigen::MatrixXd &patch,
                          bool debug);

  Eigen::VectorXd evaluate(const Point &p_w,
                           const Eigen::Quaterniond &Qwb,
                           const Eigen::Vector3d &twb,
                           int wx,
                           int wy,
                           PolarType polarType = NEUTRAL,
                           FieldType fieldType = INV_TIME_SURFACE);

  Eigen::MatrixXd df(const Point &p_w,
                     const Eigen::Quaterniond &Qwb,
                     const Eigen::Vector3d &twb,
                     int wx,
                     int wy,
                     PolarType polarType = NEUTRAL,
                     FieldType fieldType = INV_TIME_SURFACE);

 public:
  double time_stamp_;
  double decay_factor_;
  double truncate_threshold_;

  std::unordered_map<VisualizationType, cv::Mat> neutral_visualization_fields_;
  std::unordered_map<VisualizationType, cv::Mat> positive_visualization_fields_;
  std::unordered_map<VisualizationType, cv::Mat> negative_visualization_fields_;

  std::unordered_map<FieldType, OptField> neutral_fields_;
  std::unordered_map<FieldType, OptField> positive_fields_;
  std::unordered_map<FieldType, OptField> negative_fields_;

 public:

//  static std::tuple<TimeSurface::PolarType, double> determinePolarAndWeight(const Point &p_w,
//                                                                            const Eigen::Matrix4d &T_last,
//                                                                            const Eigen::Matrix4d &T_current);

  static std::pair<TimeSurface::PolarType, double> determinePolarAndWeight(const Point &p_w,
                                                                           const Eigen::Matrix4d &T_last,
                                                                           const Eigen::Matrix4d &T_current);
  static void initTimeSurface(const EventCamera::Ptr &event_cam);
  static void updateHistoryEvent(const EventMsg &msg);
  static EventCamera::Ptr event_cam_;
  static cv::Mat history_event_;
  static cv::Mat history_positive_event_;
  static cv::Mat history_negative_event_;
};

}  // namespace CannyEVIT

#endif  // EVIT_NEW_TIMESURFACE_H
