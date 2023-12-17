//
// Created by mpl on 23-11-8.
//

#ifndef CANNYEVIT_EVENTCAMERA_H
#define CANNYEVIT_EVENTCAMERA_H

#include <Eigen/Core>
#include <memory>
#include <opencv2/opencv.hpp>

namespace CannyEVIT {
enum DistortionModel { PLUMB_BOB, EQUIDISTANT };

class EventCamera {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  typedef std::shared_ptr<EventCamera> Ptr;
  typedef std::shared_ptr<EventCamera const> ConstPtr;

 public:
  EventCamera(std::string configPath);
  ~EventCamera() = default;

  void undistortImage(const cv::Mat& src, cv::Mat& dest);

  inline Eigen::Vector2d World2Cam(const Eigen::Vector3d& p) {
    Eigen::Vector3d x_homo = P_.block<3, 3>(0, 0) * p + P_.block<3, 1>(0, 3);
    return x_homo.head(2) / x_homo(2);
  }

 public:
  int width_, height_;

  Eigen::Matrix3d K_;
  Eigen::Matrix3d K_inv_;
  Eigen::Matrix<double, 3, 4> P_;
  cv::Mat distortion_parameter_;
  DistortionModel distortion_type_;
  cv::Mat undistortion_map1_, undistortion_map2_;
  Eigen::MatrixXi undistort_recitify_mask_;

  Eigen::Matrix4d Tbc_;
  Eigen::Matrix4d Tcb_;

 public:
  inline Eigen::Matrix<double, 3, 4> getProjectionMatrix() { return P_; }

  inline Eigen::MatrixXi& getUndistortRectifyMask() { return undistort_recitify_mask_; }

  inline int width() { return width_; }

  inline int height() { return height_; }

  inline Eigen::Matrix3d Rbc() { return Tbc_.block<3, 3>(0, 0); }

  inline Eigen::Vector3d tbc() { return Tbc_.block<3, 1>(0, 3); }

  inline Eigen::Matrix4d Tbc() { return Tbc_; }

  inline Eigen::Matrix3d Rcb() { return Tcb_.block<3, 3>(0, 0); }

  inline Eigen::Vector3d tcb() { return Tcb_.block<3, 1>(0, 3); }

  inline Eigen::Matrix4d Tcb() { return Tcb_; }
};

}  // namespace CannyEVIT

#endif  // CANNYEVIT_EVENTCAMERA_H
