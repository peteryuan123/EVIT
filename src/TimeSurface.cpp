//
// Created by mpl on 23-11-7.
//

#include <Eigen/Core>
#include <Eigen/Dense>
#include <opencv2/core/eigen.hpp>

#include "TimeSurface.h"
#include "imageProcessing/sobel.h"
#include "imageProcessing/distanceField.h"
#include "imageProcessing/canny.h"
#include "Util.h"

using namespace CannyEVIT;

EventCamera::Ptr TimeSurface::event_cam_ = nullptr;
cv::Mat TimeSurface::history_event_ = cv::Mat();
cv::Mat TimeSurface::history_positive_event_ = cv::Mat();
cv::Mat TimeSurface::history_negative_event_ = cv::Mat();

TimeSurface::TimeSurface(double time_stamp, double decay_factor)
    : time_stamp_(time_stamp), decay_factor_(decay_factor) {
  processTimeSurface(history_event_, time_stamp, decay_factor_, time_surface_, inverse_time_surface_,
                     gradX_inverse_time_surface_, gradY_inverse_time_surface_);

  constructDistanceField(time_surface_, distance_field_, gradX_distance_field_, gradY_distance_field_, img_canny_);

  processTimeSurface(history_positive_event_, time_stamp, decay_factor_, time_surface_positive_,
                     inverse_time_surface_positive_, gradX_inverse_time_surface_positive_,
                     gradY_inverse_time_surface_positive_);
  constructDistanceField(time_surface_positive_, distance_field_positive_, gradX_distance_field_positive_, gradY_distance_field_positive_, img_canny_positive_);

  processTimeSurface(history_negative_event_, time_stamp, decay_factor_, time_surface_negative_,
                     inverse_time_surface_negative_, gradX_inverse_time_surface_negative_,
                     gradY_inverse_time_surface_negative_);
  constructDistanceField(time_surface_negative_, distance_field_negative_, gradX_distance_field_negative_, gradY_distance_field_negative_, img_canny_negative_);

}

void TimeSurface::processTimeSurface(const cv::Mat &history_event, double time_stamp, double decay_factor,
                                     cv::Mat &time_surface, Eigen::MatrixXd &inverse_time_surface,
                                     Eigen::MatrixXd &inverse_gradX, Eigen::MatrixXd &inverse_gradY) {
  cv::exp((history_event - time_stamp) / decay_factor, time_surface);
  event_cam_->undistortImage(time_surface, time_surface);
  time_surface = time_surface * 255.0;
  time_surface.convertTo(time_surface, CV_8U);
  cv::GaussianBlur(time_surface, time_surface, cv::Size(5, 5), 0.0);
  cv::Mat inverse_time_surface_cv = 255.0 - time_surface;
  //    inverse_time_surface_cv = inverse_time_surface_cv / 255.0;

  // test
//  cv::cv2eigen(inverse_time_surface_cv, inverse_time_surface);
//  Eigen::ArrayXXd grad_x, grad_y;
//  image_processing::sobel(inverse_time_surface, grad_x, grad_y);
//  inverse_gradX = grad_x;
//  inverse_gradY = grad_y;
  // test

  cv::Mat inverse_gradX_cv, inverse_gradY_cv;
  cv::Sobel(inverse_time_surface_cv, inverse_gradX_cv, CV_64F, 1, 0);
  cv::Sobel(inverse_time_surface_cv, inverse_gradY_cv, CV_64F, 0, 1);

  cv::cv2eigen(inverse_time_surface_cv, inverse_time_surface);
  cv::cv2eigen(inverse_gradX_cv, inverse_gradX);
  cv::cv2eigen(inverse_gradY_cv, inverse_gradY);


}

void TimeSurface::constructDistanceField(const cv::Mat &time_surface,
                                         Eigen::MatrixXd &distance_field,
                                         Eigen::MatrixXd &gradX_distance_field,
                                         Eigen::MatrixXd &gradY_distance_field,
                                         cv::Mat& img) {
  // test distance field
  Eigen::MatrixXd time_surface_eigen;
  cv::cv2eigen(time_surface, time_surface_eigen);
  Eigen::ArrayXXd grad_x, grad_y, grad_mag;
  image_processing::sobel_mag(time_surface_eigen, grad_x, grad_y, grad_mag);
  std::vector<std::pair<int, int>> uv_edge;
  image_processing::canny(grad_mag, grad_x, grad_y, uv_edge, 10);
  Eigen::ArrayXXd distance_field_tmp;
  image_processing::chebychevDistanceField(event_cam_->height(), event_cam_->width(), uv_edge, distance_field_tmp);
  distance_field = distance_field_tmp;
  Eigen::ArrayXXd gradX_distance_field_tmp, gradY_distance_field_tmp;
  image_processing::sobel(distance_field_, gradX_distance_field_tmp, gradY_distance_field_tmp);
  gradX_distance_field = gradX_distance_field_tmp;
  gradY_distance_field = gradY_distance_field_tmp;

  img = cv::Mat(event_cam_->height(), event_cam_->width(), CV_8U, cv::Scalar(0));
  for (auto& pos: uv_edge)
    img.at<uint8_t>(pos.first, pos.second) = 255;
}


void TimeSurface::drawCloud(CannyEVIT::pCloud cloud, const Eigen::Matrix4d &Twc, const std::string &window_name,
                            PolarType polarType, bool showGrad) {
  cv::Mat img_drawing;
  switch (polarType) {
    case NEUTRAL:
      img_drawing = time_surface_.clone();
      break;
    case POSITIVE:
      img_drawing = time_surface_positive_.clone();
      break;
    case NEGATIVE:
      img_drawing = time_surface_negative_.clone();
      break;
    case DISTANCE_FIELD:
      img_drawing = img_canny_.clone();
      break;
  }
  img_drawing.convertTo(img_drawing, CV_8UC1);
  cv::cvtColor(img_drawing, img_drawing, cv::COLOR_GRAY2BGR);

  Eigen::Matrix3d Rwc = Twc.block<3, 3>(0, 0);
  Eigen::Vector3d twc = Twc.block<3, 1>(0, 3);
  for (size_t i = 0; i < cloud->size(); i++) {
    Point pt = cloud->at(i);
    Eigen::Vector3d p(pt.x, pt.y, pt.z);
    Eigen::Vector3d pc = Rwc.transpose() * (p - twc);
    Eigen::Vector2d p_2d = event_cam_->World2Cam(pc);
    cv::Point cvpt(p_2d.x(), p_2d.y());

    if (showGrad) {
      Eigen::Vector3d p_normal(pt.x_gradient_, pt.y_gradient_, pt.z_gradient_);
      Eigen::Vector3d p_normal_c = Rwc.transpose() * p_normal;
      Eigen::Vector3d p_normal_end_c = pc + p_normal_c;
      Eigen::Vector2d p_normal_end_2d = event_cam_->World2Cam(p_normal_end_c);
      Eigen::Vector2d direction = p_normal_end_2d - p_2d;
      direction.normalize();
      direction = direction * 10;
      p_normal_end_2d = p_2d + direction;
      cv::Point cv_normal2d_end(p_normal_end_2d.x(), p_normal_end_2d.y());
      cv::line(img_drawing, cvpt, cv_normal2d_end, CV_RGB(0, 255, 0));
    }
    cv::circle(img_drawing, cvpt, 0, CV_RGB(255, 0, 0), cv::FILLED);
  }
  cv::imshow(window_name, img_drawing);
  cv::imshow(window_name + "_negative", img_canny_negative_);
  cv::imshow(window_name + "_positive", img_canny_positive_);
}

bool TimeSurface::isValidPatch(Eigen::Vector2d &patchCentreCoord, Eigen::MatrixXi &mask, size_t wx, size_t wy) {
  if (patchCentreCoord(0) < (wx - 1) / 2 || patchCentreCoord(0) > event_cam_->width() - (wx - 1) / 2 - 1 ||
      patchCentreCoord(1) < (wy - 1) / 2 || patchCentreCoord(1) > event_cam_->height() - (wy - 1) / 2 - 1)
    return false;
  if (mask(static_cast<int>(patchCentreCoord(1) - (wy - 1) / 2), static_cast<int>(patchCentreCoord(0) - (wx - 1) / 2)) <
      125)
    return false;
  if (mask(static_cast<int>(patchCentreCoord(1) - (wy - 1) / 2), static_cast<int>(patchCentreCoord(0) + (wx - 1) / 2)) <
      125)
    return false;
  if (mask(static_cast<int>(patchCentreCoord(1) + (wy - 1) / 2), static_cast<int>(patchCentreCoord(0) - (wx - 1) / 2)) <
      125)
    return false;
  if (mask(static_cast<int>(patchCentreCoord(1) + (wy - 1) / 2), static_cast<int>(patchCentreCoord(0) + (wx - 1) / 2)) <
      125)
    return false;
  return true;
}

bool TimeSurface::patchInterpolation(const Eigen::MatrixXd &img, const Eigen::Vector2d &location, int wx, int wy,
                                     Eigen::MatrixXd &patch, bool debug) {
  // compute SrcPatch_UpLeft coordinate and SrcPatch_DownRight coordinate
  // check patch bourndary is inside img boundary
  Eigen::Vector2i SrcPatch_UpLeft, SrcPatch_DownRight;
  SrcPatch_UpLeft << floor(location[0]) - (wx - 1) / 2, floor(location[1]) - (wy - 1) / 2;
  SrcPatch_DownRight << floor(location[0]) + (wx - 1) / 2, floor(location[1]) + (wy - 1) / 2;

  if (SrcPatch_UpLeft[0] < 0 || SrcPatch_UpLeft[1] < 0) {
    if (debug) {
      std::cout << "patchInterpolation 1: " << SrcPatch_UpLeft.transpose();
    }
    return false;
  }
  if (SrcPatch_DownRight[0] >= img.cols() || SrcPatch_DownRight[1] >= img.rows()) {
    if (debug) {
      std::cout << "patchInterpolation 2: " << SrcPatch_DownRight.transpose();
    }
    return false;
  }

  // compute q1 q2 q3 q4
  Eigen::Vector2d double_indices;
  double_indices << location[1], location[0];

  std::pair<int, int> lower_indices(floor(double_indices[0]), floor(double_indices[1]));
  std::pair<int, int> upper_indices(lower_indices.first + 1, lower_indices.second + 1);

  double q1 = upper_indices.second - double_indices[1];
  double q2 = double_indices[1] - lower_indices.second;
  double q3 = upper_indices.first - double_indices[0];
  double q4 = double_indices[0] - lower_indices.first;

  // extract Src patch, size (wy+1) * (wx+1)
  int wx2 = wx + 1;
  int wy2 = wy + 1;
  if (SrcPatch_UpLeft[1] + wy >= img.rows() || SrcPatch_UpLeft[0] + wx >= img.cols()) {
    if (debug) {
      std::cout << "patchInterpolation 3: " << SrcPatch_UpLeft.transpose() << ", location: " << location.transpose()
                << ", floor(location[0]): " << floor(location[0]) << ", (wx - 1) / 2: " << (wx - 1) / 2
                << ", ans: " << floor(location[0]) - (wx - 1) / 2 << ", wx: " << wx << " wy: " << wy
                << ", img.row: " << img.rows() << " img.col: " << img.cols();
    }
    return false;
  }
  Eigen::MatrixXd SrcPatch = img.block(SrcPatch_UpLeft[1], SrcPatch_UpLeft[0], wy2, wx2);

  // Compute R, size (wy+1) * wx.
  Eigen::MatrixXd R;
  R = q1 * SrcPatch.block(0, 0, wy2, wx) + q2 * SrcPatch.block(0, 1, wy2, wx);

  // Compute F, size wy * wx.
  patch = q3 * R.block(0, 0, wy, wx) + q4 * R.block(1, 0, wy, wx);
  return true;
}

Eigen::VectorXd TimeSurface::evaluate(const CannyEVIT::Point &p_w, const Eigen::Quaterniond &Qwb,
                                      const Eigen::Vector3d &twb, int wx, int wy, PolarType polarType) {
  Eigen::Vector3d point_world(p_w.x, p_w.y, p_w.z);
  Eigen::Vector3d point_body = Qwb.toRotationMatrix().transpose() * (point_world - twb);
  Eigen::Vector3d point_cam = event_cam_->Rcb() * point_body + event_cam_->tcb();
  Eigen::Vector2d uv = event_cam_->World2Cam(point_cam);

  Eigen::MatrixXd tau1(wy, wx);
  if (!isValidPatch(uv, event_cam_->getUndistortRectifyMask(), wx, wy))
    tau1.setConstant(255.0);
  else {
    bool success = false;
    switch (polarType) {
      case NEUTRAL:
        success = patchInterpolation(inverse_time_surface_, uv, wx, wy, tau1, false);
        break;
      case POSITIVE:
        success = patchInterpolation(inverse_time_surface_positive_, uv, wx, wy, tau1, false);
        break;
      case NEGATIVE:
        success = patchInterpolation(inverse_time_surface_negative_, uv, wx, wy, tau1, false);
        break;
      case DISTANCE_FIELD:
        success = patchInterpolation(distance_field_, uv, wx, wy, tau1, false);
        break;
    }
    if (!success) tau1.setConstant(255.0);
  }
  return tau1.reshaped();
}

Eigen::MatrixXd TimeSurface::df(const CannyEVIT::Point &p_w, const Eigen::Quaterniond &Qwb, const Eigen::Vector3d &twb,
                                int wx, int wy, PolarType polarType) {
  Eigen::MatrixXd jacobian(wx * wy, 6);

  Eigen::Vector3d point_world(p_w.x, p_w.y, p_w.z);
  Eigen::Vector3d point_body = Qwb.toRotationMatrix().transpose() * (point_world - twb);
  Eigen::Vector3d point_cam = event_cam_->Rcb() * point_body + event_cam_->tcb();
  Eigen::Vector2d uv = event_cam_->World2Cam(point_cam);
  if (!isValidPatch(uv, event_cam_->getUndistortRectifyMask(), wx, wy))
    jacobian.setZero();
  else {
    Eigen::MatrixXd gx(wy, wx), gy(wy, wx);
    bool gx_success = false, gy_success = false;
    switch (polarType) {
      case NEUTRAL:
        gx_success = patchInterpolation(gradX_inverse_time_surface_, uv, wx, wy, gx, false);
        gy_success = patchInterpolation(gradY_inverse_time_surface_, uv, wx, wy, gy, false);
        gx = gx / 8.0; gy = gy / 8.0;
        break;
      case POSITIVE:
        gx_success = patchInterpolation(gradX_inverse_time_surface_positive_, uv, wx, wy, gx, false);
        gy_success = patchInterpolation(gradY_inverse_time_surface_positive_, uv, wx, wy, gy, false);
        gx = gx / 8.0; gy = gy / 8.0;
        break;
      case NEGATIVE:
        gx_success = patchInterpolation(gradX_inverse_time_surface_negative_, uv, wx, wy, gx, false);
        gy_success = patchInterpolation(gradY_inverse_time_surface_negative_, uv, wx, wy, gy, false);
        gx = gx / 8.0; gy = gy / 8.0;
        break;
      case DISTANCE_FIELD:
        gx_success = patchInterpolation(gradX_distance_field_, uv, wx, wy, gx, false);
        gy_success = patchInterpolation(gradY_distance_field_, uv, wx, wy, gy, false);
        break;
    }

    if (!gx_success) gx.setZero();
    if (!gy_success) gy.setZero();
    Eigen::MatrixXd grad(2, wx * wy);
    grad.row(0) = gx.reshaped(1, wx * wy);
    grad.row(1) = gy.reshaped(1, wx * wy);

    Eigen::Matrix<double, 2, 3> duv_dPc;
    duv_dPc.setZero();
    duv_dPc.block<2, 2>(0, 0) = event_cam_->getProjectionMatrix().block<2, 2>(0, 0) / point_cam.z();
    const double P11 = event_cam_->getProjectionMatrix()(0, 0);
    const double P12 = event_cam_->getProjectionMatrix()(0, 1);
    const double P14 = event_cam_->getProjectionMatrix()(0, 3);
    const double P21 = event_cam_->getProjectionMatrix()(1, 0);
    const double P22 = event_cam_->getProjectionMatrix()(1, 1);
    const double P24 = event_cam_->getProjectionMatrix()(1, 3);
    const double z2 = pow(point_cam.z(), 2);
    duv_dPc(0, 2) = -(P11 * point_cam.x() + P12 * point_cam.y() + P14) / z2;
    duv_dPc(1, 2) = -(P21 * point_cam.x() + P22 * point_cam.y() + P24) / z2;

    Eigen::Matrix<double, 3, 6> dPc_dRt;
    dPc_dRt.block<3, 3>(0, 0) = -event_cam_->Rcb() * Qwb.conjugate().toRotationMatrix();
    dPc_dRt.block<3, 3>(0, 3) = event_cam_->Rcb() * Utility::skewSymmetric(point_body);

    jacobian = grad.transpose() * duv_dPc * dPc_dRt;
  }

  return jacobian;
}

// static function
void TimeSurface::initTimeSurface(const EventCamera::Ptr &event_cam) {
  event_cam_ = event_cam;
  history_event_ = cv::Mat(event_cam->height(), event_cam->width(), CV_64F, 0.0);
  history_positive_event_ = cv::Mat(event_cam->height(), event_cam->width(), CV_64F, 0.0);
  history_negative_event_ = cv::Mat(event_cam->height(), event_cam->width(), CV_64F, 0.0);
}

void TimeSurface::updateHistoryEvent(EventMsg msg) {
  if (msg.time_stamp_ < history_positive_event_.at<double>(msg.y_, msg.x_)) {
    LOG(WARNING) << "Time surface time misaligned";
    return;
  }

  if (msg.polarity_)
    history_positive_event_.at<double>(msg.y_, msg.x_) = msg.time_stamp_;
  else
    history_negative_event_.at<double>(msg.y_, msg.x_) = msg.time_stamp_;

  history_event_.at<double>(msg.y_, msg.x_) = msg.time_stamp_;
}

std::tuple<TimeSurface::PolarType, double> TimeSurface::determinePolarAndWeight(const CannyEVIT::Point &p_w,
                                                                                const Eigen::Matrix4d &T_current,
                                                                                const Eigen::Matrix4d &T_predict) {
  Eigen::Vector3d p_world(p_w.x, p_w.y, p_w.z);
  Eigen::Vector3d p_gradient_world(p_w.x_gradient_, p_w.y_gradient_, p_w.z_gradient_);

  // This may be confusing, in fact, the "gradient" of the point is not the gradient direction of that point, it is a
  // global point along that gradient direction.
  Eigen::Vector3d p_cam_current = T_current.block<3, 3>(0, 0) * p_world + T_current.block<3, 1>(0, 3);
  Eigen::Vector3d p_cam_predict = T_predict.block<3, 3>(0, 0) * p_world + T_predict.block<3, 1>(0, 3);
  Eigen::Vector3d p_gradient_cam_current = T_current.block<3, 3>(0, 0) * p_gradient_world + T_current.block<3, 1>(0, 3);

  Eigen::Vector2d uv_current = event_cam_->World2Cam(p_cam_current);
  Eigen::Vector2d uv_predict = event_cam_->World2Cam(p_cam_predict);
  Eigen::Vector2d uv_gradient_current = event_cam_->World2Cam(p_gradient_cam_current);

  Eigen::Vector2d flow = uv_predict - uv_current;
  Eigen::Vector2d gradient = uv_gradient_current - uv_current;
  flow.normalize();
  gradient.normalize();

  // gradient is the pixel value increasing direction
  // if the flow is at the opposite direction of gradient, the positive event is triggered
  double cosTheta = flow.dot(gradient);
  double weight = std::abs(cosTheta);
  if (cosTheta < -0.866)
    return std::make_tuple(TimeSurface::PolarType::POSITIVE, weight);
  else if (cosTheta > 0.866)
    return std::make_tuple(TimeSurface::PolarType::NEGATIVE, weight);
  else
    return std::make_tuple(TimeSurface::PolarType::NEUTRAL, weight);
}