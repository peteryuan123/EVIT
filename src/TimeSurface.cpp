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
  processTimeSurface(history_event_,
                     time_stamp,
                     decay_factor_,
                     neutral_visualization_fields_[VisualizationType::TIME_SURFACE],
                     neutral_fields_[FieldType::INV_TIME_SURFACE]);

  processTimeSurface(history_positive_event_,
                     time_stamp,
                     decay_factor_,
                     positive_visualization_fields_[VisualizationType::TIME_SURFACE],
                     positive_fields_[FieldType::INV_TIME_SURFACE]);

  processTimeSurface(history_negative_event_,
                     time_stamp,
                     decay_factor_,
                     negative_visualization_fields_[VisualizationType::TIME_SURFACE],
                     negative_fields_[FieldType::INV_TIME_SURFACE]);

  constructDistanceField(neutral_visualization_fields_[VisualizationType::TIME_SURFACE],
                         neutral_fields_[FieldType::DISTANCE_FIELD],
                         neutral_visualization_fields_[VisualizationType::CANNY]);

  constructDistanceField(positive_visualization_fields_[VisualizationType::TIME_SURFACE],
                         positive_fields_[FieldType::DISTANCE_FIELD],
                         positive_visualization_fields_[VisualizationType::CANNY]);

  constructDistanceField(negative_visualization_fields_[VisualizationType::TIME_SURFACE],
                         negative_fields_[FieldType::DISTANCE_FIELD],
                         negative_visualization_fields_[VisualizationType::CANNY]);

}

void TimeSurface::processTimeSurface(const cv::Mat &history_event,
                                     double time_stamp,
                                     double decay_factor,
                                     cv::Mat &time_surface,
                                     CannyEVIT::OptField &inv_time_surface_field) {
  cv::exp((history_event - time_stamp) / decay_factor, time_surface);
  event_cam_->undistortImage(time_surface, time_surface);
  time_surface = time_surface * 255.0;
//  cv::Point minIdx, maxIdx;
//  double minVal, maxVal;
//  cv::minMaxLoc(time_surface, &minVal, &maxVal, &minIdx, &maxIdx);
//  std::cout << "before blur:" << std::endl;
//  std::cout << "min_val:" << minVal << ",max_val" << maxVal << std::endl;
  cv::GaussianBlur(time_surface, time_surface, cv::Size(5, 5), 0.0);
//  cv::minMaxLoc(time_surface, &minVal, &maxVal, &minIdx, &maxIdx);
//  std::cout << "after blur:" << std::endl;
//  std::cout << "min_val:" << minVal << ",max_val" << maxVal << std::endl;


  cv::Mat inverse_time_surface_cv = 255.0 - time_surface;

//  time_surface.convertTo(time_surface, CV_8U);
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

  cv::cv2eigen(inverse_time_surface_cv, inv_time_surface_field.field_);
  cv::cv2eigen(inverse_gradX_cv, inv_time_surface_field.gradX_);
  cv::cv2eigen(inverse_gradY_cv, inv_time_surface_field.gradY_);

}

void TimeSurface::constructDistanceField(const cv::Mat &time_surface,
                                         CannyEVIT::OptField &distance_field,
                                         cv::Mat &visualization_img) {
  // test distance field TODO: NEED TO ACCELERATE IT
  Eigen::MatrixXd time_surface_eigen;
  cv::cv2eigen(time_surface, time_surface_eigen);
  Eigen::ArrayXXd grad_x, grad_y, grad_mag;
  image_processing::sobel_mag(time_surface_eigen, grad_x, grad_y, grad_mag);
  std::vector<std::pair<int, int>> uv_edge;
  image_processing::canny(grad_mag, grad_x, grad_y, uv_edge, 50);
  Eigen::ArrayXXd distance_field_tmp;
  image_processing::chebychevDistanceField(event_cam_->height(), event_cam_->width(), uv_edge, distance_field_tmp);
  distance_field.field_ = distance_field_tmp;
  Eigen::ArrayXXd gradX_distance_field_tmp, gradY_distance_field_tmp;
  image_processing::sobel(distance_field.field_, gradX_distance_field_tmp, gradY_distance_field_tmp);
  distance_field.gradX_ = gradX_distance_field_tmp;
  distance_field.gradY_ = gradY_distance_field_tmp;

  visualization_img = cv::Mat(event_cam_->height(), event_cam_->width(), CV_8U, cv::Scalar(0.0));
  for (auto &pos : uv_edge)
    visualization_img.at<uint8_t>(pos.first, pos.second) = 255;
}

void TimeSurface::drawCloud(CannyEVIT::pCloud cloud,
                            const Eigen::Matrix4d &Twc,
                            const std::string &window_name,
                            PolarType polarType,
                            VisualizationType visType,
                            bool showGrad) {
  std::unordered_map<VisualizationType, cv::Mat> *visualization_field_ptr;
  switch (polarType) {
    case NEUTRAL:visualization_field_ptr = &neutral_visualization_fields_;
      break;
    case POSITIVE:visualization_field_ptr = &positive_visualization_fields_;
      break;
    case NEGATIVE:visualization_field_ptr = &negative_visualization_fields_;
      break;
  }
  cv::Mat img_drawing = visualization_field_ptr->at(visType).clone();

  img_drawing.convertTo(img_drawing, CV_8UC1);
//  img_drawing.convertTo(img_drawing, CV_32FC1);
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

bool TimeSurface::patchInterpolation(const Eigen::MatrixXd &img,
                                     const Eigen::Vector2d &location,
                                     int wx,
                                     int wy,
                                     Eigen::MatrixXd &patch,
                                     bool debug) {
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

Eigen::VectorXd TimeSurface::evaluate(const CannyEVIT::Point &p_w,
                                      const Eigen::Quaterniond &Qwb,
                                      const Eigen::Vector3d &twb,
                                      int wx,
                                      int wy,
                                      PolarType polarType,
                                      FieldType fieldType) {
  Eigen::Vector3d point_world(p_w.x, p_w.y, p_w.z);
  Eigen::Vector3d point_body = Qwb.toRotationMatrix().transpose() * (point_world - twb);
  Eigen::Vector3d point_cam = event_cam_->Rcb() * point_body + event_cam_->tcb();
  Eigen::Vector2d uv = event_cam_->World2Cam(point_cam);

  std::unordered_map<FieldType, OptField> *fields;
  switch (polarType) {

    case NEUTRAL:fields = &neutral_fields_;
      break;
    case POSITIVE:fields = &positive_fields_;
      break;
    case NEGATIVE:fields = &negative_fields_;
      break;
  }

  OptField &opt_field = fields->at(fieldType);
  Eigen::MatrixXd tau1(wy, wx);
  if (!isValidPatch(uv, event_cam_->getUndistortRectifyMask(), wx, wy))
    tau1.setConstant(255.0);
  else {
    bool success = patchInterpolation(opt_field.field_, uv, wx, wy, tau1, false);
    if (!success) tau1.setConstant(255.0);
  }
  return tau1.reshaped();
}

Eigen::MatrixXd TimeSurface::df(const CannyEVIT::Point &p_w,
                                const Eigen::Quaterniond &Qwb,
                                const Eigen::Vector3d &twb,
                                int wx,
                                int wy,
                                PolarType polarType,
                                FieldType fieldType) {
  Eigen::MatrixXd jacobian(wx * wy, 6);

  Eigen::Vector3d point_world(p_w.x, p_w.y, p_w.z);
  Eigen::Vector3d point_body = Qwb.toRotationMatrix().transpose() * (point_world - twb);
  Eigen::Vector3d point_cam = event_cam_->Rcb() * point_body + event_cam_->tcb();
  Eigen::Vector2d uv = event_cam_->World2Cam(point_cam);

  std::unordered_map<FieldType, OptField> *fields;
  switch (polarType) {

    case NEUTRAL:fields = &neutral_fields_;
      break;
    case POSITIVE:fields = &positive_fields_;
      break;
    case NEGATIVE:fields = &negative_fields_;
      break;
  }
  OptField &opt_field = fields->at(fieldType);
  if (!isValidPatch(uv, event_cam_->getUndistortRectifyMask(), wx, wy))
    jacobian.setZero();
  else {
    Eigen::MatrixXd gx(wy, wx), gy(wy, wx);
    bool gx_success = false, gy_success = false;
    gx_success = patchInterpolation(opt_field.gradX_, uv, wx, wy, gx, false);
    gy_success = patchInterpolation(opt_field.gradY_, uv, wx, wy, gy, false);
    if (!gx_success) gx.setZero();
    if (!gy_success) gy.setZero();

    Eigen::MatrixXd grad(2, wx * wy);
    grad.row(0) = gx.reshaped(1, wx * wy);
    grad.row(1) = gy.reshaped(1, wx * wy);

    if(fieldType == INV_TIME_SURFACE) //TODO: MAKE ALL CONSISTENT
      grad = grad / 8.0;

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