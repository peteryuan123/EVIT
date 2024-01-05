//
// Created by mpl on 23-11-9.
//
#include "Frame.h"

using namespace CannyEVIT;

Frame::Frame(TimeSurface::Ptr time_surface_observation, IntegrationBase::Ptr integration, EventCamera::Ptr event_camera)
    : time_stamp_(time_surface_observation->time_stamp_),
      is_active_(true),
      time_surface_observation_(time_surface_observation),
      last_frame_(nullptr),
      integration_(integration),
      event_camera_(event_camera),
      Qwb_(Eigen::Quaterniond::Identity()),
      twb_(Eigen::Vector3d::Zero()),
      Twb_(Eigen::Matrix4d::Identity()),
      velocity_(Eigen::Vector3d::Zero()),
      acc_bias_(Eigen::Vector3d::Zero()),
      gyr_bias_(Eigen::Vector3d::Zero()),
      opt_pose_(Eigen::Vector<double, 7>::Zero()),
      opt_speed_bias_(Eigen::Vector<double, 9>::Zero()) {}

void Frame::optToState() {
  set_twb(opt_pose_.segment<3>(0));
  set_Rwb(Eigen::Quaterniond(opt_pose_[6], opt_pose_[3], opt_pose_[4], opt_pose_[5]));
  set_velocity(opt_speed_bias_.segment<3>(0));
  set_Ba(opt_speed_bias_.segment<3>(3));
  set_Bg(opt_speed_bias_.segment<3>(6));
//  std::cout << twb_.transpose() << std::endl;
//  std::cout << Qwb_.coeffs().transpose() << std::endl;
//  std::cout << velocity_.transpose() << std::endl;
//  std::cout << acc_bias_.transpose() << std::endl;
//  std::cout << gyr_bias_.transpose() << std::endl;
//  std::cout << "--------------" << std::endl;
}

void Frame::stateToOpt() {
  opt_pose_ << twb_.x(), twb_.y(), twb_.z(), Qwb_.x(), Qwb_.y(), Qwb_.z(), Qwb_.w();
  opt_speed_bias_
      << velocity_.x(), velocity_.y(), velocity_.z(), acc_bias_.x(), acc_bias_.y(), acc_bias_.z(), gyr_bias_.x(), gyr_bias_.y(), gyr_bias_.z();
}

// getter
Eigen::Matrix3d Frame::Rwb() { return Qwb_.toRotationMatrix(); }

Eigen::Vector3d Frame::twb() { return twb_; }

Eigen::Quaterniond Frame::Qwb() { return Qwb_; }

Eigen::Matrix4d Frame::Twb() { return Twb_; }

Eigen::Matrix3d Frame::Rwc() { return Qwb_.toRotationMatrix() * event_camera_->Rbc(); }

Eigen::Vector3d Frame::twc() { return Qwb_ * event_camera_->tbc() + twb_; }

Eigen::Quaterniond Frame::Qwc() { return Qwb_ * Eigen::Quaterniond(event_camera_->Rbc()); }

Eigen::Matrix4d Frame::Twc() { return Twb_ * event_camera_->Tbc(); }

Eigen::Vector3d Frame::velocity() { return velocity_; }

Eigen::Vector3d Frame::Ba() { return acc_bias_; }

Eigen::Vector3d Frame::Bg() { return gyr_bias_; }

bool Frame::isActive(){
  return is_active_;
}
// setter

void Frame::set_active(bool active){
  is_active_ = active;
}

void Frame::set_integration(const IntegrationBase::Ptr &integration) { integration_ = integration; }

void Frame::set_timeStamp(double time_stamp) { time_stamp_ = time_stamp; }

void Frame::set_Twb(const Eigen::Matrix4d &Twb) {
  Twb_ = Twb;
  Qwb_ = Eigen::Quaterniond(Twb.block<3, 3>(0, 0));
  Qwb_.normalize();
  twb_ = Twb.block<3, 1>(0, 3);
}

void Frame::set_Twb(const Eigen::Matrix3d &Rwb, const Eigen::Vector3d &twb) {
  Twb_.block<3, 3>(0, 0) = Rwb;
  Twb_.block<3, 1>(0, 3) = twb;
  Qwb_ = Eigen::Quaterniond(Rwb);
  Qwb_.normalize();
  twb_ = twb;
}

void Frame::set_Twb(const Eigen::Quaterniond &Qwb, const Eigen::Vector3d &twb) {
  Qwb_ = Qwb;
  Qwb_.normalize();
  Twb_.block<3, 3>(0, 0) = Qwb_.toRotationMatrix();
  Twb_.block<3, 1>(0, 3) = twb;
  twb_ = twb;
}

void Frame::set_Rwb(const Eigen::Matrix3d &Rwb) {
  Twb_.block<3, 3>(0, 0) = Rwb;
  Qwb_ = Eigen::Quaterniond(Rwb);
  Qwb_.normalize();
}

void Frame::set_Rwb(const Eigen::Quaterniond &Qwb) {
  Qwb_ = Qwb;
  Qwb_.normalize();
  Twb_.block<3, 3>(0, 0) = Qwb_.toRotationMatrix();
}

void Frame::set_twb(const Eigen::Vector3d &twb) {
  Twb_.block<3, 1>(0, 3) = twb;
  twb_ = twb;
}

void Frame::set_velocity(const Eigen::Vector3d &velocity) { velocity_ = velocity; }

void Frame::set_Ba(const Eigen::Vector3d &ba) { acc_bias_ = ba; }

void Frame::set_Bg(const Eigen::Vector3d &bg) { gyr_bias_ = bg; }