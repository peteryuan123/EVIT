//
// Created by mpl on 23-12-22.
//
#include "EigenOptimization/Factor/ResidualImuItem.h"
#include "Util.h"
#include <glog/logging.h>
using namespace CannyEVIT;

void ResidualImuItem::computeResidual(const Eigen::Vector3d &Pi,
                                      const Eigen::Quaterniond &Qi,
                                      const Eigen::Vector3d &Vi,
                                      const Eigen::Vector3d &Bai,
                                      const Eigen::Vector3d &Bgi,
                                      const Eigen::Vector3d &Pj,
                                      const Eigen::Quaterniond &Qj,
                                      const Eigen::Vector3d &Vj,
                                      const Eigen::Vector3d &Baj,
                                      const Eigen::Vector3d &Bgj) {
  Eigen::Matrix<double, 15, 15> sqrt_info =
      Eigen::LLT<Eigen::Matrix<double, 15, 15>>(pre_integration_->covariance.inverse()).matrixL().transpose();

  residuals_ = sqrt_info * pre_integration_->evaluate(Pi, Qi, Vi, Bai, Bgi, Pj, Qj, Vj, Baj, Bgj);
}

void ResidualImuItem::computeJacobian(const Eigen::Vector3d &Pi,
                                      const Eigen::Quaterniond &Qi,
                                      const Eigen::Vector3d &Vi,
                                      [[maybe_unused]] const Eigen::Vector3d &Bai,
                                      const Eigen::Vector3d &Bgi,
                                      const Eigen::Vector3d &Pj,
                                      const Eigen::Quaterniond &Qj,
                                      const Eigen::Vector3d &Vj,
                                      [[maybe_unused]] const Eigen::Vector3d &Baj,
                                      [[maybe_unused]] const Eigen::Vector3d &Bgj) {
  jacobian_i_.setZero();
  jacobian_j_.setZero();
  if (pre_integration_->jacobian.maxCoeff() > 1e8 || pre_integration_->jacobian.minCoeff() < -1e8) {
    LOG(WARNING) << "numerical unstable in preintegration";
    // std::cout << pre_integration->jacobian << std::endl;
  }
  Eigen::Matrix<double, 15, 15> sqrt_info =
      Eigen::LLT<Eigen::Matrix<double, 15, 15>>(pre_integration_->covariance.inverse()).matrixL().transpose();

  double sum_dt = pre_integration_->sum_dt;
  Eigen::Matrix3d dp_dba = pre_integration_->jacobian.template block<3, 3>(O_P, O_BA);
  Eigen::Matrix3d dp_dbg = pre_integration_->jacobian.template block<3, 3>(O_P, O_BG);

  Eigen::Matrix3d dq_dbg = pre_integration_->jacobian.template block<3, 3>(O_R, O_BG);

  Eigen::Matrix3d dv_dba = pre_integration_->jacobian.template block<3, 3>(O_V, O_BA);
  Eigen::Matrix3d dv_dbg = pre_integration_->jacobian.template block<3, 3>(O_V, O_BG);

  Eigen::Quaterniond corrected_delta_q =
      pre_integration_->delta_q * Utility::deltaQ(dq_dbg * (Bgi - pre_integration_->linearized_bg));

  // ---------------------Jacobian for pose_i
  // OP
  jacobian_i_.block<3, 3>(O_P, O_P) = -Qi.inverse().toRotationMatrix();
  jacobian_i_.block<3, 3>(O_P, O_R) =
      Utility::skewSymmetric(Qi.inverse() * (0.5 * IntegrationBase::G * sum_dt * sum_dt + Pj - Pi - Vi * sum_dt));
  jacobian_i_.block<3, 3>(O_P, O_V) = -Qi.inverse().toRotationMatrix() * sum_dt;
  jacobian_i_.block<3, 3>(O_P, O_BA) = -dp_dba;
  jacobian_i_.block<3, 3>(O_P, O_BG) = -dp_dbg;

  // OR
  jacobian_i_.block<3, 3>(O_R, O_R) =
      -(Utility::Qleft(Qj.inverse() * Qi) * Utility::Qright(corrected_delta_q)).bottomRightCorner<3, 3>();
  jacobian_i_.block<3, 3>(O_R, O_BG) =
      -Utility::Qleft(Qj.inverse() * Qi * pre_integration_->delta_q).bottomRightCorner<3, 3>() * dq_dbg;

  // OV
  jacobian_i_.block<3, 3>(O_V, O_R) = Utility::skewSymmetric(Qi.inverse() * (IntegrationBase::G * sum_dt + Vj - Vi));
  jacobian_i_.block<3, 3>(O_V, O_V) = -Qi.inverse().toRotationMatrix();
  jacobian_i_.block<3, 3>(O_V, O_BA) = -dv_dba;
  jacobian_i_.block<3, 3>(O_V, O_BG) = -dv_dbg;

  // OBA
  jacobian_i_.block<3, 3>(O_BA, O_BA) = -Eigen::Matrix3d::Identity();

  // OBG
  jacobian_i_.block<3, 3>(O_BG, O_BG) = -Eigen::Matrix3d::Identity();
  jacobian_i_ = sqrt_info * jacobian_i_;

  // ------------------Jacobian for pose_j
  // OP
  jacobian_j_.block<3, 3>(O_P, O_P) = Qi.inverse().toRotationMatrix();

  // OR
  jacobian_j_.block<3, 3>(O_R, O_R) =
      Utility::Qleft(corrected_delta_q.inverse() * Qi.inverse() * Qj).bottomRightCorner<3, 3>();

  // OV
  jacobian_j_.block<3, 3>(O_V, O_V) = Qi.inverse().toRotationMatrix();

  // OBA
  jacobian_j_.block<3, 3>(O_BA, O_BA) = Eigen::Matrix3d::Identity();

  // OBG
  jacobian_j_.block<3, 3>(O_BG, O_BG) = Eigen::Matrix3d::Identity();
  jacobian_j_ = sqrt_info * jacobian_j_;

}
