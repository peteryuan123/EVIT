//
// Created by mpl on 23-12-22.
//

#ifndef CANNYEVIT_INCLUDE_EIGENOPTIMIZATION_FACTOR_RESIDUALIMUITEM_H_
#define CANNYEVIT_INCLUDE_EIGENOPTIMIZATION_FACTOR_RESIDUALIMUITEM_H_

#include <Eigen/Core>
#include "ImuIntegration.h"

namespace CannyEVIT {

struct ResidualImuItem {
  ResidualImuItem(IntegrationBase::Ptr pre_integration) : pre_integration_(pre_integration){};

  void computeResidual(const Eigen::Vector3d &Pi,
                       const Eigen::Quaterniond &Qi,
                       const Eigen::Vector3d &Vi,
                       const Eigen::Vector3d &Bai,
                       const Eigen::Vector3d &Bgi,
                       const Eigen::Vector3d &Pj,
                       const Eigen::Quaterniond &Qj,
                       const Eigen::Vector3d &Vj,
                       const Eigen::Vector3d &Baj,
                       const Eigen::Vector3d &Bgj);

  void computeJacobian(const Eigen::Vector3d &Pi,
                       const Eigen::Quaterniond &Qi,
                       const Eigen::Vector3d &Vi,
                       const Eigen::Vector3d &Bai,
                       const Eigen::Vector3d &Bgi,
                       const Eigen::Vector3d &Pj,
                       const Eigen::Quaterniond &Qj,
                       const Eigen::Vector3d &Vj,
                       const Eigen::Vector3d &Baj,
                       const Eigen::Vector3d &Bgj);

  IntegrationBase::Ptr pre_integration_;

  Eigen::Vector<double, 15> residuals_;
  Eigen::Matrix<double, 15, 15> jacobian_i_;
  Eigen::Matrix<double, 15, 15> jacobian_j_;
};

using ResidualImuItems = std::vector<ResidualImuItem, Eigen::aligned_allocator<ResidualImuItem>>;

}

#endif //CANNYEVIT_INCLUDE_EIGENOPTIMIZATION_FACTOR_RESIDUALIMUITEM_H_
