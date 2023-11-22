//
// Created by mpl on 23-11-11.
//
#include "Manifold/PoseLocalManifold.h"
#include "Util.h"

using namespace CannyEVIT;

int PoseLocalManifold::AmbientSize() const
{
    return 7;
}

int PoseLocalManifold::TangentSize() const
{
    return 6;
}


bool PoseLocalManifold::Plus(const double *x, const double *delta, double *x_plus_delta) const
{
//    Eigen::Map<const Eigen::Vector3d> _p(x);
//    Eigen::Map<const Eigen::Quaterniond> _q(x + 3);
//
//    Eigen::Map<const Eigen::Vector3d> dp(delta);
//
//    Eigen::Quaterniond dq = Utility::deltaQ(Eigen::Map<const Eigen::Vector3d>(delta + 3));
//
//    Eigen::Map<Eigen::Vector3d> p(x_plus_delta);
//    Eigen::Map<Eigen::Quaterniond> q(x_plus_delta + 3);
//
//    p = _p + dp;
//    q = (_q * dq).normalized();


    double q_delta[4];
    q_delta[0] = delta[0] / 2.0;
    q_delta[1] = delta[1] / 2.0;
    q_delta[2] = delta[2] / 2.0;
    q_delta[3] = 1.0;

    x_plus_delta[0] = x[0] + delta[0];
    x_plus_delta[1] = x[1] + delta[1];
    x_plus_delta[2] = x[2] + delta[2];
    x_plus_delta[kW] =
            q_delta[kW-3] * x[kW] - q_delta[kX-3] * x[kX] -
            q_delta[kY-3] * x[kY] - q_delta[kZ-3] * x[kZ];
    x_plus_delta[kX] =
            q_delta[kX-3] * x[kW] + q_delta[kW-3] * x[kX] +
            q_delta[kZ-3] * x[kY] - q_delta[kY-3] * x[kZ];
    x_plus_delta[kY] =
            q_delta[kY-3] * x[kW] - q_delta[kZ-3] * x[kX] +
            q_delta[kW-3] * x[kY] + q_delta[kX-3] * x[kZ];
    x_plus_delta[kZ] =
            q_delta[kZ-3] * x[kW] + q_delta[kY-3] * x[kX] -
            q_delta[kX-3] * x[kY] + q_delta[kW-3] * x[kZ];
    return true;
}

bool PoseLocalManifold::PlusJacobian(const double *x, double *jacobian) const {
    Eigen::Map<Eigen::Matrix<double, 7, 6, Eigen::RowMajor>> jacobian_eigen(jacobian);

    Eigen::Map<const Eigen::Vector3d> q_vec(x+3);

    jacobian_eigen.block<3, 3>(0, 0).setIdentity();
    jacobian_eigen.block<3, 3>(0, 3).setZero();
    jacobian_eigen.block<4, 3>(3, 0).setZero();
    jacobian_eigen.block<3, 3>(3, 3) = x[kW] * Eigen::Matrix3d::Identity() + Utility::skewSymmetric(q_vec);
    jacobian_eigen.block<1, 3>(6, 3) = -q_vec.transpose();
    return true;
}

bool PoseLocalManifold::Minus(const double *y, const double *x, double *y_minus_x) const
{
//    Eigen::Map<Eigen::Vector3d> delta_p(y_minus_x);
//    Eigen::Map<const Eigen::Vector3d> p_y(y);
//    Eigen::Map<const Eigen::Vector3d> p_x(x);
//    delta_p = p_y - p_x;
//
//    Eigen::Map<const Eigen::Quaterniond> q_y(y+3);
//    Eigen::Map<const Eigen::Quaterniond> q_x(x+3);
//    Eigen::Quaterniond delta_Quaternion = q_x.conjugate() * q_y;
//
//    Eigen::Map<Eigen::Vector3d> delta_q(y_minus_x+3);
//    delta_q = delta_Quaternion.vec() * 2.0;

    y_minus_x[0] = y[0] - x[0];
    y_minus_x[1] = y[1] - x[1];
    y_minus_x[2] = y[2] - x[2];

    double ambient_y_minus_x[4];
    ambient_y_minus_x[kW-3] =
            y[kW] * x[kW] + y[kX] * x[kX] +
            y[kY] * x[kY] + y[kZ] * x[kZ];
    ambient_y_minus_x[kX-3] =
            -y[kW] * x[kX] + y[kX] * x[kW] +
            y[kY] * x[kZ] - y[kZ] * x[kY];
    ambient_y_minus_x[kY-3] =
            -y[kW] * x[kY] - y[kX] * x[kZ] +
            y[kY] * x[kW] + y[kZ] * x[kX];
    ambient_y_minus_x[kZ-3] =
            -y[kW] * x[kZ] + y[kX] * x[kY] -
            y[kY] * x[kX] + y[kZ] * x[kW];

    y_minus_x[3] = ambient_y_minus_x[0] * 2;
    y_minus_x[4] = ambient_y_minus_x[1] * 2;
    y_minus_x[5] = ambient_y_minus_x[2] * 2;
    return true;
}

bool PoseLocalManifold::MinusJacobian(const double *x, double *jacobian) const
{
    Eigen::Map<Eigen::Matrix<double, 6, 7, Eigen::RowMajor>> jacobian_eigen(jacobian);

    Eigen::Map<const Eigen::Vector3d> q_vec(x+3);

    jacobian_eigen.block<3, 3>(0, 0).setIdentity();
    jacobian_eigen.block<3, 4>(0, 3).setZero();
    jacobian_eigen.block<3, 3>(3, 0).setZero();
    jacobian_eigen.block<3, 3>(3, 3) = 2 * (x[kW] * Eigen::Matrix3d::Identity() - Utility::skewSymmetric(q_vec));
    jacobian_eigen.block<3, 1>(3, 6) = -2 * q_vec;
    return true;
}