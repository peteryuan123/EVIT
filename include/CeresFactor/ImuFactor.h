//
// Created by mpl on 23-11-11.
//
#ifndef CANNYEVIT_IMUFACTOR_H
#define CANNYEVIT_IMUFACTOR_H

#include <iostream>
#include <eigen3/Eigen/Dense>
#include <ceres/ceres.h>
#include "ImuIntegration.h"

namespace CannyEVIT
{
    class IMUFactor : public ceres::SizedCostFunction<15, 7, 9, 7, 9>
    {
    public:
        IMUFactor() = delete;
        IMUFactor(IntegrationBase* _pre_integration);

        virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const;

        IntegrationBase* pre_integration;

        //bool Evaluate_Direct(double const *const *parameters, Eigen::Matrix<double, 15, 1> &residuals, Eigen::Matrix<double, 15, 30> &jacobians);

        //void checkCorrection();
        //void checkTransition();
        //void checkJacobian(double **parameters);
    };


}



#endif //CANNYEVIT_IMUFACTOR_H
