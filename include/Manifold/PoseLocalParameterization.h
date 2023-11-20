//
// Created by mpl on 23-11-11.
// THIS IS COPIED FROM VINSMONO
//

#ifndef CANNYEVIT_POSELOCALPARAMETERIZATION_H
#define CANNYEVIT_POSELOCALPARAMETERIZATION_H

#include <ceres/ceres.h>

namespace CannyEVIT {
    class PoseLocalParameterization : public ceres::LocalParameterization {
        virtual bool Plus(const double *x, const double *delta, double *x_plus_delta) const;

        virtual bool ComputeJacobian(const double *x, double *jacobian) const;

        virtual int GlobalSize() const;

        virtual int LocalSize() const;
    };
}
#endif //CANNYEVIT_POSELOCALPARAMETERIZATION_H
