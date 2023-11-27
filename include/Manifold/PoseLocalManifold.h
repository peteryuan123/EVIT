//
// Created by mpl on 23-11-11.
//

#ifndef CANNYEVIT_QUATERNIONRIGHTMANIFOLD_H
#define CANNYEVIT_QUATERNIONRIGHTMANIFOLD_H

#include <ceres/ceres.h>

namespace CannyEVIT
{
    class PoseLocalManifold: public ceres::Manifold
    {
        int AmbientSize() const override;
        int TangentSize() const override;
        bool Plus(const double *x, const double *delta, double *x_plus_delta) const override;
        bool PlusJacobian(const double* x, double* jacobian) const override;
        bool Minus(const double *y, const double *x, double *y_minus_x) const override;
        bool MinusJacobian(const double* x, double* jacobian) const override;

        static constexpr int kW = 6;
        static constexpr int kX = 3;
        static constexpr int kY = 4;
        static constexpr int kZ = 5;
    };
}


#endif //CANNYEVIT_QUATERNIONRIGHTMANIFOLD_H
