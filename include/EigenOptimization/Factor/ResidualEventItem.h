//
// Created by mpl on 23-11-20.
//

#ifndef CANNYEVIT_RESIDUALEVENTITEM_H
#define CANNYEVIT_RESIDUALEVENTITEM_H

#include <Eigen/Eigen>
#include <memory>
#include "Type.h"

namespace CannyEVIT
{
    struct ResidualEventInfo
    {
        typedef std::shared_ptr<ResidualEventInfo> Ptr;

        ResidualEventInfo(const Point& point, size_t residual_size)
        : p_(point), irls_weight_(Eigen::VectorXd(residual_size))
        {}

        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        Point p_;           // 3D coordinate in world frame
        Eigen::VectorXd irls_weight_;
    };

}



#endif //CANNYEVIT_RESIDUALEVENTITEM_H
