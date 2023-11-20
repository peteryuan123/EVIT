//
// Created by mpl on 23-11-12.
//

#ifndef CANNYEVIT_EVENTFACTOR_H
#define CANNYEVIT_EVENTFACTOR_H

#include <ceres/ceres.h>
#include "Type.h"
#include "TimeSurface.h"

namespace CannyEVIT
{
class EventFactor : public ceres::CostFunction
    {
    public:
        EventFactor() = delete;
        EventFactor(const Point& p_w, const TimeSurface::Ptr& time_surface, int wx, int wy);

        virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const;

        Point p_w_;
        TimeSurface::Ptr time_surface_;
        int wx_, wy_;

    };


}

#endif //CANNYEVIT_EVENTFACTOR_H
