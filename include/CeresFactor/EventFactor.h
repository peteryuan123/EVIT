//
// Created by mpl on 23-11-12.
//

#ifndef CANNYEVIT_EVENTFACTOR_H
#define CANNYEVIT_EVENTFACTOR_H

#include <ceres/ceres.h>

#include "TimeSurface.h"
#include "Type.h"

namespace CannyEVIT {
class EventFactor : public ceres::CostFunction {
 public:
  EventFactor() = delete;
  EventFactor(const Point &p_w,
              const TimeSurface::Ptr &time_surface,
              int wx,
              int wy,
              TimeSurface::PolarType polar = TimeSurface::PolarType::NEUTRAL,
              TimeSurface::FieldType field_type = TimeSurface::FieldType::INV_TIME_SURFACE,
              double weight = 1.0);

  virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const;

  TimeSurface::PolarType polar_;
  TimeSurface::FieldType field_type_;
  Point p_w_;
  TimeSurface::Ptr time_surface_;
  int wx_, wy_;
  double weight_;
};

}  // namespace CannyEVIT

#endif  // CANNYEVIT_EVENTFACTOR_H
