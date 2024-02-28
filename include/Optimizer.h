//
// Created by mpl on 23-11-11.
//

#ifndef CANNYEVIT_OPTIMIZER_H
#define CANNYEVIT_OPTIMIZER_H

#include <memory>
#include <opencv2/opencv.hpp>

#include "EventCamera.h"
#include "Frame.h"
#include "Type.h"

namespace CannyEVIT {
// Note: Optimizer is only responsible to optimize the "opt vector" in each frame, so remember to update state after optimization
class Optimizer {
 public:
  typedef std::shared_ptr<Optimizer> Ptr;

  Optimizer(const std::string &config_path, EventCamera::Ptr event_camera);

  bool OptimizeEventProblemCeres(pCloud cloud, Frame::Ptr frame);
  bool OptimizeSlidingWindowProblemCeres(pCloud cloud, std::deque<Frame::Ptr> &window);
  bool OptimizeSlidingWindowProblemCeresBatch(pCloud cloud, std::deque<Frame::Ptr> &window);

  bool OptimizeEventProblemEigen(pCloud cloud, Frame::Ptr frame);
  bool OptimizeSlidingWindowProblemEigen(pCloud cloud, std::deque<Frame::Ptr> &window);

  bool OptimizeVelovityBias(const std::vector<Frame::Ptr> &window);
  bool initVelocityBias(const std::vector<Frame::Ptr> &window);

  void sampleVisibleIndices(pCloud cloud, const std::deque<Frame::Ptr> &window, std::vector<size_t> &set_samples);

 public:
  EventCamera::Ptr event_camera_;

  // config param
  int patch_size_X_;
  int patch_size_Y_;
  bool polarity_prediction_;
  size_t max_registration_point_;
  size_t batch_size_;
  TimeSurface::FieldType field_type_;
  TimeSurface::VisualizationType vis_type_;

};

}  // namespace CannyEVIT

#endif  // CANNYEVIT_OPTIMIZER_H
