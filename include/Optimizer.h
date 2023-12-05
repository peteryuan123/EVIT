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

namespace CannyEVIT
{
    class Optimizer
    {
    public:
        typedef std::shared_ptr<Optimizer> Ptr;

        Optimizer(const std::string& config_path, EventCamera::Ptr event_camera);

        bool OptimizeEventProblemCeres(pCloud cloud, Frame::Ptr frame);
        bool OptimizeSlidingWindowProblemCeres(pCloud cloud, std::deque<Frame::Ptr> window);
        bool OptimizeSlidingWindowProblemCeresBatch(pCloud cloud, std::deque<Frame::Ptr> window);

        bool OptimizeVelovityBias(const std::vector<Frame::Ptr>& window);
        bool initVelocityBias(const std::vector<Frame::Ptr>& window);

    public:
        EventCamera::Ptr event_camera_;

        int patch_size_X_;
        int patch_size_Y_;
    };


}



#endif //CANNYEVIT_OPTIMIZER_H
