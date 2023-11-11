//
// Created by mpl on 23-11-11.
//

#include "Optimizer.h"
#include "easylogging++.h"
using namespace CannyEVIT;

Optimizer::Optimizer(const std::string &config_path, EventCamera::Ptr event_camera)
: event_camera_(event_camera), patch_size_X_(0), patch_size_Y_(0)
{

    cv::FileStorage fs(config_path, cv::FileStorage::READ );
    if (!fs.isOpened())
        LOG(FATAL)<< "config not open: " << config_path;\

    if (fs["patch_size_X"].isNone())
        LOG(ERROR) << "config: patch_size_X is not set";
    patch_size_X_ = fs["patch_size_X"];

    if (fs["patch_size_Y"].isNone())
        LOG(ERROR) << "config: patch_size_Y is not set";
    patch_size_Y_ = fs["patch_size_Y"];
}
