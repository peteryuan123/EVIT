//
// Created by mpl on 24-1-4.
//

#ifndef CANNYEVIT_INCLUDE_VIEWER_H_
#define CANNYEVIT_INCLUDE_VIEWER_H_

#include <pangolin/pangolin.h>
#include "System.h"

namespace CannyEVIT {

class System;

class Viewer {

 public:

  Viewer(const std::string &config_path, System *system);
  ~Viewer() = default;

  void Run();

  void drawCamera(const Eigen::Matrix4d &Twc, bool isActivate);
  void drawAxis(const Eigen::Matrix4d &Twb);

 public:
  System *system_;

  float point_size_;
  float frame_size_;

  std::vector<Eigen::Matrix4d> history_camera_poses_;
  std::vector<Eigen::Matrix4d> history_imu_poses_;

  std::vector<Eigen::Matrix4d> tracked_camera_poses_;
  std::vector<Eigen::Matrix4d> tracked_imu_poses_;

  std::mutex viewer_mutex_;

  bool update_;
};

}

#endif //CANNYEVIT_INCLUDE_VIEWER_H_
