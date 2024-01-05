//
// Created by mpl on 23-11-7.
//

#ifndef CANNYEVIT_SYSTEM
#define CANNYEVIT_SYSTEM

#include <Eigen/Geometry>
#include <condition_variable>
#include <deque>
#include <string>
#include <thread>

#include "EventCamera.h"
#include "Frame.h"
#include "Optimizer.h"
#include "Type.h"
#include "Viewer.h"

namespace CannyEVIT {

class Viewer;

class System {

  enum State {
    Init,
    Tracking,
  };

 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  System(const std::string &config_path);
  ~System();

 public:
  void readParam(const std::string &config_path);

  void GrabEventMsg(double time_stamp, size_t x, size_t y, bool polarity);
  void GrabImuMsg(double time_stamp, double accX, double accY, double accZ, double gyrX, double gyrY, double gyrZ);

  bool getMeasurement(FrameData &data);
  void process();

  void predictIMUPose(double dt,
                      const Eigen::Vector3d &acc0,
                      const Eigen::Vector3d &gyr0,
                      const Eigen::Vector3d &acc1,
                      const Eigen::Vector3d &gyr1,
                      const Eigen::Vector3d &ba,
                      const Eigen::Vector3d &bg,
                      Eigen::Quaterniond &Q,
                      Eigen::Vector3d &t,
                      Eigen::Vector3d &v);

  void loadPointCloud(const std::string &cloud_path);

  Frame::Ptr localizeFrameOnHighFreq(double target_timestamp,
                                     std::vector<EventMsg>::iterator &iter,
                                     const std::vector<EventMsg>::iterator &end,
                                     Frame::Ptr last_frame,
                                     double time_interval);

  std::vector<Frame::Ptr> getAllFrames();

 public:
  pCloud cloud_;
  std::deque<EventMsg> event_deque_;
  std::deque<ImuMsg> imu_deque_;
  EventCamera::Ptr event_cam_;

  // for sync
  std::mutex data_mutex_;
  std::mutex viewer_mutex_;
  std::condition_variable con_;

  // for state
  bool is_system_start_;
  bool is_first_frame_;
  State state_;

  // frame
  double imu_t0_;
  Eigen::Vector3d acc0_;
  Eigen::Vector3d gyr0_;

  std::deque<Frame::Ptr> sliding_window_;
  std::vector<Frame::Ptr> history_frames_;

 public:
  // config param
  std::string cloud_path_;
  std::string result_path_;
  double start_time_;
  double timeSurface_decay_factor_;
  size_t imu_num_for_frame_;
  size_t window_size_;
  TimeSurface::VisualizationType viz_type_;
  size_t imu_num_for_init_frame_;
  size_t frame_num_for_init_;
  int init_freq_;

  Eigen::Matrix3d R0_;
  Eigen::Vector3d t0_;
  Eigen::Vector3d V0_;

 public:
  Optimizer::Ptr optimizer_;
  Viewer* viewer_;

  std::shared_ptr<std::thread> thread_process_;
  std::shared_ptr<std::thread> thread_viewer_;
};

}  // namespace CannyEVIT

#endif /* CANNYEVIT_SYSTEM */
