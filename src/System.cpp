//
// Created by mpl on 23-11-7.
//

#include <fstream>
#include <Eigen/Core>
#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>

#include "Frame.h"
#include "ImuIntegration.h"
#include "System.h"
#include "TimeSurface.h"
using namespace CannyEVIT;

System::System(const std::string &config_path)
    : cloud_(new std::vector<Point>()),
      event_cam_(nullptr),
      is_system_start_(true),
      is_first_frame_(true),
      is_step_by_step_(false),
      step_(false),
      state_(State::Init),
      imu_t0_(0.0),
      acc0_(Eigen::Vector3d::Zero()),
      gyr0_(Eigen::Vector3d::Zero()),
      imu_num_for_init_frame_(0),
      frame_num_for_init_(0),
      init_freq_(0),
      min_num_events_for_frame_(0),
      min_num_imu_for_frame_(0),
      last_frame_(nullptr),
      optimizer_(nullptr) {
  google::InitGoogleLogging("CannyEVIT");
  readParam(config_path);
  loadPointCloud(cloud_path_);

  event_cam_.reset(new EventCamera(config_path));
  optimizer_.reset(new Optimizer(config_path, event_cam_));
  viewer_ = new Viewer(config_path, this);

  TimeSurface::initTimeSurface(event_cam_);
  IntegrationBase::setCalib(config_path);
  thread_process_.reset(new std::thread(&System::process, this));
  thread_viewer_.reset(new std::thread(&Viewer::Run, viewer_));
}

System::~System() {
  is_system_start_ = false;
  thread_process_->join();
  result_dest_.close();
  delete viewer_;
}

void System::readParam(const std::string &config_path) {
  cv::FileStorage fs(config_path, cv::FileStorage::READ);
  LOG(INFO) << "-------------- SYSTEM --------------";
  LOG(INFO) << std::fixed;
  std::cout << std::fixed;
  if (!fs.isOpened())
    LOG(FATAL) << "Config Not Open !!!";
  else {
    LOG(INFO) << "open config file at " << config_path;

    if (fs["start_time"].isNone()) LOG(ERROR) << "config: start_time is not set";
    start_time_ = fs["start_time"];

    if (fs["cloud_path"].isNone()) LOG(ERROR) << "config: cloud_path is not set";
    cloud_path_ = fs["cloud_path"].string();

    if (fs["result_path"].isNone()) LOG(ERROR) << "config: result_path is not set";
    result_path_ = fs["result_path"].string();
    result_dest_.open(result_path_ + "/result.txt");
    result_dest_ << std::fixed << std::setprecision(5);

    if (fs["timeSurface_decay_factor"].isNone()) LOG(ERROR) << "config: timeSurface_decay_factor is not set";
    timeSurface_decay_factor_ = fs["timeSurface_decay_factor"];

    if (fs["timeSurface_truncate_threshold"].isNone())
      LOG(ERROR) << "config: timeSurface_truncate_threshold is not set";
    timeSurface_truncate_threshold_ = fs["timeSurface_truncate_threshold"];

    if (fs["imu_num_for_frame"].isNone()) LOG(ERROR) << "config: imu_num_for_frame is not set";
    imu_num_for_frame_ = static_cast<int>(fs["imu_num_for_frame"]);

    if (fs["window_size"].isNone()) LOG(ERROR) << "config: window_size is not set";
    window_size_ = static_cast<int>(fs["window_size"]);

    if (fs["R0"].isNone()) LOG(ERROR) << "config: R0 is not set";
    cv::Mat R0 = fs["R0"].mat();
    cv::cv2eigen(R0, R0_);

    if (fs["V0"].isNone()) LOG(ERROR) << "config: V0 is not set";
    cv::Mat V0 = fs["V0"].mat();
    cv::cv2eigen(V0, V0_);

    if (fs["t0"].isNone()) LOG(ERROR) << "config: t0 is not set";
    cv::Mat t0 = fs["t0"].mat();
    cv::cv2eigen(t0, t0_);

    if (fs["frame_num_for_init"].isNone()) LOG(ERROR) << "config: frame_num_for_init is not set";
    frame_num_for_init_ = static_cast<int>(fs["frame_num_for_init"]);

    if (fs["imu_num_for_init_frame"].isNone()) LOG(ERROR) << "config: imu_num_for_init_frame is not set";
    imu_num_for_init_frame_ = static_cast<int>(fs["imu_num_for_init_frame"]);

    if (fs["init_freq"].isNone()) LOG(ERROR) << "config: init_freq is not set";
    init_freq_ = fs["init_freq"];

    if (fs["min_num_events_for_frame"].isNone()) LOG(ERROR) << "config: min_num_events_for_frame is not set";
    min_num_events_for_frame_ = static_cast<int>(fs["min_num_events_for_frame"]);

    if (fs["min_num_imu_for_frame"].isNone()) LOG(ERROR) << "config: min_num_imu_for_frame is not set";
    min_num_imu_for_frame_ = static_cast<int>(fs["min_num_imu_for_frame"]);

    if (fs["field_type"].isNone()) LOG(ERROR) << "config: field_type is not set";
    if (fs["field_type"].string() == "distance_field")
      viz_type_ = TimeSurface::VisualizationType::CANNY;
    else if (fs["field_type"].string() == "inv_time_surface")
      viz_type_ = TimeSurface::VisualizationType::TIME_SURFACE;
    else
      LOG(ERROR) << "Unsupported field type";

    if (!fs["log_dir"].isNone()) FLAGS_log_dir = fs["log_dir"].string();
    if (!fs["log_color"].isNone()) FLAGS_colorlogtostderr = static_cast<int>(fs["log_color"]);
    if (!fs["log_also_to_stderr"].isNone()) FLAGS_alsologtostderr = static_cast<int>(fs["log_also_to_stderr"]);
  }

  LOG(INFO) << "start_time:" << std::fixed << start_time_;
  LOG(INFO) << "cloud_path:" << cloud_path_;
  LOG(INFO) << "result_path:" << result_path_;
  LOG(INFO) << "timeSurface_decay_factor:" << timeSurface_decay_factor_;
  LOG(INFO) << "timeSurface_truncate_threshold_:" << timeSurface_truncate_threshold_;
  LOG(INFO) << "imu_num_for_frame:" << imu_num_for_frame_;
  LOG(INFO) << "R0:\n" << R0_;
  LOG(INFO) << "t0:" << t0_.transpose();
  LOG(INFO) << "V0:" << V0_.transpose();
  fs.release();
}

void System::loadPointCloud(const std::string &cloud_path) {
  std::ifstream src;
  src.open(cloud_path);
  double x_position, y_position, z_position, x_gradient, y_gradient, z_gradient;
  while (src >> x_position >> y_position >> z_position >> x_gradient >> y_gradient >> z_gradient)
    cloud_->emplace_back(x_position, y_position, z_position, x_gradient, y_gradient, z_gradient);
  LOG(INFO) << "load " << cloud_->size() << "points";
  src.close();
}

void System::GrabEventMsg(double time_stamp, size_t x, size_t y, bool polarity) {
  {
    std::lock_guard<std::mutex> guard(data_mutex_);
    if (event_deque_.empty())
      event_deque_.emplace_back(time_stamp, x, y, polarity);
    else {
      event_deque_.emplace_back(time_stamp, x, y, polarity);
      int i = static_cast<int>(event_deque_.size()) - 2;
      while (i >= 0 && event_deque_[i].time_stamp_ > time_stamp) {
        event_deque_[i + 1] = event_deque_[i];
        i--;
      }
      if (event_deque_.size() - i > 200)
        LOG(WARNING) << "Event does not come in order, this may cause performance loss";
      event_deque_[i + 1] = EventMsg(time_stamp, x, y, polarity);
    }
  }
  if (event_deque_.size() >= min_num_events_for_frame_)
    con_.notify_one();
}

void System::GrabImuMsg(double time_stamp,
                        double accX,
                        double accY,
                        double accZ,
                        double gyrX,
                        double gyrY,
                        double gyrZ) {
  {
    std::lock_guard<std::mutex> guard(data_mutex_);
    if (time_stamp < start_time_ && std::fabs(time_stamp - start_time_) > 0.003) return;
    imu_deque_.emplace_back(time_stamp, Eigen::Vector3d(accX, accY, accZ), Eigen::Vector3d(gyrX, gyrY, gyrZ));
  }
  if (imu_deque_.size() >= min_num_imu_for_frame_)
    con_.notify_one();
}

void System::predictIMUPose(double dt,
                            const Eigen::Vector3d &acc0,
                            const Eigen::Vector3d &gyr0,
                            const Eigen::Vector3d &acc1,
                            const Eigen::Vector3d &gyr1,
                            const Eigen::Vector3d &ba,
                            const Eigen::Vector3d &bg,
                            Eigen::Quaterniond &Q,
                            Eigen::Vector3d &t,
                            Eigen::Vector3d &v) {
  Eigen::Vector3d un_acc_0 = Q * (acc0 - ba) - IntegrationBase::G;
  Eigen::Vector3d un_gyr = 0.5 * (gyr0 + gyr1) - bg;
  Q *= Utility::deltaQ(un_gyr * dt);
  Eigen::Vector3d un_acc_1 = Q * (acc1 - ba) - IntegrationBase::G;
  Eigen::Vector3d un_acc = 0.5 * (un_acc_0 + un_acc_1);
  t += dt * v + 0.5 * dt * dt * un_acc;
  v += dt * un_acc;
}

bool System::getMeasurement(FrameData &data) {

  if (event_deque_.empty() || imu_deque_.empty()) return false;
  if (event_deque_.size() < min_num_events_for_frame_) return false;
  if (imu_deque_.size() < min_num_imu_for_frame_) return false;

  // case 1
  double latest_time_stamp_for_min_event_num = event_deque_[min_num_events_for_frame_ - 1].time_stamp_;
  if (imu_deque_.back().time_stamp_ < latest_time_stamp_for_min_event_num) return false;

  // case 2
  double latest_time_stamp_for_min_imu_num = imu_deque_[min_num_imu_for_frame_ - 1].time_stamp_;
  if (event_deque_.back().time_stamp_ < latest_time_stamp_for_min_imu_num) return false;

  std::cout << "latest_time_stamp_for_min_event_num:" << latest_time_stamp_for_min_event_num << std::endl;
  std::cout << "latest_time_stamp_for_min_imu_num:" << latest_time_stamp_for_min_imu_num << std::endl;

  // indicate imu_idx, make sure enough imu observations
  size_t imu_idx = 0;
  while (imu_deque_[imu_idx].time_stamp_ < latest_time_stamp_for_min_event_num)
    imu_idx++;
  imu_idx = std::max(imu_idx, min_num_imu_for_frame_ - 1);

  // make data frame
  double frame_time_stamp = imu_deque_[imu_idx].time_stamp_;
  while (!event_deque_.empty() && event_deque_.front().time_stamp_ < frame_time_stamp) {
    data.eventData.emplace_back(event_deque_.front());
    event_deque_.pop_front();
  }
  for (size_t i = 0; i <= imu_idx; i++) {
    data.imuData.emplace_back(imu_deque_.front());
    imu_deque_.pop_front();
  }
  data.time_stamp_ = frame_time_stamp;

  return true;
}

void System::Track(CannyEVIT::FrameData &frame_data) {


  if (is_first_frame_) {
    last_frame_ = std::make_shared<Frame>(start_time_, nullptr, nullptr, event_cam_);
    last_frame_->set_Twb(R0_, t0_);
    last_frame_->set_velocity(V0_);
    imu_t0_ = frame_data.imuData.front().time_stamp_;
    acc0_ = frame_data.imuData.front().acc_;
    gyr0_ = frame_data.imuData.front().gyr_;
    is_first_frame_ = false;
  }

  Eigen::Quaterniond last_q = last_frame_->Qwb();
  Eigen::Vector3d last_t = last_frame_->twb(), last_V = last_frame_->velocity();

  // predict pose and make integration
  IntegrationBase::Ptr integration(new IntegrationBase(acc0_, gyr0_, Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero()));
  for (const auto &imu_msg : frame_data.imuData) {
    double dt = imu_msg.time_stamp_ - imu_t0_;
    predictIMUPose(dt,
                   acc0_,
                   gyr0_,
                   imu_msg.acc_,
                   imu_msg.gyr_,
                   Eigen::Vector3d::Zero(),
                   Eigen::Vector3d::Zero(),
                   last_q,
                   last_t,
                   last_V);
    integration->push_back(dt, imu_msg.acc_, imu_msg.gyr_);
    imu_t0_ = imu_msg.time_stamp_;
    acc0_ = imu_msg.acc_;
    gyr0_ = imu_msg.gyr_;
  }

  // make time surface
  for (const auto &event_msg : frame_data.eventData) {
    TimeSurface::updateHistoryEvent(event_msg);
  }
  std::cout << "--------\n";
  TimeSurface::Ptr time_surface_observation
      (new TimeSurface(frame_data.time_stamp_, timeSurface_decay_factor_, timeSurface_truncate_threshold_));

  // make frame
  Frame::Ptr cur_frame(new Frame(frame_data.time_stamp_, time_surface_observation, integration, event_cam_));
  cur_frame->set_Twb(last_q, last_t);
  cur_frame->set_velocity(last_V);
  cur_frame->set_Ba(last_frame_->Ba());
  cur_frame->set_Bg(last_frame_->Bg());
  cur_frame->time_surface_observation_->drawCloud(cloud_, cur_frame->Twc(), "before optimize");

  switch (state_) {
    case Init: {

      optimizer_->OptimizeEventProblemCeres(cloud_, cur_frame);
      cur_frame->optToState();
      cur_frame->time_surface_observation_->drawCloud(cloud_, cur_frame->Twc(), "init");
      initial_vector_.push_back(cur_frame);

      if (initial_vector_.size() == frame_num_for_init_) {
        optimizer_->OptimizeVelovityBias(initial_vector_);

        for (auto iter = initial_vector_.rbegin(); iter != initial_vector_.rend(); iter++) {
          (*iter)->optToState();
          (*iter)->integration_->repropagate((*iter)->Ba(), (*iter)->Bg());
          sliding_window_.push_front(*iter);
          if (sliding_window_.size() == window_size_) break;
        }
        state_ = Tracking;
      }

      cv::waitKey(0);
      break;
    }

    case Tracking: {
      sliding_window_.push_back(cur_frame);
      cur_frame->time_surface_observation_->drawCloud(cloud_, last_frame_->Twc(), "last_frame");
      cur_frame->time_surface_observation_->drawCloud(cloud_, cur_frame->Twc(), "pred_frame");

//      optimizer_->OptimizeSlidingWindowProblemCeresBatch(cloud_, sliding_window_);
      optimizer_->OptimizeSlidingWindowProblemCeres(cloud_, sliding_window_);

      // for visualization
      {
        std::lock_guard<std::mutex> guard(viewer_mutex_);
        for (size_t i = 0; i < sliding_window_.size(); i++)
          sliding_window_[i]->optToState();
        if (sliding_window_.size() > window_size_) {
          sliding_window_.front()->set_active(false);
          sliding_window_.front()->time_surface_observation_.reset(); // TODO: try to find another way to reduce memory burden
          history_frames_.emplace_back(std::move(sliding_window_.front()));

          double time_stamp = history_frames_.back()->time_stamp_;
          const Eigen::Quaterniond &Qwb = history_frames_.back()->Qwb();
          const Eigen::Vector3d &twb = history_frames_.back()->twb();
          result_dest_ << time_stamp << " " << twb.x() << " " << twb.y() << " " << twb.z()
                       << " " << Qwb.x() << " " << Qwb.y() << " " << Qwb.z() << " " << Qwb.w() << std::endl;

          sliding_window_.pop_front();
        }
      }

      for (size_t i = 0; i < sliding_window_.size(); i++)
        sliding_window_[i]->integration_->repropagate(sliding_window_[i]->Ba(), sliding_window_[i]->Bg());

      sliding_window_.back()->time_surface_observation_->drawCloud(
          cloud_,
          sliding_window_.back()->Twc(),
          "last_frame_reprojection_in_neutral",
          TimeSurface::PolarType::NEUTRAL,
          viz_type_);
      cv::waitKey(10);

      break;
    }

  }
  last_frame_ = cur_frame;
}

void System::Track(Frame::Ptr frame) {

}

void System::process() {
  while (is_system_start_) {
    std::unique_lock<std::mutex> lk(data_mutex_);
    FrameData data;
    con_.wait(lk, [&data, this] { return getMeasurement(data); });
    lk.unlock();

    std::cout << data.time_stamp_ << std::endl;
    std::cout << data.eventData.size() << std::endl;
    std::cout << data.eventData.back().time_stamp_ << std::endl;
    std::cout << data.imuData.size() << std::endl;
    std::cout << data.imuData.back().time_stamp_ << std::endl;
    std::cout << "----------------" << std::endl;
    Track(data);

    if (is_step_by_step_) {
      std::cout << "Tracking: Waiting to the next step" << std::endl;
      while (!step_ && is_step_by_step_)
        usleep(500);
      step_ = false;
    }

  }
}

void System::setStepByStep(bool val) {
  is_step_by_step_ = val;
}

void System::Step() {
  step_ = true;
}

std::vector<Frame::Ptr> System::getAllFrames() {
  std::lock_guard<std::mutex> guard(viewer_mutex_);
  std::vector<Frame::Ptr> frames(history_frames_.begin(), history_frames_.end());
  frames.insert(frames.end(), sliding_window_.begin(), sliding_window_.end());
  return frames;
}

Frame::Ptr System::localizeFrameOnHighFreq(double target_timestamp,
                                           std::vector<EventMsg>::iterator &iter,
                                           const std::vector<EventMsg>::iterator &end,
                                           Frame::Ptr last_frame,
                                           double time_interval) {
  while (last_frame->time_stamp_ < target_timestamp && fabs(last_frame->time_stamp_ - target_timestamp) > 1e-4) {
    double cur_frame_timestamp = std::min(last_frame->time_stamp_ + time_interval, target_timestamp);
    while (iter->time_stamp_ < cur_frame_timestamp && iter != end) {
      TimeSurface::updateHistoryEvent(*iter);
      iter++;
    }
    TimeSurface::Ptr cur_time_surface_observation
        (new TimeSurface(cur_frame_timestamp, timeSurface_decay_factor_, timeSurface_truncate_threshold_));
    Frame::Ptr cur_frame(new Frame(cur_frame_timestamp, cur_time_surface_observation, nullptr, event_cam_));
    cur_frame->set_velocity(last_frame->velocity());
    cur_frame->set_Twb(last_frame->Twb());
    optimizer_->OptimizeEventProblemCeres(cloud_, cur_frame);
//    optimizer_->OptimizeEventProblemEigen(cloud_, cur_frame);
    cur_frame->optToState();
    last_frame = cur_frame;
  }
  return last_frame;
}