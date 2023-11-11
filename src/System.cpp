//
// Created by mpl on 23-11-7.
//

#include "System.h"
#include "easylogging++.h"
#include "TimeSurface.h"
#include "ImuIntegration.h"
#include "Frame.h"
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
INITIALIZE_EASYLOGGINGPP
using namespace CannyEVIT;

System::System(const std::string &config_path):
cloud_(new std::vector<Point>()), event_cam_(nullptr), is_system_start_(true),
is_first_frame_(true), imu_t0_(0.0), acc0_(Eigen::Vector3d::Zero()), gyr0_(Eigen::Vector3d::Zero())
{
    readParam(config_path);
    loadPointCloud(cloud_path_);

    event_cam_.reset(new EventCamera(config_path));
    TimeSurface::initTimeSurface(event_cam_);
    IntegrationBase::setCalib(config_path);
    thread_process_.reset(new std::thread(&System::process, this));
}

System::~System()
{
    is_system_start_ = false;
    thread_process_->join();
}

void System::readParam(const std::string &config_path)
{
    el::Configurations conf("/home/mpl/code/EVIT_NEW/config/logConfig.conf");
    el::Loggers::reconfigureLogger("default", conf);
    el::Loggers::reconfigureAllLoggers(conf);
    el::Loggers::addFlag(el::LoggingFlag::NewLineForContainer);
    el::Loggers::addFlag(el::LoggingFlag::AutoSpacing);
    el::Loggers::addFlag(el::LoggingFlag::ColoredTerminalOutput);

    cv::FileStorage fs(config_path, cv::FileStorage::READ);
    LOG(INFO) << "-------------- SYSTEM --------------";
    LOG(INFO) << std::fixed;
    std::cout << std::fixed;
    if (!fs.isOpened())
        LOG(FATAL) << "Config Not Open !!!";
    else
    {
        LOG(INFO) << "open config file at " << config_path;

        if (fs["start_time"].isNone())
            LOG(ERROR) << "config: start_time is not set";
        start_time_ = fs["start_time"];

        if (fs["cloud_path"].isNone())
            LOG(ERROR) << "config: cloud_path is not set";
        cloud_path_ = fs["cloud_path"].string();

        if (fs["result_path"].isNone())
            LOG(ERROR) << "config: result_path is not set";
        result_path_ = fs["result_path"].string();

        if (fs["timeSurface_decay_factor"].isNone())
            LOG(ERROR) << "config: timeSurface_decay_factor is not set";
        timeSurface_decay_factor_ = fs["timeSurface_decay_factor"];

        if (fs["imu_num_for_frame"].isNone())
            LOG(ERROR) << "config: imu_num_for_frame is not set";
        imu_num_for_frame_ = fs["imu_num_for_frame"];

        if (fs["window_size"].isNone())
            LOG(ERROR) << "config: window_size is not set";
        window_size_ = fs["window_size"];

        if (fs["R0"].isNone())
            LOG(ERROR) << "config: R0 is not set";
        cv::Mat R0 = fs["R0"].mat();
        cv::cv2eigen(R0, R0_);

        if (fs["V0"].isNone())
            LOG(ERROR) << "config: V0 is not set";
        cv::Mat V0 = fs["V0"].mat();
        cv::cv2eigen(V0, V0_);

        if (fs["t0"].isNone())
            LOG(ERROR) << "config: t0 is not set";
        cv::Mat t0 = fs["t0"].mat();
        cv::cv2eigen(t0, t0_);

    }

    LOG(INFO) << "start_time:" << start_time_;
    LOG(INFO) << "cloud_path:" << cloud_path_;
    LOG(INFO) << "result_path:" << result_path_;
    LOG(INFO) << "timeSurface_decay_factor:" << timeSurface_decay_factor_;
    LOG(INFO) << "imu_num_for_frame:" << imu_num_for_frame_;
    LOG(INFO) << "R0:\n" << R0_;
    LOG(INFO) << "t0:" << t0_.transpose();
    LOG(INFO) << "V0:" << V0_.transpose();
    fs.release();
}

void System::loadPointCloud(const std::string &cloud_path)
{
    std::ifstream src;
    src.open(cloud_path);
    double x_position, y_position, z_position, x_normal, y_normal, z_normal;
    while(src >> x_position >> y_position >> z_position >> x_normal >> y_normal >> z_normal)
        cloud_->emplace_back(x_position, y_position, z_position, x_normal, y_normal, z_normal);
    LOG(INFO) << "load " << cloud_->size() << "points";
    src.close();
}

void System::GrabEventMsg(double time_stamp, size_t x, size_t y, bool polarity)
{
    std::lock_guard<std::mutex> guard(data_mutex_);
    if (event_deque_.size() == 0)
        event_deque_.emplace_back(time_stamp, x, y, polarity);
    else
    {
        event_deque_.emplace_back(time_stamp, x, y, polarity);
        int i = event_deque_.size() - 2;
        while(i >= 0 && event_deque_[i].time_stamp_ > time_stamp)
        {
            event_deque_[i+1] = event_deque_[i];
            i--;
        }
        if (event_deque_.size() - i > 200)
            LOG(WARNING) << "Event does not come in order, this may cause performace loss";
        event_deque_[i+1] = EventMsg(time_stamp, x, y, polarity);
    }
}

void System::GrabImuMsg(double time_stamp, double accX, double accY, double accZ, double gyrX, double gyrY, double gyrZ)
{
    {
        std::lock_guard<std::mutex> guard(data_mutex_);
        if (time_stamp < start_time_ && std::abs(time_stamp - start_time_) > 0.001)
            return;
        imu_deque_.emplace_back(time_stamp, Eigen::Vector3d(accX, accY, accZ),  Eigen::Vector3d(gyrX, gyrY, gyrZ));
    }
    con_.notify_one();
}

void System::predictIMUPose(double dt, const Eigen::Vector3d &acc0, const Eigen::Vector3d &gyr0,
                            const Eigen::Vector3d &acc1, const Eigen::Vector3d &gyr1, const Eigen::Vector3d &ba,
                            const Eigen::Vector3d &bg,  Eigen::Quaterniond &Q,  Eigen::Vector3d &t, Eigen::Vector3d &v)
{
    Eigen::Vector3d un_acc_0 = Q * (acc0 - ba) - IntegrationBase::G;
    Eigen::Vector3d un_gyr = 0.5 * (gyr0 + gyr1) - bg;
    Q *= Utility::deltaQ(un_gyr * dt);
    Eigen::Vector3d un_acc_1 = Q * (acc1 - ba) - IntegrationBase::G;
    Eigen::Vector3d un_acc = 0.5 * (un_acc_0 + un_acc_1);
    t += dt * v + 0.5 * dt * dt * un_acc;
    v += dt * un_acc;
}

bool System::getMeasurement(FrameData& data)
{
    if (imu_deque_.size() <= imu_num_for_frame_ || event_deque_.size() == 0 ||
        event_deque_.back().time_stamp_ < imu_deque_[imu_num_for_frame_ - 1].time_stamp_)
        return false;

    for (int i = 0; i < imu_num_for_frame_; i++)
    {
        data.imuData.emplace_back(std::move(imu_deque_.front()));
        imu_deque_.pop_front();
    }

    while (event_deque_.front().time_stamp_ < data.imuData.back().time_stamp_)
    {
        data.eventData.emplace_back(std::move(event_deque_.front()));
        event_deque_.pop_front();
    }

    data.time_stamp_ = imu_deque_.back().time_stamp_;
    return true;
}

void System::process()
{
    while(is_system_start_)
    {
        std::unique_lock<std::mutex> lk(data_mutex_);
        FrameData data;
        con_.wait(lk, [&data, this] {return getMeasurement(data);});
        lk.unlock();

        auto event_it = data.eventData.begin();

        // for first frame
        if (is_first_frame_)
        {
            imu_t0_ = data.imuData.front().time_stamp_;
            acc0_ = data.imuData.front().acc_;
            gyr0_ = data.imuData.front().gyr_;

            // make first frame
            LOG(INFO) << data.eventData.front().time_stamp_;
            while (event_it->time_stamp_ < start_time_ && event_it != data.eventData.end()) {
                TimeSurface::updateHistoryEvent(*event_it);
                event_it++;
            }
            IntegrationBase::Ptr first_dummy_integration_(
                    new IntegrationBase(acc0_, gyr0_, Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero()));
            TimeSurface::Ptr time_surface_observation_for_first_frame(
                    new TimeSurface(start_time_, timeSurface_decay_factor_));
            Frame::Ptr first_frame(
                    new Frame(time_surface_observation_for_first_frame, first_dummy_integration_, event_cam_));

            // set first pose
            first_frame->set_Twb(R0_, t0_);
            first_frame->set_velocity(V0_);

            sliding_window_.push_back(first_frame);
            is_first_frame_ = false;
        }

        // make current time surface
        while (event_it != data.eventData.end())
        {
            TimeSurface::updateHistoryEvent(*event_it);
            event_it++;
        }
        double current_frame_time = data.imuData.back().time_stamp_;
        TimeSurface::Ptr current_time_surface_observation(new TimeSurface(current_frame_time, timeSurface_decay_factor_));

        // make current integration and predict pose
        IntegrationBase::Ptr current_integration(new IntegrationBase(acc0_, gyr0_, Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero()));
        Frame::Ptr last_frame = sliding_window_.back();
        Eigen::Quaterniond last_q = last_frame->Qwb();
        Eigen::Vector3d last_t = last_frame->twb();
        Eigen::Vector3d last_v = last_frame->velocity();
        for (auto imu_iter = data.imuData.begin(); imu_iter != data.imuData.end(); imu_iter++)
        {
            double dt = imu_iter->time_stamp_ - imu_t0_;
            current_integration->push_back(dt, imu_iter->acc_, imu_iter->gyr_);

            predictIMUPose(dt, acc0_, gyr0_, imu_iter->acc_, imu_iter->gyr_, last_frame->Ba(), last_frame->Bg(),
                           last_q, last_t, last_v);

            imu_t0_ = imu_iter->time_stamp_;
            acc0_ = imu_iter->acc_;
            gyr0_ = imu_iter->gyr_;
        }
        Frame::Ptr current_frame(new Frame(current_time_surface_observation, current_integration, event_cam_));
        current_frame->set_Twb(last_q, last_t);
        current_frame->set_velocity(last_v);

        LOG(INFO) << "current frame:" << current_frame_time;
        current_frame->time_surface_observation_->drawCloud(cloud_, current_frame->Twc(), "current_frame");
        cv::waitKey(10);

        sliding_window_.push_back(current_frame);
        if (sliding_window_.size() > window_size_)
        {
            history_frames_.emplace_back(std::move(sliding_window_.front()));
            sliding_window_.pop_front();
        }
    }
}