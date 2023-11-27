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
cloud_(new std::vector<Point>()), event_cam_(nullptr), is_system_start_(true), is_first_frame_(true),
state_(State::Init), imu_t0_(0.0), acc0_(Eigen::Vector3d::Zero()), gyr0_(Eigen::Vector3d::Zero()),
imu_num_for_init_frame_(0), frame_num_for_init_(0), init_freq_(0), optimizer_(nullptr)
{
    readParam(config_path);
    loadPointCloud(cloud_path_);

    event_cam_.reset(new EventCamera(config_path));
    optimizer_.reset(new Optimizer(config_path, event_cam_));
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

        if (fs["frame_num_for_init"].isNone())
            LOG(ERROR) << "config: frame_num_for_init is not set";
        frame_num_for_init_ = fs["frame_num_for_init"];

        if (fs["imu_num_for_init_frame"].isNone())
            LOG(ERROR) << "config: imu_num_for_init_frame is not set";
        imu_num_for_init_frame_ = fs["imu_num_for_init_frame"];

        if (fs["init_freq"].isNone())
            LOG(ERROR) << "config: init_freq is not set";
        init_freq_ = fs["init_freq"];
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
    if (event_deque_.empty())
        event_deque_.emplace_back(time_stamp, x, y, polarity);
    else
    {
        event_deque_.emplace_back(time_stamp, x, y, polarity);
        int i = static_cast<int>(event_deque_.size()) - 2;
        while(i >= 0 && event_deque_[i].time_stamp_ > time_stamp)
        {
            event_deque_[i+1] = event_deque_[i];
            i--;
        }
        if (event_deque_.size() - i > 200)
            LOG(WARNING) << "Event does not come in order, this may cause performance loss";
        event_deque_[i+1] = EventMsg(time_stamp, x, y, polarity);
    }
}

void System::GrabImuMsg(double time_stamp, double accX, double accY, double accZ, double gyrX, double gyrY, double gyrZ)
{
    {
        std::lock_guard<std::mutex> guard(data_mutex_);
        if (time_stamp < start_time_ && std::fabs(time_stamp - start_time_) > 0.0001)
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
    switch (state_) {
        case Init:
        {
            int imu_num_for_init = imu_num_for_init_frame_ * frame_num_for_init_;
            if (event_deque_.empty() || imu_deque_.size() < imu_num_for_init)
                return false;
            if (event_deque_.back().time_stamp_ < imu_deque_[imu_num_for_init-1].time_stamp_)
                return false;

            for (int i = 0; i < imu_num_for_init; i++)
            {
                data.imuData.emplace_back(imu_deque_.front());
                imu_deque_.pop_front();
            }
            while (event_deque_.front().time_stamp_ < data.imuData.back().time_stamp_)
            {
                data.eventData.emplace_back(event_deque_.front());
                event_deque_.pop_front();
            }
            data.time_stamp_ = data.imuData.back().time_stamp_;
            break;
        }
        case Tracking:
        {
            if (imu_deque_.size() <= imu_num_for_frame_ || event_deque_.empty() ||
                event_deque_.back().time_stamp_ < imu_deque_[imu_num_for_frame_ - 1].time_stamp_)
                return false;

            for (int i = 0; i < imu_num_for_frame_; i++)
            {
                data.imuData.emplace_back(imu_deque_.front());
                imu_deque_.pop_front();
            }
            while (event_deque_.front().time_stamp_ < data.imuData.back().time_stamp_)
            {
                data.eventData.emplace_back(event_deque_.front());
                event_deque_.pop_front();
            }
            data.time_stamp_ = imu_deque_.back().time_stamp_;
            break;
        }
    }
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

        switch (state_){

            case Init:
            {
                std::vector<Frame::Ptr> initial_list;
                LOG(INFO) << "event back stamp:" << data.eventData.back().time_stamp_ ;
                LOG(INFO) << "imu back stamp:" << data.imuData.back().time_stamp_ ;

                // localize start frame
                auto event_it = data.eventData.begin();
                auto event_end = data.eventData.end();
                while(event_it->time_stamp_ < start_time_)
                {
                    TimeSurface::updateHistoryEvent(*event_it);
                    event_it++;
                }
                TimeSurface::Ptr first_time_surface_observation(new TimeSurface(start_time_, timeSurface_decay_factor_));
                Frame::Ptr first_frame(new Frame(first_time_surface_observation, nullptr, event_cam_));
                first_frame->set_velocity(V0_);
                first_frame->set_Twb(R0_, t0_);
                first_frame->time_surface_observation_->drawCloud(cloud_, first_frame->Twc(), "before");
                optimizer_->OptimizeEventProblemCeres(cloud_, first_frame);
                Frame::Ptr last_frame = first_frame;
                first_frame->time_surface_observation_->drawCloud(cloud_, first_frame->Twc(), "first frame");
                cv::waitKey(0);

                double time_interval = 1.0 / static_cast<double>(init_freq_);
                // localize first frame, first initial frame does not need integration
                imu_t0_ = data.imuData.front().time_stamp_;
                acc0_ = data.imuData.front().acc_;
                gyr0_ = data.imuData.front().gyr_;

                double target_frame_timestamp = data.imuData.front().time_stamp_;
                last_frame = localizeFrameOnHighFreq(target_frame_timestamp, event_it, event_end, last_frame, time_interval);
                last_frame->time_surface_observation_->drawCloud(cloud_, last_frame->Twc(), "init");
                cv::waitKey(0);
                initial_list.push_back(last_frame);

                // localize on each frame
                for (int i = 0; i < frame_num_for_init_; i++)
                {

                    // integrate imu firstly
                    IntegrationBase::Ptr target_integration(new IntegrationBase(acc0_, gyr0_, Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero()));
                    for (int j = 0; j < imu_num_for_init_frame_; j++)
                    {
                        int index = i * imu_num_for_init_frame_ + j;
                        std::cout << index << std::endl;
                        double dt = data.imuData[index].time_stamp_ - imu_t0_;
                        target_integration->push_back(dt, data.imuData[index].acc_, data.imuData[index].gyr_);
                        imu_t0_ = data.imuData[index].time_stamp_;
                        acc0_ = data.imuData[index].acc_;
                        gyr0_ = data.imuData[index].gyr_;
                    }

                    // localize one each frame
                    double target_frame_timestamp = imu_t0_;
                    last_frame = localizeFrameOnHighFreq(target_frame_timestamp, event_it, event_end, last_frame, time_interval);
                    last_frame->time_surface_observation_->drawCloud(cloud_, last_frame->Twc(), "init");
                    cv::waitKey(0);
                    last_frame->set_integration(target_integration);
                    initial_list.push_back(last_frame);
                }

                // TODO: MAY REPLACE AS LINEAR SOLVER
//                optimizer_->OptimizeVelovityBias(initial_list);
                optimizer_->initVelocityBias(initial_list);
                for (auto iter = initial_list.rbegin(); iter != initial_list.rend(); iter++)
                {
                    (*iter)->integration_->repropagate((*iter)->Ba(), (*iter)->Bg());
                    sliding_window_.push_front(*iter);
                    if (sliding_window_.size() == window_size_)
                        break;
                }
                state_ = Tracking;
                break;
            }

            case Tracking:
            {
                auto event_it = data.eventData.begin();
                // make current time surface
                while (event_it != data.eventData.end())
                {
                    TimeSurface::updateHistoryEvent(*event_it);
                    event_it++;
                }
                double current_frame_time = data.imuData.back().time_stamp_;
                TimeSurface::Ptr current_time_surface_observation(new TimeSurface(current_frame_time, timeSurface_decay_factor_));


                Frame::Ptr last_frame = sliding_window_.back();
                Eigen::Quaterniond last_q = last_frame->Qwb();
                Eigen::Vector3d last_t = last_frame->twb();
                Eigen::Vector3d last_v = last_frame->velocity();
                Eigen::Vector3d last_ba = last_frame->Ba();
                Eigen::Vector3d last_bg = last_frame->Bg();

                // make current integration and predict pose
                IntegrationBase::Ptr current_integration(new IntegrationBase(acc0_, gyr0_, last_ba, last_bg));

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
                current_frame->set_Ba(last_ba);
                current_frame->set_Bg(last_bg);

        //        LOG(INFO) << "current frame:" << current_frame_time;
        //        LOG(INFO) << "Last pose:\n" << current_frame->Twb();
                current_frame->time_surface_observation_->drawCloud(cloud_, last_frame->Twc(), "last_frame");
                current_frame->time_surface_observation_->drawCloud(cloud_, current_frame->Twc(), "pred_frame");
//                // Optimize here
//                optimizer_->OptimizeEventProblemCeres(cloud_, current_frame);
//                current_frame->time_surface_observation_->drawCloud(cloud_, current_frame->Twc(), "after_frame");
//                LOG(INFO) << "Current pose:\n" << current_frame->Twb();

                sliding_window_.push_back(current_frame);
                optimizer_->OptimizeSlidingWindowProblemCeres(cloud_, sliding_window_);

                for (int i = 0; i < sliding_window_.size(); i++)
                {
                    sliding_window_[i]->integration_->repropagate(sliding_window_[i]->Ba(), sliding_window_[i]->Bg());
                    sliding_window_[i]->time_surface_observation_->drawCloud(cloud_, sliding_window_[i]->Twc(), std::to_string(i));
                }
                cv::waitKey(10);

                if (sliding_window_.size() > window_size_)
                {
    //            history_frames_.emplace_back(std::move(sliding_window_.front()));
                    sliding_window_.pop_front();
                }
                break;
            }
        }



    }
}

Frame::Ptr System::localizeFrameOnHighFreq(double target_timestamp,
                                           std::vector<EventMsg>::iterator& iter,
                                           const std::vector<EventMsg>::iterator& end,
                                           Frame::Ptr last_frame, double time_interval)
{

    while(last_frame->time_stamp_ < target_timestamp &&
          fabs(last_frame->time_stamp_ - target_timestamp) > 1e-4)
    {
        double cur_frame_timestamp = std::min(last_frame->time_stamp_ + time_interval, target_timestamp);
        while(iter->time_stamp_ < cur_frame_timestamp && iter != end)
        {
            TimeSurface::updateHistoryEvent(*iter);
            iter++;
        }
        TimeSurface::Ptr cur_time_surface_observation(new TimeSurface(cur_frame_timestamp, timeSurface_decay_factor_));
        Frame::Ptr cur_frame(new Frame(cur_time_surface_observation, nullptr, event_cam_));
        cur_frame->set_velocity(last_frame->velocity());
        cur_frame->set_Twb(last_frame->Twb());
        optimizer_->OptimizeEventProblemCeres(cloud_, cur_frame);
        last_frame = cur_frame;
    }

    return last_frame;

}