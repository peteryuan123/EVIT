//
// Created by mpl on 23-11-17.
//
#include "EigenOptimization/Problem/EventProblem.h"
#include <random>

using namespace CannyEVIT;

EventProblemLM::EventProblemLM(EventProblemConfig config)
:GenericFunctor<double>(6, 0), config_(config),cloud_(nullptr), frame_(nullptr), point_num_(0),
 patch_size_(config.patch_size_X_ * config.patch_size_Y_), batch_num_(0),
 residual_start_index_(0), residual_end_index_(0)
{}

void EventProblemLM::setProblem(Frame::Ptr frame, CannyEVIT::pCloud cloud)
{
    frame_ = frame;
    cloud_ = cloud;
    point_num_ = std::min(config_.MAX_REGISTRATION_POINTS_, cloud->size());

    std::set<size_t> set_samples;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dist(0, cloud->size());
    while (set_samples.size() < point_num_)
        set_samples.insert(dist(gen));

    for (auto iter = set_samples.begin(); iter != set_samples.end(); iter++)
        residuals_info_.emplace_back(new ResidualEventInfo(cloud_->at(*iter), patch_size_));

    resetNumberValues(point_num_ * patch_size_);
    residual_end_index_ = m_values;
}


int EventProblemLM::operator()(const Eigen::Matrix<double, 6, 1> &x, Eigen::VectorXd &fvec) const
{
    Eigen::Vector3d twb = frame_->twb();
    Eigen::Quaterniond Qwb = frame_->Qwb();
    addPerturbation(Qwb, twb, x);
    for (size_t i = residual_start_index_; i < residual_end_index_; i++)
    {
        auto& residual_info = residuals_info_[i];
        Eigen::VectorXd residual = frame_->time_surface_observation_->evaluate(residual_info->p_, Qwb, twb,config_.patch_size_X_,
                                                                               config_.patch_size_Y_);
        for (int j = 0; j < residual.size(); j++)
        {
            residual_info->irls_weight_[j] = 1.0;
            if (residual[j] > config_.huber_threshold_)
                residual_info->irls_weight_[j] = config_.huber_threshold_ / residual[j];
            fvec[i*patch_size_ + j] = residual_info->irls_weight_[j] * residual[j];
        }
    }
    return 0;
}


int EventProblemLM::df(const Eigen::Matrix<double, 6, 1> &x, Eigen::MatrixXd &fjac) const
{
    if (x != Eigen::Matrix<double, 6, 1>::Zero())
    {
        std::cerr << "The Jacobian is not evaluated at Zero !!!!!!!!!!!!!";
        exit(-1);
    }
    fjac.resize(m_values, 6);

    for (size_t i = 0; i < point_num_; i++)
    {

    }

    return 0;
}

void EventProblemLM::addPerturbation(Eigen::Quaterniond &Qwb, Eigen::Vector3d &twb,
                                     const Eigen::Matrix<double, 6, 1> &dx) const
{
    twb = twb + dx.block<3, 1>(0, 0);
    Eigen::Quaterniond dQ = Eigen::Quaterniond(1, dx[3]/2.0, dx[4]/2.0, dx[5]/2.0);
    Qwb = Qwb * dQ;
    Qwb.normalize();
}


void EventProblemLM::addMotionUpdate(const Eigen::Matrix<double, 6, 1> &dx)
{
    Eigen::Vector3d new_twb = frame_->twb();
    Eigen::Quaterniond new_Qwb = frame_->Qwb();
    addPerturbation(new_Qwb, new_twb, dx);
    frame_->set_twb(new_twb);
    frame_->set_Rwb(new_Qwb);
}