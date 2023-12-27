//
// Created by mpl on 23-11-17.
//
#include <random>
#include <thread>
#include <glog/logging.h>
#include "EigenOptimization/Problem/EventProblem.h"

using namespace CannyEVIT;

EventProblemLM::EventProblemLM(const EventProblemConfig &config)
    : GenericFunctor<double>(6, 0),
      cloud_(nullptr),
      frame_(nullptr),
      point_num_(0),
      config_(config),
      opt_Qwb_(Eigen::Quaterniond::Identity()),
      opt_twb_(Eigen::Vector3d::Zero()),
      cur_batch_start_index_(0) {
  indices_for_cur_batch_.reserve(config_.batch_size_);
}

void EventProblemLM::setProblem(Frame::Ptr frame, const CannyEVIT::pCloud &cloud) {
  frame_ = std::move(frame);
  cloud_ = cloud;
  point_num_ = std::min(config_.max_registration_points_, cloud->size());
  opt_Qwb_ = frame_->Qwb();
  opt_twb_ = frame_->twb();

  std::set<size_t> set_samples;
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<size_t> dist(0, cloud->size() - 1);
  while (set_samples.size() < point_num_) set_samples.insert(dist(gen));

//TODO: fix polarity prediction
  for (auto iter = set_samples.begin(); iter != set_samples.end(); iter++)
    event_residuals_.emplace_back(cloud_->at(*iter),
                                  frame_->time_surface_observation_,
                                  config_.patch_size_x_,
                                  config_.patch_size_y_,
                                  TimeSurface::PolarType::NEUTRAL,
                                  config_.field_type_);

  resetNumberValues(point_num_ * config_.patch_size_x_ * config_.patch_size_y_);
}

void EventProblemLM::nextBatch() {
  indices_for_cur_batch_.clear();
  indices_for_cur_batch_.reserve(config_.batch_size_);
  while (indices_for_cur_batch_.size() < config_.batch_size_) {
    indices_for_cur_batch_.emplace_back(cur_batch_start_index_);
    cur_batch_start_index_++;
    if (cur_batch_start_index_ >= point_num_)
      cur_batch_start_index_ = 0;
  }

  resetNumberValues(config_.batch_size_ * config_.patch_size_x_ * config_.patch_size_y_);
}

int EventProblemLM::operator()(const Eigen::Matrix<double, 6, 1> &x, Eigen::VectorXd &fvec) const {
  fvec.setZero();

  Eigen::Quaterniond Qwb = opt_Qwb_;
  Eigen::Vector3d twb = opt_twb_;
  addPerturbation(x, Qwb, twb);

  std::vector<Job> jobs(config_.thread_num_);
  for (size_t i = 0; i < config_.thread_num_; i++) {
    jobs[i].pvRI = const_cast<ResidualEventItems *>(&event_residuals_);
    jobs[i].Qwb = const_cast<Eigen::Quaterniond *>(&Qwb);
    jobs[i].twb = const_cast<Eigen::Vector3d *>(&twb);
    jobs[i].threadID = i;
  }

  std::vector<std::thread> threads;
  for (size_t i = 0; i < config_.thread_num_; i++)
    threads.emplace_back(std::bind(&EventProblemLM::residualThread, this, jobs[i], std::ref(fvec)));
  for (auto &thread : threads)
    if (thread.joinable())
      thread.join();

  return 0;
}

int EventProblemLM::df(const Eigen::Matrix<double, 6, 1> &x, Eigen::MatrixXd &fjac) const {
  if (x != Eigen::Matrix<double, 6, 1>::Zero()) {
    std::cerr << "The Jacobian is not evaluated at Zero !!!!!!!!!!!!!";
    exit(-1);
  }
  fjac.resize(m_values, 6);

  Eigen::Quaterniond Qwb = opt_Qwb_;
  Eigen::Vector3d twb = opt_twb_;
  addPerturbation(x, Qwb, twb);

  std::vector<Job> jobs(config_.thread_num_);
  for (size_t i = 0; i < config_.thread_num_; i++) {
    jobs[i].pvRI = const_cast<ResidualEventItems *>(&event_residuals_);
    jobs[i].Qwb = const_cast<Eigen::Quaterniond *>(&Qwb);
    jobs[i].twb = const_cast<Eigen::Vector3d *>(&twb);
    jobs[i].threadID = i;
  }

  std::vector<std::thread> threads;
  for (size_t i = 0; i < config_.thread_num_; i++)
    threads.emplace_back(std::bind(&EventProblemLM::JacobianThread, this, jobs[i], std::ref(fjac)));
  for (auto &thread : threads)
    if (thread.joinable())
      thread.join();

  return 0;
}

void EventProblemLM::residualThread(Job &job, Eigen::VectorXd &fvec) const {
  ResidualEventItems &vRI = *job.pvRI;
  const Eigen::Quaterniond &Qwb = *job.Qwb;
  const Eigen::Vector3d &twb = *job.twb;
  size_t threadID = job.threadID;

  size_t patch_size = config_.patch_size_x_ * config_.patch_size_y_;
  switch (config_.loss_function_type_) {
    case L2: {
      for (size_t i = threadID; i < indices_for_cur_batch_.size(); i += config_.thread_num_) {
        size_t index = indices_for_cur_batch_[i];
        vRI[index].computeResidual(Qwb, twb);
        ResidualEventItem &ri = const_cast<ResidualEventItem &>(vRI[index]);
        fvec.segment(i * patch_size, patch_size) = ri.residuals_; // / sqrt(var);
      }
      break;
    }

    case Huber: {
      for (size_t i = threadID; i < indices_for_cur_batch_.size(); i += config_.thread_num_) {
        size_t index = indices_for_cur_batch_[i];
        vRI[index].computeResidual(Qwb, twb);
        ResidualEventItem &ri = const_cast<ResidualEventItem &>(vRI[index]);

        for (size_t j = 0; j < patch_size; j++) {
          double irls_weight = 1.0;
          if (ri.residuals_[j] > config_.huber_threshold_)
            irls_weight = config_.huber_threshold_ / ri.residuals_[j];
          fvec[i * patch_size + j] = sqrt(irls_weight) * ri.residuals_[j];
          ri.irls_weight_[j] = sqrt(irls_weight);
        }
      }
      break;
    }

    default:LOG(ERROR) << "Unsupported Loss function type";
      break;
  }

}

void EventProblemLM::JacobianThread(Job &job, Eigen::MatrixXd &fjac) const {
  ResidualEventItems &vRI = *job.pvRI;
  const Eigen::Quaterniond &Qwb = *job.Qwb;
  const Eigen::Vector3d &twb = *job.twb;
  size_t threadID = job.threadID;
  size_t patch_size = config_.patch_size_x_ * config_.patch_size_y_;

  switch (config_.loss_function_type_) {
    case L2: {
      for (size_t i = threadID; i < indices_for_cur_batch_.size(); i += config_.thread_num_) {
        size_t index = indices_for_cur_batch_[i];
        vRI[index].computeJacobian(Qwb, twb);
        ResidualEventItem &ri = const_cast<ResidualEventItem &>(vRI[index]);
        fjac.block(i * patch_size, 0, patch_size, 6) = ri.jacobian_;
      }
      break;
    }

    case Huber: {
      for (size_t i = threadID; i < indices_for_cur_batch_.size(); i += config_.thread_num_) {
        size_t index = indices_for_cur_batch_[i];
        vRI[index].computeJacobian(Qwb, twb);
        ResidualEventItem &ri = const_cast<ResidualEventItem &>(vRI[index]);
        for (size_t j = 0; j < patch_size; j++)
          fjac.row(i * patch_size + j) = ri.irls_weight_[j] * ri.jacobian_.row(j);
      }
      break;
    }

    default:LOG(ERROR) << "Unsupported Loss function type";
      break;
  }

}

void EventProblemLM::addPerturbation(const Eigen::Matrix<double, 6, 1> &dx,
                                     Eigen::Quaterniond &Qwb,
                                     Eigen::Vector3d &twb) const {
  twb = twb + dx.block<3, 1>(0, 0);
  Eigen::Quaterniond dQ = Eigen::Quaterniond(1, dx[3] / 2.0, dx[4] / 2.0, dx[5] / 2.0);
  Qwb = Qwb * dQ;
  Qwb.normalize();
}

void EventProblemLM::addMotionUpdate(const Eigen::Matrix<double, 6, 1> &dx) {
  addPerturbation(dx, opt_Qwb_, opt_twb_);
}