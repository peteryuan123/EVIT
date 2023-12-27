//
// Created by mpl on 23-12-25.
//

#include <random>
#include <thread>
#include "EigenOptimization/Problem/SlidingWindowProblem.h"

using namespace CannyEVIT;

SlidingWindowProblem::SlidingWindowProblem(const SlidingWindowProblemConfig &config, size_t window_size)
    : GenericFunctor<double>(15 * window_size, 0),
      config_(config),
      window_size_(window_size),
      cloud_(nullptr),
      point_num_(0),
      cur_batch_start_index_(0) {
  opt_Qwb_vec_.reserve(window_size_);
  opt_twb_vec_.reserve(window_size_);
  indices_for_cur_batch_.reserve(config_.batch_size_);
  event_residuals_.reserve(window_size_ * config_.max_registration_points_);
}

void SlidingWindowProblem::nextBatch() {
  indices_for_cur_batch_.clear();
  indices_for_cur_batch_.reserve(config_.batch_size_);
  while (indices_for_cur_batch_.size() < config_.batch_size_) {
    indices_for_cur_batch_.emplace_back(cur_batch_start_index_);
    cur_batch_start_index_++;
    if (cur_batch_start_index_ >= point_num_)
      cur_batch_start_index_ = 0;
  }

  size_t event_residual_size = config_.batch_size_ * window_size_ * config_.patch_size_x_ * config_.patch_size_y_;
  size_t imu_residual_size = (window_size_ - 1) * 15;
  resetNumberValues(event_residual_size + imu_residual_size);
}

void SlidingWindowProblem::setProblem(const std::deque<Frame::Ptr> &window, const pCloud &cloud) {
  std::cout << "set probelm" << std::endl;
  window_size_ = window.size();
  cloud_ = cloud;
  for (const auto &frame : window) {
    window_.emplace_back(frame);
    opt_Qwb_vec_.emplace_back(frame->Qwb());
    opt_twb_vec_.emplace_back(frame->twb());
    opt_velocity_vec_.emplace_back(frame->velocity());
    opt_ba_vec_.emplace_back(frame->Ba());
    opt_bg_vec_.emplace_back(frame->Bg());
  }
  point_num_ = std::min(config_.max_registration_points_, cloud->size());
  event_residuals_.reserve(window_size_ * point_num_);

  std::set<size_t> set_samples;
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<size_t> dist(0, cloud->size() - 1);
  while (set_samples.size() < point_num_) set_samples.insert(dist(gen));

  // construct event residuals
  //TODO: fix polarity prediction and this problem should work as the same as Event problem when window size equals 1
  for (auto iter = set_samples.begin(); iter != set_samples.end(); iter++) {
    for (const auto &frame : window_) {
      event_residuals_.emplace_back(cloud_->at(*iter),
                                    frame->time_surface_observation_,
                                    config_.patch_size_x_,
                                    config_.patch_size_y_,
                                    TimeSurface::PolarType::NEUTRAL,
                                    config_.field_type_);
    }
  }

  for (size_t i = 1; i < window_size_; i++)
    imu_residuals_.emplace_back(window_[i]->integration_);

  // construct imu residuals
  size_t event_residual_size = point_num_ * window_size_ * config_.patch_size_x_ * config_.patch_size_y_;
  size_t imu_residual_size = (window_size_ - 1) * 15;
  resetNumberValues(event_residual_size + imu_residual_size);
  std::cout << "set probelm done" << std::endl;

}

int SlidingWindowProblem::operator()(const Eigen::MatrixXd &x, Eigen::VectorXd &fvec) const {

  std::vector<Eigen::Quaterniond> Qwb_vec = opt_Qwb_vec_;
  std::vector<Eigen::Vector3d> twb_vec = opt_twb_vec_;
  std::vector<Eigen::Vector3d> velocity_vec = opt_velocity_vec_;
  std::vector<Eigen::Vector3d> ba_vec = opt_ba_vec_;
  std::vector<Eigen::Vector3d> bg_vec = opt_bg_vec_;
  addPerturbation(x, Qwb_vec, twb_vec, velocity_vec, ba_vec, bg_vec);

  // construct event residual
  std::vector<Job> jobs(config_.thread_num_);
  for (size_t i = 0; i < config_.thread_num_; i++) {
    jobs[i].pvRI = const_cast<ResidualEventItems *>(&event_residuals_);
    jobs[i].Qwb_vec = const_cast<std::vector<Eigen::Quaterniond> *>(&Qwb_vec);
    jobs[i].twb_vec = const_cast<std::vector<Eigen::Vector3d> *>(&twb_vec);
    jobs[i].threadID = i;
  }

  std::vector<std::thread> threads;
  for (size_t i = 0; i < config_.thread_num_; i++)
    threads.emplace_back(std::bind(&SlidingWindowProblem::residualThread, this, jobs[i], fvec));
  for (auto &thread : threads)
    if (thread.joinable())
      thread.join();

  // construct imu residual
  size_t event_residual_size = config_.batch_size_ * window_size_ * config_.patch_size_x_ * config_.patch_size_y_;
  for (size_t i = 1; i < window_size_; i++) {
    ResidualImuItem &ri = const_cast<ResidualImuItem &>(imu_residuals_[i - 1]);
    ri.computeResidual(twb_vec[i - 1],
                       Qwb_vec[i - 1],
                       velocity_vec[i - 1],
                       ba_vec[i - 1],
                       bg_vec[i - 1],
                       twb_vec[i],
                       Qwb_vec[i],
                       velocity_vec[i],
                       ba_vec[i],
                       bg_vec[i]);
//    TODO: ADD HUBER LOSS
    fvec.segment<15>(event_residual_size + 15 * (i - 1)) = ri.residuals_;
  }

  return 0;
}

int SlidingWindowProblem::df(const Eigen::MatrixXd &x, Eigen::MatrixXd &fjac) const {

  if (x.any()) {
    std::cerr << "The Jacobian is not evaluated at Zero !!!!!!!!!!!!!";
    exit(-1);
  }
  fjac.resize(m_values, window_size_ * 15);

  std::vector<Eigen::Quaterniond> Qwb_vec = opt_Qwb_vec_;
  std::vector<Eigen::Vector3d> twb_vec = opt_twb_vec_;
  std::vector<Eigen::Vector3d> velocity_vec = opt_velocity_vec_;
  std::vector<Eigen::Vector3d> ba_vec = opt_ba_vec_;
  std::vector<Eigen::Vector3d> bg_vec = opt_bg_vec_;
  addPerturbation(x, Qwb_vec, twb_vec, velocity_vec, ba_vec, bg_vec);

  // construct event jacobian
  std::vector<Job> jobs(config_.thread_num_);
  for (size_t i = 0; i < config_.thread_num_; i++) {
    jobs[i].pvRI = const_cast<ResidualEventItems *>(&event_residuals_);
    jobs[i].Qwb_vec = const_cast<std::vector<Eigen::Quaterniond> *>(&Qwb_vec);
    jobs[i].twb_vec = const_cast<std::vector<Eigen::Vector3d> *>(&twb_vec);
    jobs[i].threadID = i;
  }

  std::vector<std::thread> threads;
  for (size_t i = 0; i < config_.thread_num_; i++)
    threads.emplace_back(std::bind(&SlidingWindowProblem::JacobianThread, this, jobs[i], fjac));
  for (auto &thread : threads)
    if (thread.joinable())
      thread.join();

  // construct imu jacobian
  size_t event_residual_size = config_.batch_size_ * window_size_ * config_.patch_size_x_ * config_.patch_size_y_;
  for (size_t i = 1; i < window_size_; i++) {
    ResidualImuItem &ri = const_cast<ResidualImuItem &>(imu_residuals_[i - 1]);
    ri.computeJacobian(twb_vec[i - 1],
                       Qwb_vec[i - 1],
                       velocity_vec[i - 1],
                       ba_vec[i - 1],
                       bg_vec[i - 1],
                       twb_vec[i],
                       Qwb_vec[i],
                       velocity_vec[i],
                       ba_vec[i],
                       bg_vec[i]);
//    TODO: ADD HUBER LOSS
    fjac.block<15, 15>(event_residual_size + 15 * (i - 1), 15 * (i - 1)) = ri.jacobian_i_;
    fjac.block<15, 15>(event_residual_size + 15 * (i - 1), 15 * i) = ri.jacobian_j_;
  }

  return 0;
}

void SlidingWindowProblem::residualThread(SlidingWindowProblem::Job &job, Eigen::VectorXd &fvec) const {
  ResidualEventItems &vRI = *job.pvRI;
  const std::vector<Eigen::Quaterniond> &Qwb_vec = *job.Qwb_vec;
  const std::vector<Eigen::Vector3d> &twb_vec = *job.twb_vec;
  size_t threadID = job.threadID;

  size_t patch_size = config_.patch_size_x_ * config_.patch_size_y_;
  size_t window_residual_num = window_size_ * patch_size;

  switch (config_.loss_function_type_) {
    case L2: {
      for (size_t i = threadID; i < indices_for_cur_batch_.size(); i += config_.thread_num_) {
        size_t point_index = indices_for_cur_batch_[i];
        for (size_t j = 0; j < window_size_; j++) {
          std::cout << Qwb_vec[j].coeffs().transpose() << std::endl;
          std::cout << twb_vec[j].transpose() << std::endl;
          vRI[point_index * window_size_ + j].computeResidual(Qwb_vec[j], twb_vec[j]);
          fvec.segment(i * window_residual_num + j * patch_size, patch_size) =
              vRI[point_index * window_size_ + j].residuals_;
        }
      }
      break;
    }

    case Huber: {
      for (size_t i = threadID; i < indices_for_cur_batch_.size(); i += config_.thread_num_) {
        size_t point_index = indices_for_cur_batch_[i];
        for (size_t j = 0; j < window_size_; j++) {
          ResidualEventItem &ri = const_cast<ResidualEventItem &>(vRI[point_index * window_size_ + j]);
          ri.computeResidual(Qwb_vec[j], twb_vec[j]);
//          LOG(INFO) << j;
//          LOG(INFO) << Qwb_vec[j].coeffs().transpose();
//          LOG(INFO) << twb_vec[j].transpose();
          for (size_t patch_idx = 0; patch_idx < patch_size; patch_idx++) {
            double irls_weight = 1.0;
            if (ri.residuals_[patch_idx] > config_.huber_threshold_)
              irls_weight = config_.huber_threshold_ / ri.residuals_[patch_idx];

            fvec[i * window_residual_num + j * patch_size + patch_idx] =
                sqrt(irls_weight) * vRI[point_index * window_size_ + j].residuals_[patch_idx];
            ri.irls_weight_[patch_idx] = sqrt(irls_weight);
          }
        }
      }
      break;
    }

    default: LOG(ERROR) << "Unsupported Loss function type";
      break;
  }

}

void SlidingWindowProblem::JacobianThread(CannyEVIT::SlidingWindowProblem::Job &job, Eigen::MatrixXd &fjac) const {
  ResidualEventItems &vRI = *job.pvRI;
  const std::vector<Eigen::Quaterniond> &Qwb_vec = *job.Qwb_vec;
  const std::vector<Eigen::Vector3d> &twb_vec = *job.twb_vec;
  size_t threadID = job.threadID;

  size_t patch_size = config_.patch_size_x_ * config_.patch_size_y_;
  size_t window_residual_num = window_size_ * patch_size;

  switch (config_.loss_function_type_) {
    case L2: {
      for (size_t i = threadID; i < indices_for_cur_batch_.size(); i += config_.thread_num_) {
        size_t point_index = indices_for_cur_batch_[i];
        for (size_t j = 0; j < window_size_; j++) {
          vRI[point_index * window_size_ + j].computeJacobian(Qwb_vec[j], twb_vec[j]);
          fjac.block(i * window_residual_num + j * patch_size, j * 15, patch_size, 6) =
              vRI[point_index * window_size_ + j].jacobian_;
        }
      }
      break;
    }

    case Huber: {
      for (size_t i = threadID; i < indices_for_cur_batch_.size(); i += config_.thread_num_) {
        size_t point_index = indices_for_cur_batch_[i];
        for (size_t j = 0; j < window_size_; j++) {
          ResidualEventItem &ri = const_cast<ResidualEventItem &>(vRI[point_index * window_size_ + j]);
          ri.computeJacobian(Qwb_vec[j], twb_vec[j]);
          for (size_t patch_idx = 0; patch_idx < patch_size; patch_idx++) {
            fjac.block(i * window_residual_num + j * patch_size + patch_idx, j * 15, 1, 6) =
                ri.irls_weight_[patch_idx] * ri.jacobian_.row(patch_idx);
          }
        }
      }
      break;
    }

    default: LOG(ERROR) << "Unsupported Loss function type";
      break;
  }
}

void SlidingWindowProblem::addMotionUpdate(const Eigen::MatrixXd &dx) {
  addPerturbation(dx, opt_Qwb_vec_, opt_twb_vec_, opt_velocity_vec_, opt_ba_vec_, opt_bg_vec_);
}

void SlidingWindowProblem::addPerturbation(const Eigen::MatrixXd &x,
                                           std::vector<Eigen::Quaterniond> &Qwb_vec,
                                           std::vector<Eigen::Vector3d> &twb_vec,
                                           std::vector<Eigen::Vector3d> &velocity_vec,
                                           std::vector<Eigen::Vector3d> &ba_vec,
                                           std::vector<Eigen::Vector3d> &bg_vec) const {
  for (size_t i = 0; i < window_size_; i++) {
    Eigen::Quaterniond &Qwb = Qwb_vec[i];
    Eigen::Vector3d &twb = twb_vec[i];
    Eigen::Vector3d &velocity = velocity_vec[i];
    Eigen::Vector3d &ba = ba_vec[i];
    Eigen::Vector3d &bg = bg_vec[i];

    twb += x.block<3, 1>(15 * i, 0);
    Eigen::Quaterniond dQ = Eigen::Quaterniond(1, x(15 * i + 3, 0) / 2.0, x(15 * i + 4, 0) / 2.0, x(15 * i + 5, 0) / 2.0);
    Qwb = Qwb * dQ;
    Qwb.normalize();
    velocity += x.block<3, 1>(15 * i + 6, 0);
    ba += x.block<3, 1>(15 * i + 9, 0);
    bg += x.block<3, 1>(15 * i + 12, 0);
  }
}


