//
// Created by mpl on 23-11-11.
//

#include <ceres/loss_function.h>

#include <random>
#include <tuple>
#include <utility>

#include <unsupported/Eigen/NonLinearOptimization>

#include "CeresFactor/EventAutoDiffFactor.h"
#include "CeresFactor/EventFactor.h"
#include "CeresFactor/ImuFactor.h"
#include "Manifold/PoseLocalManifold.h"
#include "Manifold/PoseLocalParameterization.h"

#include "EigenOptimization/Problem/EventProblem.h"
#include "EigenOptimization/Problem/SlidingWindowProblem.h"
#include "Optimizer.h"

using namespace CannyEVIT;

Optimizer::Optimizer(const std::string &config_path, EventCamera::Ptr event_camera)
    : event_camera_(std::move(event_camera)),
      patch_size_X_(0),
      patch_size_Y_(0),
      polarity_prediction_(false),
      max_registration_point_(0) {
  LOG(INFO) << "-------------- Optimizer --------------";

  cv::FileStorage fs(config_path, cv::FileStorage::READ);
  if (!fs.isOpened()) LOG(FATAL) << "config not open: " << config_path;

  if (fs["patch_size_X"].isNone()) LOG(ERROR) << "config: patch_size_X is not set";
  patch_size_X_ = fs["patch_size_X"];

  if (fs["patch_size_Y"].isNone()) LOG(ERROR) << "config: patch_size_Y is not set";
  patch_size_Y_ = fs["patch_size_Y"];

  if (fs["polarity_prediction"].isNone()) LOG(ERROR) << "config: polarity_prediction is not set";
  polarity_prediction_ = static_cast<int>(fs["polarity_prediction"]);

  if (fs["max_registration_point"].isNone()) LOG(ERROR) << "config: max_registration_point is not set";
  max_registration_point_ = static_cast<int>(fs["max_registration_point"]);

  if (fs["batch_size"].isNone()) LOG(ERROR) << "config: batch_size is not set";
  batch_size_ = static_cast<int>(fs["batch_size"]);

  if (fs["field_type"].isNone()) LOG(ERROR) << "config: field_type is not set";
  if (fs["field_type"].string() == "distance_field") {
    field_type_ = TimeSurface::FieldType::DISTANCE_FIELD;
    vis_type_ = TimeSurface::VisualizationType::CANNY;
  } else if (fs["field_type"].string() == "inv_time_surface") {
    field_type_ = TimeSurface::FieldType::INV_TIME_SURFACE;
    vis_type_ = TimeSurface::VisualizationType::TIME_SURFACE;
  } else
    LOG(ERROR) << "Unsupported field type";

  LOG(INFO) << "patch_size_X:" << patch_size_X_;
  LOG(INFO) << "patch_size_Y:" << patch_size_Y_;
  LOG(INFO) << "polarity_prediction_:" << polarity_prediction_;
  LOG(INFO) << "batch_size_:" << batch_size_;
  LOG(INFO) << "max_registration_point_:" << max_registration_point_;
}

bool Optimizer::OptimizeEventProblemCeres(pCloud cloud, Frame::Ptr frame) {
  frame->stateToOpt();

  size_t point_num = std::min(max_registration_point_, cloud->size());
  std::set<size_t> sample_indices;
  Utility::uniformSample<size_t>(0, cloud->size() - 1, point_num, sample_indices);

  frame->time_surface_observation_->drawCloud(cloud,
                                              frame->Twc(),
                                              "shuffled cloud",
                                              TimeSurface::PolarType::NEUTRAL,
                                              vis_type_,
                                              false,
                                              sample_indices);

  ceres::Problem problem;
  ceres::Manifold *local_para = new PoseLocalManifold();
  //    ceres::LocalParameterization* local_para = new PoseLocalParameterization();
  problem.AddParameterBlock(frame->opt_pose_.data(), 7, local_para);
  ceres::LossFunction *loss_function = new ceres::HuberLoss(10);

  for (size_t index : sample_indices) {
    problem.AddResidualBlock(
        new EventFactor(cloud->at(index), frame->time_surface_observation_, patch_size_X_, patch_size_Y_),
        loss_function,
        frame->opt_pose_.data());
  }
  ceres::Solver::Options options;
  //    options.check_gradients = true;
  //    options.minimizer_progress_to_stdout = true;
  options.linear_solver_type = ceres::DENSE_SCHUR;
  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);
  LOG(INFO) << summary.BriefReport();

  return true;
}

bool Optimizer::OptimizeSlidingWindowProblemCeres(pCloud cloud, std::deque<Frame::Ptr> &window) {
  ceres::Problem problem;
  ceres::LossFunction *loss_function = new ceres::HuberLoss(10);

  // create opt parameter
  size_t window_size = window.size();
  for (size_t i = 0; i < window_size; i++) {
    window[i]->stateToOpt();
    PoseLocalManifold *local_para = new PoseLocalManifold();
    //        ceres::LocalParameterization* local_para = new PoseLocalParameterization();
    problem.AddParameterBlock(window[i]->opt_pose_.data(), 7, local_para);
    problem.AddParameterBlock(window[i]->opt_speed_bias_.data(), 9);
  }

  size_t point_num = std::min(max_registration_point_, cloud->size());
  std::set<size_t> sample_indices;
  Utility::uniformSample<size_t>(0, cloud->size() - 1, point_num, sample_indices);

  for (size_t i = 0; i < window_size; i++) {
    for (size_t index : sample_indices) {
      problem.AddResidualBlock(
          new EventFactor(cloud->at(index), window[i]->time_surface_observation_, patch_size_X_, patch_size_Y_),
          loss_function,
          window[i]->opt_pose_[i].data());
    }
  }

  for (size_t i = 1; i < window_size; i++) {
    problem.AddResidualBlock(new IMUFactor(window[i]->integration_.get()),
                             nullptr,
                             window[i - 1]->opt_pose_.data(),
                             window[i - 1]->opt_speed_bias_.data(),
                             window[i]->opt_pose_.data(),
                             window[i]->opt_speed_bias_.data());
  }

  ceres::Solver::Options options;
  options.linear_solver_type = ceres::DENSE_SCHUR;
  //    options.use_nonmonotonic_steps = true;
  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);
  LOG(INFO) << summary.BriefReport();

  return true;
}

bool Optimizer::OptimizeSlidingWindowProblemCeresBatch(pCloud cloud, std::deque<Frame::Ptr> &window) {
  // opt variable
  size_t window_size = window.size();
  for (size_t i = 0; i < window_size; i++)
    window[i]->stateToOpt();

  // calculate number of batch
  size_t optimized_points_num = std::min(max_registration_point_, cloud->size());
  size_t problem_num = (optimized_points_num - 1) / batch_size_ + 1;
  std::vector<ceres::Problem> problem_list;

  // do not take ownership
  ceres::Problem::Options problem_options;
  problem_options.loss_function_ownership = ceres::DO_NOT_TAKE_OWNERSHIP;
  problem_options.manifold_ownership = ceres::DO_NOT_TAKE_OWNERSHIP;
  problem_options.cost_function_ownership = ceres::DO_NOT_TAKE_OWNERSHIP;

  // construct problem add parameter
  ceres::Manifold *local_para = new PoseLocalManifold();
  for (size_t i = 0; i < problem_num; i++) {
    problem_list.emplace_back(problem_options);
    for (size_t j = 0; j < window_size; j++) {
      problem_list[i].AddParameterBlock(window[j]->opt_pose_.data(), 7, local_para);
      problem_list[i].AddParameterBlock(window[j]->opt_speed_bias_.data(), 9);
    }
  }

  // construct imu factors
  std::vector<IMUFactor *> imu_factors;
  for (size_t i = 1; i < window_size; i++) imu_factors.emplace_back(new IMUFactor(window[i]->integration_.get()));

  // construct event factors
  std::vector<EventFactor *> event_factors;
  // sample N indices
  std::set<size_t> sample_indices;
  Utility::uniformSample<size_t>(0, cloud->size() - 1, optimized_points_num, sample_indices);

  for (size_t i = 0; i < window_size; i++) {
    std::set<size_t> positive_indices, negative_indices, neutral_indices;

    for (size_t index : sample_indices) {
      std::pair<TimeSurface::PolarType, double> polar_and_weight;
      if (polarity_prediction_) {
        polar_and_weight =
            TimeSurface::determinePolarAndWeight(cloud->at(index), window[i]->last_frame_->Twb(), window[i]->Twb());
        switch (polar_and_weight.first) {
          case TimeSurface::NEUTRAL:neutral_indices.insert(index);
            break;
          case TimeSurface::POSITIVE:positive_indices.insert(index);
            break;
          case TimeSurface::NEGATIVE:negative_indices.insert(index);
            break;
        }
      } else
        polar_and_weight = std::make_pair(TimeSurface::PolarType::NEUTRAL, 1.0);

      event_factors.emplace_back(new EventFactor(cloud->at(index),
                                                 window[i]->time_surface_observation_,
                                                 patch_size_X_,
                                                 patch_size_Y_,
                                                 polar_and_weight.first,
                                                 field_type_,
                                                 polar_and_weight.second));
    }

    if (polarity_prediction_) {
      window[i]->time_surface_observation_->drawCloud(cloud,
                                                      window[i]->Twc(),
                                                      "neutral_before_optimization_frame_" + std::to_string(i),
                                                      TimeSurface::NEUTRAL,
                                                      vis_type_,
                                                      true,
                                                      neutral_indices,
                                                      window[i]->last_frame_->Twc());

      window[i]->time_surface_observation_->drawCloud(cloud,
                                                      window[i]->Twc(),
                                                      "positive_before_optimization_frame_" + std::to_string(i),
                                                      TimeSurface::POSITIVE,
                                                      vis_type_,
                                                      true,
                                                      positive_indices,
                                                      window[i]->last_frame_->Twc());
      window[i]->time_surface_observation_->drawCloud(cloud,
                                                      window[i]->Twc(),
                                                      "negative_before_optimization_frame_" + std::to_string(i),
                                                      TimeSurface::NEGATIVE,
                                                      vis_type_,
                                                      true,
                                                      negative_indices,
                                                      window[i]->last_frame_->Twc());
    }

  }
  cv::waitKey(0);

  // add imu residuals
  for (auto &problem : problem_list) {
    for (size_t i = 1; i < window_size; i++) {
      problem.AddResidualBlock(imu_factors[i - 1],
                               nullptr,
                               window[i - 1]->opt_pose_.data(),
                               window[i - 1]->opt_speed_bias_.data(),
                               window[i]->opt_pose_.data(),
                               window[i]->opt_speed_bias_.data());
    }
  }

  // add event residuals
  ceres::LossFunction *event_loss_function = new ceres::HuberLoss(10);
  size_t batch_residual_num = window_size * batch_size_;
  for (size_t i = 0; i < event_factors.size(); i++) {
    size_t problem_index = i / batch_residual_num;
    size_t pose_index = i % window_size;
    problem_list[problem_index].AddResidualBlock(event_factors[i],
                                                 event_loss_function,
                                                 window[pose_index]->opt_pose_.data());
  }

  size_t max_iteration = 50;
  size_t cur_iteration = 0;
  ceres::Solver::Options options;
  //    options.minimizer_progress_to_stdout = true;
  //    options.use_nonmonotonic_steps = true;
  options.linear_solver_type = ceres::DENSE_SCHUR;
  options.max_num_iterations = 1;
  ceres::TerminationType cur_state = ceres::TerminationType::NO_CONVERGENCE;

  // batch Optimization
  while (cur_state != ceres::TerminationType::CONVERGENCE && cur_iteration < max_iteration) {
    ceres::Solver::Summary summary;
    size_t cur_problem_index = cur_iteration % problem_num;
    ceres::Solve(options, &problem_list[cur_problem_index], &summary);
    cur_state = summary.termination_type;
    cur_iteration++;
    std::cout << summary.BriefReport() << std::endl;
  }
  std::cout << cur_iteration << std::endl;


  // delete
  delete event_loss_function;
  delete local_para;
  for (auto p : event_factors) delete p;
  for (auto p : imu_factors) delete p;
  std::cout << "done" << std::endl;

  return true;
}

// TODO: make this optimize opt part
bool Optimizer::OptimizeEventProblemEigen(pCloud cloud, Frame::Ptr frame) {
  EventProblemConfig event_problem_config(patch_size_X_, patch_size_Y_);
  event_problem_config.max_iteration_ = 50;
  EventProblemLM problem(event_problem_config);
  problem.setProblem(frame, cloud);

  Eigen::LevenbergMarquardt<EventProblemLM, double> lm(problem);
  lm.resetParameters();
  lm.parameters.ftol = 1e-3;
  lm.parameters.xtol = 1e-3;
  lm.parameters.maxfev = event_problem_config.max_iteration_ * 8; // maximum number of function evaluation

  size_t iteration = 0;
  while (true) {
    if (iteration >= event_problem_config.max_iteration_) {
      LOG(INFO) << "max iteration reached, break";
      break;
    }

    problem.nextBatch();
    Eigen::VectorXd x(6);
    x.fill(0.0);
    if (lm.minimizeInit(x) == Eigen::LevenbergMarquardtSpace::ImproperInputParameters) {
      LOG(ERROR) << "ImproperInputParameters for LM (Tracking).";
      return false;
    }

    Eigen::LevenbergMarquardtSpace::Status status = lm.minimizeOneStep(x);
    problem.addMotionUpdate(x);
    iteration++;
    if (status == Eigen::LevenbergMarquardtSpace::Status::RelativeErrorTooSmall ||
        status == Eigen::LevenbergMarquardtSpace::Status::RelativeErrorAndReductionTooSmall) {
      LOG(INFO) << "Converged! break!";
      break;
    }
  }

  frame->set_Twb(problem.opt_Qwb_, problem.opt_twb_);
  return true;
}

// TODO: make this optimize opt part
bool Optimizer::OptimizeSlidingWindowProblemEigen(pCloud cloud, std::deque<Frame::Ptr> &window) {
  SlidingWindowProblemConfig problem_config(patch_size_X_, patch_size_Y_);

  SlidingWindowProblem problem(problem_config, window.size());
  problem.setProblem(window, cloud);

  Eigen::LevenbergMarquardt<SlidingWindowProblem, double> lm(problem);
  lm.resetParameters();
  lm.parameters.ftol = 1e-3;
  lm.parameters.xtol = 1e-3;
  lm.parameters.maxfev = problem_config.max_iteration_ * 8; // maximum number of function evaluation

  size_t iteration = 0;
  while (true) {
    if (iteration >= problem_config.max_iteration_) {
      LOG(INFO) << "max iteration reached, break";
      break;
    }

    problem.nextBatch();
    Eigen::VectorXd x(15 * window.size());
    x.fill(0.0);
    if (lm.minimizeInit(x) == Eigen::LevenbergMarquardtSpace::ImproperInputParameters) {
      LOG(ERROR) << "ImproperInputParameters for LM (Tracking).";
      return false;
    }

    Eigen::LevenbergMarquardtSpace::Status status = lm.minimizeOneStep(x);
    problem.addMotionUpdate(x);
    iteration++;
    if (status == Eigen::LevenbergMarquardtSpace::Status::RelativeErrorTooSmall ||
        status == Eigen::LevenbergMarquardtSpace::Status::RelativeErrorAndReductionTooSmall) {
      LOG(INFO) << "Converged! break!";
      break;
    }
  }

  for (size_t i = 0; i < window.size(); i++) {
    window[i]->set_Twb(problem.opt_Qwb_vec_[i], problem.opt_twb_vec_[i]);
    window[i]->set_velocity(problem.opt_velocity_vec_[i]);
    window[i]->set_Ba(problem.opt_ba_vec_[i]);
    window[i]->set_Bg(problem.opt_bg_vec_[i]);
  }
  LOG(INFO) << "after optimize:";
  LOG(INFO) << window.back()->Qwb().coeffs().transpose();
  LOG(INFO) << window.back()->twb().transpose();
  return true;
}

bool Optimizer::OptimizeVelovityBias(const std::vector<Frame::Ptr> &window) {
  LOG(INFO) << "init start";
  size_t window_size = window.size();

  ceres::Problem problem;
  for (size_t i = 0; i < window.size(); i++) {

    window[i]->stateToOpt();
    PoseLocalManifold *local_para = new PoseLocalManifold();
    //        ceres::LocalParameterization* local_para = new PoseLocalParameterization();

    problem.AddParameterBlock(window[i]->opt_pose_.data(), 7, local_para);
    problem.SetParameterBlockConstant(window[i]->opt_pose_.data());
    problem.AddParameterBlock(window[i]->opt_speed_bias_.data(), 9);
  }

  for (size_t i = 1; i < window.size(); i++) {
    LOG(INFO) << window[i]->integration_->delta_p.transpose();
    problem.AddResidualBlock(new IMUFactor(window[i]->integration_.get()),
                             nullptr,
                             window[i - 1]->opt_pose_.data(),
                             window[i - 1]->opt_speed_bias_.data(),
                             window[i]->opt_pose_.data(),
                             window[i]->opt_speed_bias_.data());
  }
  LOG(INFO) << "opt start";

  ceres::Solver::Options options;
  //    options.check_gradients = true;
  options.minimizer_progress_to_stdout = true;
  //    options.linear_solver_type = ceres::DENSE_SCHUR;
  options.use_nonmonotonic_steps = true;
  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);
  LOG(INFO) << summary.BriefReport();

//  for (size_t i = 0; i < window_size; i++) {
//    window[i]->set_velocity(opt_speed_bias[i].segment<3>(0));
//    window[i]->set_Ba(opt_speed_bias[i].segment<3>(3));
//    window[i]->set_Bg(opt_speed_bias[i].segment<3>(6));
//
//    LOG(INFO) << "Ba" << i << ": " << window[i]->Ba().transpose();
//    LOG(INFO) << "v" << i << ": " << window[i]->velocity().transpose();
//    LOG(INFO) << "Bg" << i << ": " << window[i]->Bg().transpose();
//  }

  return true;
}

// TODO: make this optimize opt part, need to debug
bool Optimizer::initVelocityBias(const std::vector<Frame::Ptr> &window) {
  // solve gyroscope bias
  {
    Eigen::Matrix3d A = Eigen::Matrix3d::Zero();
    Eigen::Vector3d b = Eigen::Vector3d::Zero();
    Eigen::Vector3d delta_bg = Eigen::Vector3d::Zero();
    for (size_t i = 1; i < window.size(); i++) {
      const Frame::Ptr &frame_i = window[i - 1];
      const Frame::Ptr &frame_j = window[i];

      Eigen::Quaterniond q_ij(frame_i->Qwb().conjugate() * frame_j->Qwb());

      Eigen::Matrix3d tmp_A = frame_j->integration_->jacobian.template block<3, 3>(O_R, O_BG);
      Eigen::Vector3d tmp_b = 2 * (frame_j->integration_->delta_q.inverse() * q_ij).vec();
      A += tmp_A.transpose() * tmp_A;
      b += tmp_A.transpose() * tmp_b;
    }
    delta_bg = A.ldlt().solve(b);

    Eigen::JacobiSVD<Eigen::Matrix3d> svd(A);
    double cond = svd.singularValues()(0) / svd.singularValues()(svd.singularValues().size() - 1);
    LOG(INFO) << "conditional number in solving gyroscope bias:" << cond;
    LOG(INFO) << "Init gyroscope bias: " << delta_bg.transpose();

    for (size_t i = 1; i < window.size(); i++) {
      window[i]->set_Bg(delta_bg);
      window[i]->integration_->repropagate(Eigen::Vector3d::Zero(), delta_bg);
    }
  }
  // solve velocity and acc bias
  //    {
  //        size_t n_state = window.size() * 3 + 3;
  //        Eigen::MatrixXd A(n_state, n_state);
  //        Eigen::VectorXd b(n_state);
  //        A.setZero();
  //        b.setZero();
  //
  //        for(size_t i = 0; i < window.size() - 1; i++)
  //        {
  //            const Frame::Ptr & frame_i = window[i];
  //            const Frame::Ptr & frame_j = window[i+1];
  //
  //            Eigen::Matrix3d R_bi_w = frame_i->Rwb().transpose();
  //            double delta_t = frame_j->integration_->dt;
  //
  //            Eigen::Matrix<double, 6, 9> tmp_A;
  //            Eigen::Matrix<double, 6, 1> tmp_b;
  //            tmp_A.block<3, 3>(0, 0) = R_bi_w * delta_t;
  //            tmp_A.block<3, 3>(0, 3) = Eigen::Matrix3d::Zero();
  //            tmp_A.block<3, 3>(0, 6) = frame_j->integration_->jacobian.template block<3, 3>(O_P, O_BA);
  //            tmp_b.block<3, 1>(0, 0) = R_bi_w * (frame_j->twb() - frame_i->twb()) +
  //                                      R_bi_w * IntegrationBase::G * delta_t * delta_t / 2.0 -
  //                                      frame_j->integration_->delta_p;
  //
  //            tmp_A.block<3, 3>(3, 0) = R_bi_w;
  //            tmp_A.block<3, 3>(3, 3) = -R_bi_w;
  //            tmp_A.block<3, 3>(3, 6) = frame_j->integration_->jacobian.template block<3, 3>(O_V, O_BA);
  //            tmp_b.block<3, 1>(3, 0) = R_bi_w * IntegrationBase::G * delta_t - frame_j->integration_->delta_v;
  //
  //            Eigen::MatrixXd r_A = tmp_A.transpose() * tmp_A;
  //            Eigen::VectorXd r_b = tmp_A.transpose() * tmp_b;
  //
  //            A.block<6, 6>(i * 3, i * 3) += r_A.topLeftCorner<6, 6>();
  //            b.segment<6>(i * 3) += r_b.head<6>();
  //
  //            A.bottomRightCorner<3, 3>() += r_A.bottomRightCorner<3, 3>();
  //            b.tail<3>() += r_b.tail<3>();
  //
  //            A.block<6, 3>(i * 3, n_state - 3) += r_A.topRightCorner<6, 3>();
  //            A.block<3, 6>(n_state - 3, i * 3) += r_A.bottomLeftCorner<3, 6>();
  //        }
  //        A = A * 1000.0;
  //        b = b * 1000.0;
  //        Eigen::VectorXd x = A.ldlt().solve(b);
  //
  //        Eigen::Vector3d acc_bias = x.tail<3>();
  //        Eigen::JacobiSVD<Eigen::MatrixXd> svd(A);
  //        double cond = svd.singularValues()(0) / svd.singularValues()(svd.singularValues().size()-1);
  //        LOG(INFO) << "conditional number in solving acc bias and velocity:" << cond;
  //        LOG(INFO) << "Init acc bias: " << acc_bias.transpose();
  //
  //        for (size_t i = 0; i < window.size(); i++)
  //        {
  //            window[i]->set_Ba(acc_bias);
  //            window[i]->set_velocity(x.segment<3>(i*3));
  //            LOG(INFO) << "velocity" << i << ":" << window[i]->velocity().transpose();
  //        }
  //    }

  // solve velocity and acc bias
  {
    size_t n_state = window.size() * 3;
    Eigen::MatrixXd A(n_state, n_state);
    Eigen::VectorXd b(n_state);
    A.setZero();
    b.setZero();
    std::cout << window.size() << std::endl;
    for (size_t i = 0; i < window.size() - 1; i++) {
      const Frame::Ptr &frame_i = window[i];
      const Frame::Ptr &frame_j = window[i + 1];

      Eigen::Matrix3d R_bi_w = frame_i->Rwb().transpose();
      double delta_t = frame_j->integration_->dt;

      Eigen::Matrix<double, 6, 6> tmp_A;
      Eigen::Matrix<double, 6, 1> tmp_b;
      tmp_A.block<3, 3>(0, 0) = R_bi_w * delta_t;
      tmp_A.block<3, 3>(0, 3) = Eigen::Matrix3d::Zero();
      tmp_b.block<3, 1>(0, 0) = R_bi_w * (frame_j->twb() - frame_i->twb()) +
          R_bi_w * IntegrationBase::G * delta_t * delta_t / 2.0 - frame_j->integration_->delta_p;

      tmp_A.block<3, 3>(3, 0) = R_bi_w;
      tmp_A.block<3, 3>(3, 3) = -R_bi_w;
      tmp_b.block<3, 1>(3, 0) = R_bi_w * IntegrationBase::G * delta_t - frame_j->integration_->delta_v;

      Eigen::MatrixXd r_A = tmp_A.transpose() * tmp_A;
      Eigen::VectorXd r_b = tmp_A.transpose() * tmp_b;

      A.block<6, 6>(i * 3, i * 3) += r_A.topLeftCorner<6, 6>();
      b.segment<6>(i * 3) += r_b.head<6>();
    }
    A = A * 1000.0;
    b = b * 1000.0;
    Eigen::VectorXd x = A.ldlt().solve(b);

    Eigen::JacobiSVD<Eigen::MatrixXd> svd(A);

    double cond = svd.singularValues()(0) / svd.singularValues()(svd.singularValues().size() - 1);
    LOG(INFO) << "conditional number in solving acc bias and velocity:" << cond;

    for (size_t i = 0; i < window.size(); i++) {
      window[i]->set_velocity(x.segment<3>(i * 3));
      LOG(INFO) << "velocity" << i << ":" << window[i]->velocity().transpose();
    }
  }

  return true;
}

