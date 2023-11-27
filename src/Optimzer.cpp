//
// Created by mpl on 23-11-11.
//

#include "Optimizer.h"
#include "CeresFactor/EventFactor.h"
#include "CeresFactor/ImuFactor.h"
#include "CeresFactor/EventAutoDiffFactor.h"
#include "Manifold/PoseLocalParameterization.h"
#include "Manifold/PoseLocalManifold.h"
#include <ceres/loss_function.h>
#include <random>
#include <utility>

using namespace CannyEVIT;

Optimizer::Optimizer(const std::string &config_path, EventCamera::Ptr event_camera)
: event_camera_(std::move(event_camera)), patch_size_X_(0), patch_size_Y_(0)
{
    LOG(INFO) << "-------------- Optimizer --------------";

    cv::FileStorage fs(config_path, cv::FileStorage::READ);
    if (!fs.isOpened())
        LOG(FATAL)<< "config not open: " << config_path;

    if (fs["patch_size_X"].isNone())
        LOG(ERROR) << "config: patch_size_X is not set";
    patch_size_X_ = fs["patch_size_X"];

    if (fs["patch_size_Y"].isNone())
        LOG(ERROR) << "config: patch_size_Y is not set";
    patch_size_Y_ = fs["patch_size_Y"];

    LOG(INFO) << "patch_size_X:" << patch_size_X_;
    LOG(INFO) << "patch_size_Y:" << patch_size_Y_;
}

bool Optimizer::OptimizeEventProblemCeres(CannyEVIT::pCloud cloud, Frame::Ptr frame)
{
    Eigen::Quaterniond Qwb = frame->Qwb();
    Eigen::Vector3d twb = frame->twb();
    Eigen::Matrix<double, 7, 1> pose_param;
    pose_param << twb.x(), twb.y(), twb.z(), Qwb.x(), Qwb.y(), Qwb.z(), Qwb.w();


    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(cloud->begin(), cloud->end(), g);
    size_t MAX_REGISTRATION_POINTS_ = std::min(static_cast<size_t>(3000), cloud->size());

    pCloud cur_cloud(new std::vector<Point>(cloud->begin(), cloud->begin()+MAX_REGISTRATION_POINTS_));
    frame->time_surface_observation_->drawCloud(cur_cloud, frame->Twc(), "shuffled cloud");

    ceres::Problem problem;
    ceres::LocalParameterization* local_para = new PoseLocalParameterization();
    problem.AddParameterBlock(pose_param.data(), 7, local_para);
    ceres::LossFunction *loss_function;
    loss_function = new ceres::HuberLoss(0.5);

//    ceres::ProductManifold<ceres::EuclideanManifold<3>, ceres::EigenQuaternionManifold>* se3 =
//            new ceres::ProductManifold<ceres::EuclideanManifold<3>, ceres::EigenQuaternionManifold>(ceres::EuclideanManifold<3>(), ceres::EigenQuaternionManifold());
//    problem.SetManifold(pose_param.data(), se3);
//    problem.AddParameterBlock(pose_param.data(), 7, se3);
    for (size_t i = 0; i < MAX_REGISTRATION_POINTS_; i++)
    {
        problem.AddResidualBlock(new EventFactor(cloud->at(i), frame->time_surface_observation_, patch_size_X_, patch_size_Y_),
                                 loss_function, pose_param.data());
//        problem.AddResidualBlock(EventAutoDiffFactor::create(cloud->at(i), frame->time_surface_observation_, patch_size_X_, patch_size_Y_),
//                                 loss_function, pose_param.data());
    }
    ceres::Solver::Options options;
//    options.check_gradients = true;
//    options.minimizer_progress_to_stdout = true;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    LOG(INFO) << summary.BriefReport();

//    std::vector<EventFactor*> factors;
//    for (int i = 0; i < MAX_REGISTRATION_POINTS_; i++)
//    {
//        factors.emplace_back(new EventFactor(cloud->at(i), frame->time_surface_observation_, patch_size_X_, patch_size_Y_));
//    }
//
//    LOG(INFO) << factors.size();
//    int batchNum = MAX_REGISTRATION_POINTS_ / 10;
//    for (int i = 0; i < 20; i++)
//    {
//        ceres::Problem::Options problem_options;
//        problem_options.cost_function_ownership = ceres::Ownership::DO_NOT_TAKE_OWNERSHIP;
//        ceres::Problem problem(problem_options);
//        ceres::LocalParameterization* local_para = new PoseLocalParameterization();
//        problem.AddParameterBlock(pose_param.data(), 7, local_para);
//        ceres::LossFunction *loss_function;
//        loss_function = new ceres::HuberLoss(10.0);
//
//        for (int j = 0; j < batchNum; j++)
//        {
//            int index = i * batchNum + j;
//
//            if (index >= MAX_REGISTRATION_POINTS_)
//                break;
//            problem.AddResidualBlock(factors[index], loss_function, pose_param.data());
//        }
//
//        ceres::Solver::Options options;
//        options.linear_solver_type = ceres::DENSE_SCHUR;
//        options.max_num_iterations = 5;
////        options.minimizer_progress_to_stdout = true;
//
//        ceres::Solver::Summary summary;
//        ceres::Solve(options, &problem, &summary);
//        LOG(INFO) << summary.BriefReport();
//    }

    Eigen::Vector3d new_twb(pose_param[0], pose_param[1], pose_param[2]);
    Eigen::Quaterniond new_Qwb(pose_param[6], pose_param[3], pose_param[4], pose_param[5]);
    frame->set_Twb(new_Qwb, new_twb);
    return true;
}

bool Optimizer::OptimizeSlidingWindowProblemCeres(CannyEVIT::pCloud cloud, std::deque<Frame::Ptr> window)
{
    ceres::Problem problem;
    ceres::LossFunction *loss_function;
    loss_function = new ceres::HuberLoss(10);

    size_t window_size = window.size();

    std::vector<Eigen::Matrix<double, 7, 1>> opt_pose(window_size);
    std::vector<Eigen::Matrix<double, 9, 1>> opt_speed_bias(window_size);
    for (size_t i = 0; i < window_size; i++)
    {
        const Eigen::Quaterniond& Qwb = window[i]->Qwb();
        const Eigen::Vector3d& twb = window[i]->twb();
        const Eigen::Vector3d& velocity = window[i]->velocity();
        const Eigen::Vector3d& ba = window[i]->Ba();
        const Eigen::Vector3d& bg = window[i]->Bg();

        opt_pose[i] << twb.x(), twb.y(), twb.z(), Qwb.x(), Qwb.y(), Qwb.z(), Qwb.w();
        opt_speed_bias[i] << velocity.x(), velocity.y(), velocity.z(), ba.x(), ba.y(), ba.z(), bg.x(), bg.y(), bg.z();

//        PoseLocalManifold* local_para = new PoseLocalManifold();

        ceres::LocalParameterization* local_para = new PoseLocalParameterization();
        problem.AddParameterBlock(opt_pose[i].data(), 7, local_para);
        problem.AddParameterBlock(opt_speed_bias[i].data(), 9);
    }

    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(cloud->begin(), cloud->end(), g);
    size_t MAX_REGISTRATION_POINTS_ = std::min(static_cast<size_t>(3000), cloud->size());
    pCloud cur_cloud(new std::vector<Point>(cloud->begin(), cloud->begin()+MAX_REGISTRATION_POINTS_));


    for (size_t i = 0; i < window_size; i++)
    {
        for (size_t j = 0; j < MAX_REGISTRATION_POINTS_; j++)
        {
            problem.AddResidualBlock(new EventFactor(cloud->at(j), window[i]->time_surface_observation_, patch_size_X_, patch_size_Y_),
                                     loss_function, opt_pose[i].data());
        }
    }

    for (size_t i = 1; i < window_size; i++)
    {
        problem.AddResidualBlock(new IMUFactor(window[i]->integration_.get()), nullptr, opt_pose[i-1].data(), opt_speed_bias[i-1].data(),
                                 opt_pose[i].data(), opt_speed_bias[i].data());
    }

    ceres::Solver::Options options;
//    options.check_gradients = true;
//    options.minimizer_progress_to_stdout = true;
    options.linear_solver_type = ceres::DENSE_SCHUR;
//    options.use_nonmonotonic_steps = true;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    LOG(INFO) << summary.BriefReport();

    for (size_t i = 0; i < window_size; i++)
    {
        window[i]->set_twb(opt_pose[i].segment<3>(0));
        window[i]->set_Rwb(Eigen::Quaterniond(opt_pose[i][6], opt_pose[i][3], opt_pose[i][4], opt_pose[i][5]));
        window[i]->set_velocity(opt_speed_bias[i].segment<3>(0));
        window[i]->set_Ba(opt_speed_bias[i].segment<3>(3));
        window[i]->set_Bg(opt_speed_bias[i].segment<3>(6));
    }
    return true;
}

bool Optimizer::OptimizeVelovityBias(const std::vector<Frame::Ptr> &window)
{
    LOG(INFO) << "init start";
    size_t window_size = window.size();
    std::vector<Eigen::Matrix<double, 7, 1>> opt_pose(window_size);
    std::vector<Eigen::Matrix<double, 9, 1>> opt_speed_bias(window_size);

    ceres::Problem problem;
    for (size_t i = 0; i < window.size(); i++)
    {
        const Eigen::Quaterniond& Qwb = window[i]->Qwb();
        const Eigen::Vector3d& twb = window[i]->twb();
        const Eigen::Vector3d& velocity = window[i]->velocity();
        const Eigen::Vector3d& ba = window[i]->Ba();
        const Eigen::Vector3d& bg = window[i]->Bg();

        opt_pose[i] << twb.x(), twb.y(), twb.z(), Qwb.x(), Qwb.y(), Qwb.z(), Qwb.w();
        opt_speed_bias[i] << velocity.x(), velocity.y(), velocity.z(), ba.x(), ba.y(), ba.z(), bg.x(), bg.y(), bg.z();
//        PoseLocalManifold* local_para = new PoseLocalManifold();
        ceres::LocalParameterization* local_para = new PoseLocalParameterization();

        problem.AddParameterBlock(opt_pose[i].data(), 7, local_para);
        problem.SetParameterBlockConstant(opt_pose[i].data());
        problem.AddParameterBlock(opt_speed_bias[i].data(), 9);
    }

    for (size_t i = 1; i < window.size(); i++)
    {
        LOG(INFO) << window[i]->integration_->delta_p.transpose();
        problem.AddResidualBlock(new IMUFactor(window[i]->integration_.get()), nullptr, opt_pose[i-1].data(), opt_speed_bias[i-1].data(),
                                 opt_pose[i].data(), opt_speed_bias[i].data());
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

    for (size_t i = 0; i < window_size; i++)
    {
        window[i]->set_velocity(opt_speed_bias[i].segment<3>(0));
        window[i]->set_Ba(opt_speed_bias[i].segment<3>(3));
        window[i]->set_Bg(opt_speed_bias[i].segment<3>(6));

        LOG(INFO) << "Ba" << i << ": " << window[i]->Ba().transpose();
        LOG(INFO) << "v" << i << ": " << window[i]->velocity().transpose();
        LOG(INFO) << "Bg" << i << ": " << window[i]->Bg().transpose();
    }

    return true;
}

bool Optimizer::initVelocityBias(const std::vector<Frame::Ptr> &window)
{
    // solve gyroscope bias
    {
        Eigen::Matrix3d A = Eigen::Matrix3d::Zero();
        Eigen::Vector3d b = Eigen::Vector3d::Zero();
        Eigen::Vector3d delta_bg = Eigen::Vector3d::Zero();
        for (size_t i = 1; i < window.size(); i++)
        {
            const Frame::Ptr & frame_i = window[i-1];
            const Frame::Ptr & frame_j = window[i];

            Eigen::Quaterniond q_ij(frame_i->Qwb().conjugate() * frame_j->Qwb());

            Eigen::Matrix3d tmp_A = frame_j->integration_->jacobian.template block<3, 3>(O_R, O_BG);
            Eigen::Vector3d tmp_b = 2 * (frame_j->integration_->delta_q.inverse() * q_ij).vec();
            A += tmp_A.transpose() * tmp_A;
            b += tmp_A.transpose() * tmp_b;
        }
        delta_bg = A.ldlt().solve(b);

        Eigen::JacobiSVD<Eigen::Matrix3d> svd(A);
        double cond = svd.singularValues()(0) / svd.singularValues()(svd.singularValues().size()-1);
        LOG(INFO) << "conditional number in solving gyroscope bias:" << cond;
        LOG(INFO) << "Init gyroscope bias: " << delta_bg.transpose();

        for(size_t i = 1; i < window.size(); i++)
        {
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
        for(size_t i = 0; i < window.size() - 1; i++)
        {
            const Frame::Ptr & frame_i = window[i];
            const Frame::Ptr & frame_j = window[i+1];

            Eigen::Matrix3d R_bi_w = frame_i->Rwb().transpose();
            double delta_t = frame_j->integration_->dt;

            Eigen::Matrix<double, 6, 6> tmp_A;
            Eigen::Matrix<double, 6, 1> tmp_b;
            tmp_A.block<3, 3>(0, 0) = R_bi_w * delta_t;
            tmp_A.block<3, 3>(0, 3) = Eigen::Matrix3d::Zero();
            tmp_b.block<3, 1>(0, 0) = R_bi_w * (frame_j->twb() - frame_i->twb()) +
                                      R_bi_w * IntegrationBase::G * delta_t * delta_t / 2.0 -
                                      frame_j->integration_->delta_p;

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

        double cond = svd.singularValues()(0) / svd.singularValues()(svd.singularValues().size()-1);
        LOG(INFO) << "conditional number in solving acc bias and velocity:" << cond;

        for (size_t i = 0; i < window.size(); i++)
        {
            window[i]->set_velocity(x.segment<3>(i*3));
            LOG(INFO) << "velocity" << i << ":" << window[i]->velocity().transpose();
        }
    }

    return true;
}