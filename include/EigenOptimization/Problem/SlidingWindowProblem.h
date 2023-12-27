//
// Created by mpl on 23-12-25.
//

#ifndef CANNYEVIT_INCLUDE_EIGENOPTIMIZATION_PROBLEM_SLIDINGWINDOWPROBLEM_H_
#define CANNYEVIT_INCLUDE_EIGENOPTIMIZATION_PROBLEM_SLIDINGWINDOWPROBLEM_H_

#include "EigenOptimization/Problem/GenericFunctor.h"
#include "EigenOptimization/Factor/ResidualEventItem.h"
#include "EigenOptimization/Factor/ResidualImuItem.h"
#include "GenericFunctor.h"
#include "Frame.h"
#include "Type.h"
namespace CannyEVIT {

struct SlidingWindowProblemConfig {
  SlidingWindowProblemConfig(size_t patch_size_x = 1,
                             size_t patch_size_y = 1,
                             double huber_threshold = 10,
                             size_t max_registration_points = 3000,
                             size_t max_iteration = 30,
                             size_t thread_num = 4,
                             size_t batch_size = 300,
                             TimeSurface::FieldType field_type = TimeSurface::FieldType::INV_TIME_SURFACE,
                             LossFunctionType loss_function_type = LossFunctionType::Huber,
                             bool predict_polarity = true)
      : patch_size_x_(patch_size_x),
        patch_size_y_(patch_size_y),
        huber_threshold_(huber_threshold),
        max_registration_points_(max_registration_points),
        batch_size_(batch_size),
        thread_num_(thread_num),
        max_iteration_(max_iteration),
        field_type_(field_type),
        loss_function_type_(loss_function_type),
        predict_polarity_(predict_polarity) {}

  size_t patch_size_x_, patch_size_y_;
  double huber_threshold_;
  size_t max_registration_points_;
  size_t batch_size_;
  size_t thread_num_;
  size_t max_iteration_;
  TimeSurface::FieldType field_type_;
  LossFunctionType loss_function_type_;
  bool predict_polarity_;
};

class SlidingWindowProblem : public GenericFunctor<double> {
 public:
  typedef std::shared_ptr<SlidingWindowProblem> Ptr;

  struct Job {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    ResidualEventItems *pvRI;
    const std::vector<Eigen::Quaterniond> *Qwb_vec;
    const std::vector<Eigen::Vector3d> *twb_vec;
    size_t threadID;
  };

  explicit SlidingWindowProblem(const SlidingWindowProblemConfig &config, size_t window_size);
  ~SlidingWindowProblem() = default;

  void setProblem(const std::deque<Frame::Ptr> &window, const pCloud &cloud);
  void nextBatch();

  void addMotionUpdate(const Eigen::MatrixXd &dx);
  void addPerturbation(const Eigen::MatrixXd &x,
                       std::vector<Eigen::Quaterniond> &Qwb_vec,
                       std::vector<Eigen::Vector3d> &twb_vec,
                       std::vector<Eigen::Vector3d> &velocity_vec,
                       std::vector<Eigen::Vector3d> &ba_vec,
                       std::vector<Eigen::Vector3d> &bg_vec) const;

  int operator()(const Eigen::MatrixXd &x, Eigen::VectorXd &fvec) const;
  int df(const Eigen::MatrixXd &x, Eigen::MatrixXd &fjac) const;

  void residualThread(Job &job, Eigen::VectorXd &fvec) const;
  void JacobianThread(Job &job, Eigen::MatrixXd &fjac) const;

  SlidingWindowProblemConfig config_;
  std::vector<Frame::Ptr> window_;
  size_t window_size_;

  pCloud cloud_;
  size_t point_num_;

  ResidualEventItems event_residuals_;
  std::vector<size_t> indices_for_cur_batch_;

  ResidualImuItems imu_residuals_;

  std::vector<Eigen::Quaterniond> opt_Qwb_vec_;
  std::vector<Eigen::Vector3d> opt_twb_vec_;
  std::vector<Eigen::Vector3d> opt_velocity_vec_;
  std::vector<Eigen::Vector3d> opt_ba_vec_;
  std::vector<Eigen::Vector3d> opt_bg_vec_;

  // status
  size_t cur_batch_start_index_;
};

}

#endif //CANNYEVIT_INCLUDE_EIGENOPTIMIZATION_PROBLEM_SLIDINGWINDOWPROBLEM_H_
