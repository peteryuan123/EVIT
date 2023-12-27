//
// Created by mpl on 23-11-17.
//

#ifndef CANNYEVIT_EVENTPROBLEM_H
#define CANNYEVIT_EVENTPROBLEM_H

#include "EigenOptimization/Problem/GenericFunctor.h"
#include "EigenOptimization/Factor/ResidualEventItem.h"
#include "Frame.h"
#include "Type.h"

namespace CannyEVIT {

struct EventProblemConfig {
  EventProblemConfig(size_t patch_size_x = 1,
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

class EventProblemLM : public GenericFunctor<double> {
 public:
  typedef std::shared_ptr<EventProblemLM> Ptr;

  struct Job {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    ResidualEventItems *pvRI;
    const Eigen::Quaterniond *Qwb;
    const Eigen::Vector3d *twb;
    size_t threadID;
  };

  explicit EventProblemLM(const EventProblemConfig &config);
  ~EventProblemLM() = default;

  void setProblem(Frame::Ptr frame, const pCloud &cloud);
  void nextBatch();

  int operator()(const Eigen::Matrix<double, 6, 1> &x, Eigen::VectorXd &fvec) const;
  int df(const Eigen::Matrix<double, 6, 1> &x, Eigen::MatrixXd &fjac) const;
  void addMotionUpdate(const Eigen::Matrix<double, 6, 1> &dx);
  void addPerturbation(const Eigen::Matrix<double, 6, 1> &dx, Eigen::Quaterniond &Qwb, Eigen::Vector3d &twb) const;

  void residualThread(Job &job, Eigen::VectorXd &fvec) const;
  void JacobianThread(Job &job, Eigen::MatrixXd &fjac) const;

  pCloud cloud_;
  Frame::Ptr frame_;
  size_t point_num_;
  ResidualEventItems event_residuals_;
  std::vector<size_t> indices_for_cur_batch_;

  EventProblemConfig config_;

  Eigen::Quaterniond opt_Qwb_;
  Eigen::Vector3d opt_twb_;

  // status
  size_t cur_batch_start_index_;
};

}  // namespace CannyEVIT

#endif  // CANNYEVIT_EVENTPROBLEM_H
