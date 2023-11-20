//
// Created by mpl on 23-11-17.
//

#ifndef CANNYEVIT_EVENTPROBLEM_H
#define CANNYEVIT_EVENTPROBLEM_H

#include "EigenOptimization/Factor/GenericFunctor.h"
#include "EigenOptimization/Factor/ResidualEventItem.h"
#include "Frame.h"
#include "Type.h"


namespace CannyEVIT
{

    struct EventProblemConfig
    {
        EventProblemConfig(size_t patch_size_X = 1, size_t patch_size_Y = 1, double huber_threshold = 10,
                           size_t MAX_REGISTRATION_POINTS = 3000, size_t batch_size = 300)
        :patch_size_X_(patch_size_X), patch_size_Y_(patch_size_Y), huber_threshold_(huber_threshold),
        MAX_REGISTRATION_POINTS_(MAX_REGISTRATION_POINTS), batch_size_(batch_size)
        {}

        size_t patch_size_X_, patch_size_Y_;
        double huber_threshold_;
        size_t MAX_REGISTRATION_POINTS_;
        size_t batch_size_;
    };

    class EventProblemLM: public GenericFunctor<double>
    {

    public:
        EventProblemLM(EventProblemConfig config);
        ~EventProblemLM() = default;

        int operator()(const Eigen::Matrix<double, 6, 1>& x, Eigen::VectorXd &fvec) const;
        int df(const Eigen::Matrix<double, 6, 1>& x, Eigen::MatrixXd &fjac) const;
        void addMotionUpdate(const Eigen::Matrix<double, 6, 1> &dx);

        void addPerturbation(Eigen::Quaterniond& Qwb, Eigen::Vector3d& twb, const Eigen::Matrix<double, 6, 1> &x) const;

        void setProblem(Frame::Ptr frame, pCloud cloud);

        EventProblemConfig config_;
        pCloud cloud_;
        Frame::Ptr frame_;
        std::vector<ResidualEventInfo::Ptr> residuals_info_;

        size_t residual_start_index_, residual_end_index_;

        size_t point_num_;
        size_t patch_size_;
        size_t batch_num_;


    };


}



#endif //CANNYEVIT_EVENTPROBLEM_H
