//
// Created by mpl on 23-11-7.
//

#ifndef EVIT_NEW_TIMESURFACE_H
#define EVIT_NEW_TIMESURFACE_H

#include <opencv2/opencv.hpp>
#include <Eigen/Geometry>
#include <ceres/rotation.h>
#include <ceres/cubic_interpolation.h>
#include "Type.h"
#include "EventCamera.h"

namespace CannyEVIT
{

    class TimeSurface
    {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        typedef std::shared_ptr<TimeSurface> Ptr;
        typedef std::shared_ptr<TimeSurface const> ConstPtr;

        TimeSurface(double time_stamp, double decay_fator);

        void processTimeSurface(const cv::Mat& history_event, double time_stamp, double decay_factor,
                                cv::Mat& time_surface, Eigen::MatrixXd& inverse_time_surface,
                                Eigen::MatrixXd& inverse_gradX, Eigen::MatrixXd&  inverse_gradY);
        void drawCloud(pCloud cloud, const Eigen::Matrix4d& Twc, const std::string& window_name);

        // for compute residual
        bool isValidPatch(Eigen::Vector2d &patchCentreCoord, Eigen::MatrixXi &mask, size_t wx, size_t wy);
        bool patchInterpolation(const Eigen:: MatrixXd &img, const Eigen::Vector2d &location, int wx, int wy,
                                Eigen::MatrixXd &patch, bool debug);

        Eigen::VectorXd evaluate(const Point& p_w, const Eigen::Quaterniond& Qwb,
                                 const Eigen::Vector3d& twb, int wx, int wy);
        Eigen::MatrixXd df(const Point& p_w, const Eigen::Quaterniond& Qwb, const Eigen::Vector3d& twb, int wx, int wy);


    public:
        double time_stamp_;
        double decay_factor_;

        cv::Mat time_surface_;
        cv::Mat time_surface_positive_;
        cv::Mat time_surface_negative_;

        Eigen::MatrixXd inverse_time_surface_;
        Eigen::MatrixXd inverse_time_surface_positive_;
        Eigen::MatrixXd inverse_time_surface_negative_;

        Eigen::MatrixXd gradX_inverse_time_surface_;
        Eigen::MatrixXd gradY_inverse_time_surface_;
        Eigen::MatrixXd gradX_inverse_time_surface_positive_;
        Eigen::MatrixXd gradY_inverse_time_surface_positive_;
        Eigen::MatrixXd gradX_inverse_time_surface_negative_;
        Eigen::MatrixXd gradY_inverse_time_surface_negative_;

    public:
        static void initTimeSurface(EventCamera::Ptr event_cam);
        static void updateHistoryEvent(EventMsg msg);
        static EventCamera::Ptr event_cam_;
        static cv::Mat history_event_;
        static cv::Mat history_positive_event_;
        static cv::Mat history_negative_event_;
    };


}


#endif //EVIT_NEW_TIMESURFACE_H
