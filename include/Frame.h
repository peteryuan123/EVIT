//
// Created by mpl on 23-11-9.
//

#ifndef CANNYEVIT_FRAME_H
#define CANNYEVIT_FRAME_H

#include <memory>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include "TimeSurface.h"
#include "ImuIntegration.h"


namespace CannyEVIT
{
    class Frame
    {
    public:
        typedef std::shared_ptr<Frame> Ptr;
        typedef std::shared_ptr<Frame const> ConstPtr;

    public:
        Frame(TimeSurface::Ptr time_surface_observation, IntegrationBase::Ptr integration, EventCamera::Ptr event_camera_);

    public:
        double time_stamp_;

        TimeSurface::Ptr time_surface_observation_;
        IntegrationBase::Ptr integration_;
        EventCamera::Ptr event_camera_;

        Eigen::Quaterniond Qwb_;
        Eigen::Vector3d twb_;
        Eigen::Matrix4d Twb_;
        Eigen::Vector3d velocity_;
        Eigen::Vector3d acc_bias_, gyr_bias_;


    public:
        // getter
        Eigen::Matrix3d Rwb();
        Eigen::Vector3d twb();
        Eigen::Quaterniond Qwb();
        Eigen::Matrix4d Twb();

        Eigen::Matrix3d Rwc();
        Eigen::Vector3d twc();
        Eigen::Quaterniond Qwc();
        Eigen::Matrix4d Twc();

        Eigen::Vector3d velocity();
        Eigen::Vector3d Ba();
        Eigen::Vector3d Bg();

        //setter

        void set_integration(const IntegrationBase::Ptr& integration);
        void set_timeStamp(double time_stamp);
        void set_Twb(const Eigen::Matrix4d& Twb);
        void set_Twb(const Eigen::Matrix3d& Rwb, const Eigen::Vector3d& twb);
        void set_Twb(const Eigen::Quaterniond & Qwb, const Eigen::Vector3d& twb);
        void set_Rwb(const Eigen::Quaterniond & Qwb);
        void set_Rwb(const Eigen::Matrix3d & Rwb);
        void set_twb(const Eigen::Vector3d & twb);
        void set_velocity(const Eigen::Vector3d& velocity);
        void set_Ba(const Eigen::Vector3d& ba);
        void set_Bg(const Eigen::Vector3d& bg);
    };

}



#endif //CANNYEVIT_FRAME_H
