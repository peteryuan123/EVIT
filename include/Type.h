//
// Created by mpl on 23-11-7.
//

#ifndef EVIT_NEW_TYPE_H
#define EVIT_NEW_TYPE_H

#include <memory>
#include <vector>
#include <Eigen/Core>

namespace CannyEVIT {

    struct ImuMsg {
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        typedef std::shared_ptr<ImuMsg> Ptr;
        typedef std::shared_ptr<ImuMsg const> ConstPtr;

        ImuMsg(const double time_stamp, const Eigen::Vector3d &acc, const Eigen::Vector3d &gyr):
        time_stamp_(time_stamp), acc_(acc), gyr_(gyr)
        {}
        double time_stamp_;
        Eigen::Vector3d acc_;
        Eigen::Vector3d gyr_;
    };

    struct EventMsg
    {
        typedef std::shared_ptr<EventMsg> Ptr;
        typedef std::shared_ptr<EventMsg const> ConstPtr;

        EventMsg(double time_stamp, size_t x, size_t y, bool polarity):
        time_stamp_(time_stamp), x_(x), y_(y), polarity_(polarity)
        {}

        double time_stamp_;
        size_t x_, y_;
        bool polarity_;
    };

    struct Point
    {
        typedef std::shared_ptr<Point> Ptr;
        typedef std::shared_ptr<Point const> ConstPtr;

        Point(double x, double y, double z, double x_normal, double y_normal, double z_normal):
                x(x), y(y), z(z), x_normal_(x_normal), y_normal_(y_normal), z_normal_(z_normal)
        {};

        double x;
        double y;
        double z;

        double x_normal_;//梯度相关
        double y_normal_;
        double z_normal_;
    };

    struct FrameData
    {
        FrameData(double time_stamp, std::vector<ImuMsg> imu, std::vector<EventMsg> event):
        time_stamp_(time_stamp), imuData(std::move(imu)), eventData(std::move(event))
        {}

        FrameData():time_stamp_(0), imuData(std::vector<ImuMsg>()), eventData(std::vector<EventMsg>())
        {}
        double time_stamp_;
        std::vector<ImuMsg> imuData;
        std::vector<EventMsg> eventData;
    };

    typedef std::shared_ptr<std::vector<Point>> pCloud; // TODO: this is for temporary, define a cloud structure instead

}
#endif //EVIT_NEW_TYPE_H
