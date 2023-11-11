//
// Created by mpl on 23-11-7.
//

#include "TimeSurface.h"
#include <opencv2/core/eigen.hpp>

using namespace CannyEVIT;

EventCamera::Ptr TimeSurface::event_cam_ = nullptr;
cv::Mat TimeSurface::history_event_ = cv::Mat();
cv::Mat TimeSurface::history_positive_event_ = cv::Mat();
cv::Mat TimeSurface::history_negative_event_ = cv::Mat();

TimeSurface::TimeSurface(double time_stamp, double decay_factor):
time_stamp_(time_stamp), decay_factor_(decay_factor)
{
    processTimeSurface(history_event_, time_stamp, decay_factor, time_surface_, inverse_time_surface_,
                       gradX_inverse_time_surface_, gradY_inverse_time_surface_);

    processTimeSurface(history_positive_event_, time_stamp, decay_factor, time_surface_positive_, inverse_time_surface_positive_,
                       gradX_inverse_time_surface_positive_, gradY_inverse_time_surface_positive_);

    processTimeSurface(history_negative_event_, time_stamp, decay_factor, time_surface_negative_, inverse_time_surface_negative_,
                       gradX_inverse_time_surface_negative_, gradY_inverse_time_surface_negative_);
}

void TimeSurface::processTimeSurface(const cv::Mat &history_event, double time_stamp, double decay_factor,
                                     cv::Mat &time_surface, Eigen::MatrixXd &inverse_time_surface,
                                     Eigen::MatrixXd &inverse_gradX, Eigen::MatrixXd &inverse_gradY)
{
    cv::exp((history_event - time_stamp) / decay_factor, time_surface);
    event_cam_->undistortImage(time_surface, time_surface);
    time_surface = time_surface * 255.0;
    time_surface.convertTo(time_surface, CV_8U);
    cv::GaussianBlur(time_surface, time_surface, cv::Size(5, 5), 0.0);

    cv::Mat inverse_time_surface_cv = 255.0 - time_surface;
    cv::Mat inverse_gradX_cv, inverse_gradY_cv;
    cv::Sobel(inverse_time_surface_cv, inverse_gradX_cv, CV_64F, 1, 0);
    cv::Sobel(inverse_time_surface_cv, inverse_gradY_cv, CV_64F, 0, 1);

    cv::cv2eigen(inverse_time_surface_cv, inverse_time_surface);
    cv::cv2eigen(inverse_gradX_cv, inverse_gradX);
    cv::cv2eigen(inverse_gradY_cv, inverse_gradY);
}

void TimeSurface::drawCloud(CannyEVIT::pCloud cloud, const Eigen::Matrix4d &Twc, const std::string &window_name)
{
    cv::Mat time_surface_clone = time_surface_.clone();
    time_surface_clone.convertTo(time_surface_clone, CV_8UC1);
    cv::cvtColor(time_surface_clone, time_surface_clone, cv::COLOR_GRAY2BGR);

    Eigen::Matrix3d Rwc = Twc.block<3,3>(0, 0);
    Eigen::Vector3d twc = Twc.block<3,1>(0, 3);
    for (size_t i = 0; i < cloud->size(); i++)
    {
        Point pt = cloud->at(i);
        Eigen::Vector3d p(pt.x, pt.y, pt.z);
        Eigen::Vector3d pc = Rwc.transpose() * (p - twc);
        Eigen::Vector2d p2d = event_cam_->World2Cam(pc);
        cv::Point cvpt(p2d.x(), p2d.y());
        cv::circle(time_surface_clone, cvpt, 0, CV_RGB(255, 0, 0), cv::FILLED);
    }
    cv::imshow(window_name, time_surface_clone);
}


void TimeSurface::initTimeSurface(EventCamera::Ptr event_cam)
{
    event_cam_ = event_cam;
    history_event_ = cv::Mat(event_cam->height(), event_cam->width(), CV_64F, 0.0);
    history_positive_event_ = cv::Mat(event_cam->height(), event_cam->width(), CV_64F, 0.0);
    history_negative_event_ = cv::Mat(event_cam->height(), event_cam->width(), CV_64F, 0.0);
}

void TimeSurface::updateHistoryEvent(EventMsg msg)
{
    if (msg.polarity_)
        history_positive_event_.at<double>(msg.y_, msg.x_) = msg.time_stamp_;
    else
        history_negative_event_.at<double>(msg.y_, msg.x_) = msg.time_stamp_;

    history_event_.at<double>(msg.y_, msg.x_) = msg.time_stamp_;
}

