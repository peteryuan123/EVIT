//
// Created by mpl on 23-11-8.
//

#include "EventCamera.h"
#include <glog/logging.h>
#include <opencv2/core/eigen.hpp>
#include <Eigen/Geometry>

using namespace CannyEVIT;

EventCamera::EventCamera(std::string config_path)
: width_(0), height_(0), K_(Eigen::Matrix3d::Zero()), K_inv_(Eigen::Matrix3d::Zero()),
  P_(Eigen::Matrix<double, 3, 4>::Zero()), distortion_parameter_(cv::Mat()),undistortion_map1_(cv::Mat()),
  undistortion_map2_(cv::Mat()), Tbc_(Eigen::Matrix4d::Identity()), Tcb_(Eigen::Matrix4d::Identity())
{
    cv::FileStorage fs(config_path, cv::FileStorage::READ);
    LOG(INFO) << "--------------EVENT CAMERA--------------";

    if (!fs.isOpened())
        LOG(FATAL) << "Config Not Open !!!";
    else
    {
        LOG(INFO) << "open config file at " << config_path;

        if (fs["Rbc"].isNone())
            LOG(ERROR) << "config: Rbc is not set";
        cv::Mat Rbc = fs["Rbc"].mat();
        Eigen::Matrix3d Rbc_eigen;
        cv::cv2eigen(Rbc, Rbc_eigen);

        if (fs["tbc"].isNone())
            LOG(ERROR) << "config: tbc is not set";
        cv::Mat tbc = fs["tbc"].mat();
        Eigen::Vector3d tbc_eigen;
        cv::cv2eigen(tbc, tbc_eigen);

        Tbc_.block<3, 3>(0, 0) = Rbc_eigen;
        Tbc_.block<3, 1>(0, 3) = tbc_eigen;
        Tcb_.block<3, 3>(0, 0) = Rbc_eigen.transpose();
        Tcb_.block<3, 1>(0, 3) = -Rbc_eigen.transpose() * tbc_eigen;

        if (fs["K"].isNone())
            LOG(ERROR) << "config: K is not set";
        cv::Mat K = fs["K"].mat();
        cv::cv2eigen(K, K_);

        if (fs["height"].isNone())
            LOG(ERROR) << "config: height is not set";
        height_ = fs["height"];

        if (fs["width"].isNone())
            LOG(ERROR) << "config: width is not set";
        width_ = fs["width"];

        if (fs["distortion"].isNone())
            LOG(ERROR) << "config: distortion is not set";
        distortion_parameter_ = fs["distortion"].mat();

        if (fs["distortion_model"].isNone())
            LOG(ERROR) << "config: distortion_model is not set";
        std::string distortion_type = fs["distortion_model"].string();

        if (distortion_type == "plumb_bob")
        {
            distortion_type_ = PLUMB_BOB;
            cv::Mat P;
            if (fs["P"].isNone())
            {
                P = cv::getOptimalNewCameraMatrix(K, distortion_parameter_, cv::Size(width_, height_), 0);
                Eigen::Matrix3d Keigen;
                cv::cv2eigen(P, Keigen);
                P_.block<3, 3>(0, 0) = Keigen;
            }
            else
            {
                P = fs["P"].mat();
                cv::cv2eigen(P, P_);
            }
            // Fix Bug: cannot mix opencv version with ros node, otherwise assertion may appear here
            // make sure opencv version in EVIT is consistent with ros node
            cv::initUndistortRectifyMap(K, distortion_parameter_, cv::Mat::eye(3, 3, CV_32F),
                                        P, cv::Size(width_, height_), CV_32F,
                                        undistortion_map1_, undistortion_map2_);
        }
        else if (distortion_type == "equidistant")
        {
            distortion_type_ = EQUIDISTANT;
            cv::Mat P;
            if (fs["P"].isNone())
            {
                cv::fisheye::estimateNewCameraMatrixForUndistortRectify(
                        K, distortion_parameter_, cv::Size(width_, height_), cv::Mat::eye(3, 3, CV_32F), P, 1);
                Eigen::Matrix3d Keigen;
                cv::cv2eigen(P, Keigen);
                P_.block<3, 3>(0, 0) = Keigen;
            }
            else
            {
                P = fs["P"].mat();
                cv::cv2eigen(P, P_);
            }

            cv::fisheye::initUndistortRectifyMap(K, distortion_parameter_, cv::Mat::eye(3, 3, CV_32F),
                                                 P, cv::Size(width_, height_), CV_32F,
                                                 undistortion_map1_, undistortion_map2_);
        }
        else
            LOG(FATAL) << "[EventCam]:Unspport type:" + distortion_type;

        K_ = P_.block<3, 3>(0, 0);
        K_inv_ = K_.inverse();

        // mask
        cv::Mat undistMask = cv::Mat::ones(height_, width_, CV_32F);
        undistortImage(undistMask, undistMask);
        cv::threshold(undistMask, undistMask, 0.999, 255, cv::THRESH_BINARY);
        undistMask.convertTo(undistMask, CV_8U);
        cv::cv2eigen(undistMask, undistort_recitify_mask_);

        LOG(INFO) << "size(width, height):" << width_ << "," << height_;
        LOG(INFO) << "K:\n" << K_;
        LOG(INFO) << "P:\n" << P_;
        LOG(INFO) << "Distort:" << distortion_parameter_.t();
        LOG(INFO) << "Distort model:" << distortion_type;
        LOG(INFO) << "Tbc:\n" << Tbc_;
        fs.release();
    }


}

void EventCamera::undistortImage(const cv::Mat& src, cv::Mat& dest)
{
#ifdef OPENCV3_FOUND
    cv::remap(src, dest, undistortion_map1_, undistortion_map2_, CV_INTER_LINEAR);
#else
    cv::remap(src, dest, undistortion_map1_, undistortion_map2_, cv::INTER_LINEAR);
#endif

}



