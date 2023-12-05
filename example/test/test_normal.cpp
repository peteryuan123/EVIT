#include <iostream>
#include <fstream>
#include <Eigen/Core>
#include "EventCamera.h"
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include <glog/logging.h>

void loadCloud(std::string cloud_path, std::vector<Eigen::Vector<double, 6>>& cloud)
{
    std::ifstream src;
    src.open(cloud_path);
    double x_position, y_position, z_position, x_normal, y_normal, z_normal;
    while(src >> x_position >> y_position >> z_position >> x_normal >> y_normal >> z_normal)
    {
        Eigen::Vector<double, 6> point;
        point << x_position, y_position, z_position, x_normal, y_normal, z_normal;
        cloud.emplace_back(point);
    }
    src.close();
}

int main(int argc, char** argv)
{
    LOG(INFO) << 120378781.2341234;
    Eigen::Matrix3d Rwb;
    Rwb << 0.7938649 ,  0.31754079,  0.51860042,
            -0.33633739,  0.93979121, -0.06057756,
            -0.50661193, -0.12633435,  0.852868061;
    Eigen::Vector3d twb;
    twb << 0.22483599, 0.49789479, 0.05495352;

    Eigen::Matrix3d Rbw = Rwb.transpose();
    Eigen::Vector3d tbw = -Rwb.transpose() * twb;

    CannyEVIT::EventCamera::Ptr cam(new CannyEVIT::EventCamera("/home/mpl/data/EVIT/seq3/config_test.yaml"));
    std::vector<Eigen::Vector<double, 6>> cloud;
    loadCloud(argv[1], cloud);

    cv::Mat img = cv::Mat::zeros(cam->height_, cam->width_, CV_8UC3);

    for (auto point: cloud)
    {
        Eigen::Vector3d pos = point.segment<3>(0);
        Eigen::Vector3d normal_pt = point.segment<3>(3);

        Eigen::Vector3d pos_in_cam = cam->Rcb() *(Rbw * pos + tbw) + cam->tcb();
        Eigen::Vector3d normal_end_in_cam =  cam->Rcb() * (Rbw * normal_pt + tbw) +cam->tcb();

        Eigen::Vector2d pt_2d = cam->World2Cam(pos_in_cam);
        Eigen::Vector2d pt_normal_end_2d = cam->World2Cam(normal_end_in_cam);

        Eigen::Vector2d direction = pt_normal_end_2d - pt_2d;
        direction.normalize();
        Eigen::Vector2d final_end = pt_2d + direction * 5;

        cv::Point cvpt(pt_2d.x(), pt_2d.y());
        cv::Point cvpt_normal_end(final_end.x(), final_end.y());


        cv::circle(img, cvpt, 1, CV_RGB(255, 0, 0), cv::FILLED);
        cv::line(img, cvpt, cvpt_normal_end, CV_RGB(0, 255, 0));
    }
    cv::imshow("test", img);
    cv::waitKey(0);

    return 0;
}