//
// Created by mpl on 23-11-8.
//

#include "EventCamera.h"
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
int main()
{
    CannyEVIT::EventCamera::Ptr cam(new CannyEVIT::EventCamera("/home/mpl/data/EVIT/seq2/config_test.yaml"));
    std::cout << cam->Tbc() << std::endl;
    std::cout << "-----------\n";
    std::cout << cam->Tcb() << std::endl;
    std::cout << "-----------\n";
    std::cout << cam->tbc() << std::endl;
    std::cout << "-----------\n";
    std::cout << cam->tcb() << std::endl;
    std::cout << "-----------\n";
    std::cout << cam->Rbc() << std::endl;
    std::cout << "-----------\n";
    std::cout << cam->Rcb() << std::endl;
    std::cout << "-----------\n";
    std::cout << cam->getProjectionMatrix() << std::endl;
    std::cout << "-----------\n";
    std::cout << cam->height() << std::endl;
    std::cout << "-----------\n";
    std::cout << cam->width() << std::endl;

    cv::Mat mask;
    cv::eigen2cv(cam->getUndistortRectifyMask(), mask);
    cv::imshow("mask", mask);
    cv::waitKey(0);
}