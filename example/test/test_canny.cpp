#include <opencv2/opencv.hpp>
#include <Eigen/Eigen>
#include <opencv2/core/eigen.hpp>
#include <time.h>
#include "imageProcessing/canny.h"
#include "imageProcessing/sobel.h"
#include "imageProcessing/distanceField.h"
#include <fstream>
using namespace CannyEVIT;

int counter = 0;
void build_canny(const cv::Mat &img, std::vector<std::pair<int, int>> &edge_pos, cv::Mat &canny_img) {
  std::ofstream dest;
  dest.open("/home/mpl/data/EVIT/result/robot_fast_result/neutral/info.txt");
  Eigen::MatrixXd img_eigen;
  cv::cv2eigen(img, img_eigen);
  Eigen::ArrayXXd grad_x, grad_y, grad_mag;
  image_processing::sobel_mag(img_eigen, grad_x, grad_y, grad_mag);
  image_processing::canny(grad_mag, grad_x, grad_y, edge_pos, 30);
  canny_img = cv::Mat(img.rows, img.cols, CV_8U, cv::Scalar(0));
  for (auto &pos : edge_pos)
    canny_img.at<uint8_t>(pos.first, pos.second) = 255;

  int max_step = 10;
  int step_size = 1;

  cv::Mat filtered_canny_img = cv::Mat(img.rows, img.cols, CV_8UC3, cv::Scalar(0));
  for (auto &pos : edge_pos) {
    int r = pos.first, c = pos.second;
    double gx = grad_x(r, c), gy = grad_y(r, c);
    double dx = gx / sqrt(gx * gx + gy * gy), dy = gy / sqrt(gx * gx + gy * gy);

    dest << r << " " << c << " " << dx << " " << dy << std::endl;

    if (int(r + dy * max_step) >= img_eigen.rows() || int(r + dy * max_step) < 0
        || int(c + dx * max_step) >= img_eigen.cols() || int(c + dx * max_step) < 0)
      continue;
    for (int i = max_step; i >= 0; i -= step_size) {
      int r_sample = r + dy * i, c_sample = c + dx * i;
      double sample_value = img_eigen(r_sample, c_sample);
      dest << sample_value << " ";
    }
//    dest << std::endl;

    cv::Point start(c, r);
    cv::Point end(c + dx * 10, r + dy * 10);
    cv::line(filtered_canny_img, start, end, CV_RGB(255, 0, 0));
    filtered_canny_img.at<cv::Vec3b>(r, c) = cv::Vec3b(255, 255, 255);

  }
  dest.close();
  counter++;
  cv::imshow("grad", filtered_canny_img);
  cv::waitKey(0);
}

int main() {
//  cv::Mat img = cv::imread("/home/mpl/data/EVIT/robot/robot_normal_result/1642661139.406530.jpg", cv::IMREAD_GRAYSCALE);
//  size_t row = img.rows;
//  size_t col = img.cols;
//  cv::imshow("img", img);
//  cv::Mat edge;
//  clock_t start = clock();
//  cv::Canny(img, edge, 10, 40, 3);
//  std::vector<std::pair<int, int>> hl_first;
//  for (int i = 0; i < edge.rows; i++)
//    for(int j = 0; j < edge.cols; j++)
//      if (edge.at<uint8_t>(i, j) == 255)
//        hl_first.emplace_back(i, j);
//  std::cout << "size:" << hl_first.size() << std::endl;
//  clock_t end = clock();
//  double elapsed_time = (double)(end - start) / CLOCKS_PER_SEC;
//  std::cout << "first time:" << elapsed_time << std::endl;
//
//  cv::imshow("cv_edge", edge);

//  cv::Mat grad_x_cv, grad_y_cv;
//  cv::Sobel(img, grad_x_cv, CV_64F, 1, 0);
//  cv::Sobel(img, grad_y_cv, CV_64F, 0, 1);
//  Eigen::MatrixXf grad_x, grad_y, grad_mag;
//  cv::cv2eigen(grad_x_cv, grad_x);
//  cv::cv2eigen(grad_y_cv, grad_y);
//  grad_mag = grad_x.array()*grad_x.array() + grad_y.array()*grad_y.array();
//  std::cout << grad_mag << std::endl;
//  grad_mag = Eigen::sqrt(grad_mag.array());
//  std::cout << grad_mag << std::endl;


  cv::Mat img_neutral =
      cv::imread("/home/mpl/data/EVIT/result/robot_fast_result/neutral/1642661814.716115.jpg", cv::IMREAD_GRAYSCALE);
  cv::Mat img_positive =
      cv::imread("/home/mpl/data/EVIT/result/robot_fast_result/positive/1642661814.716115.jpg", cv::IMREAD_GRAYSCALE);
  cv::Mat img_negative =
      cv::imread("/home/mpl/data/EVIT/result/robot_fast_result/negative/1642661814.716115.jpg", cv::IMREAD_GRAYSCALE);
//  cv::threshold(img_neutral, img_neutral, 50, 255, cv::THRESH_TOZERO);
//  cv::threshold(img_positive, img_positive, 50, 255, cv::THRESH_TOZERO);
//  cv::threshold(img_negative, img_negative, 50, 255, cv::THRESH_TOZERO);

  std::vector<std::pair<int, int>> edge_pos_neutral;
  std::vector<std::pair<int, int>> edge_pos_positive;
  std::vector<std::pair<int, int>> edge_pos_negative;
  cv::Mat edge_neutral, edge_positive, edge_negative;

  build_canny(img_neutral, edge_pos_neutral, edge_neutral);
  build_canny(img_positive, edge_pos_positive, edge_positive);
  build_canny(img_negative, edge_pos_negative, edge_negative);

  cv::imshow("img_neutral", img_neutral);
  cv::imshow("img_positive", img_positive);
  cv::imshow("img_negative", img_negative);

  cv::Mat canny_img = cv::Mat(img_positive.rows, img_positive.cols, CV_8U, cv::Scalar(0));
  for (auto &pos : edge_pos_positive)
    canny_img.at<uint8_t>(pos.first, pos.second) = 255;
  for (auto &pos : edge_pos_negative)
    canny_img.at<uint8_t>(pos.first, pos.second) = 255;
  cv::imshow("edge_neutral", edge_neutral);
  cv::imshow("edge_positive", edge_positive);
  cv::imshow("edge_negative", edge_negative);
  cv::imshow("canny", canny_img);
  cv::waitKey(0);
//  Eigen::ArrayXXd distance_field;
//  image_processing::chebychevDistanceField(row, col, hl, distance_field);
//  cv::Mat distance_field_cv;
//  Eigen::MatrixXd distance_field_matrix = distance_field;
//  cv::eigen2cv(distance_field_matrix, distance_field_cv);
//  std::cout << distance_field_matrix << std::endl;
//  cv::imshow("edge_polyview", edge_polyview);
//  cv::imshow("distance_field_matrix", distance_field_cv);
//  cv::waitKey(0);



//  Eigen::MatrixXf img_eigen;
//  cv::cv2eigen(img, img_eigen);
//  Eigen::MatrixXf grad_x_eigen, grad_y_eigen;
//  image_processing::canny(img_eigen, grad_x_eigen, grad_y_eigen, 1);
//  cv::Mat grad_x, grad_y;
//  cv::eigen2cv(grad_x_eigen, grad_x);
//  cv::eigen2cv(grad_y_eigen, grad_y);
//
//  cv::imshow("grad_x", grad_x);
//  cv::imshow("grad_y", grad_y);
//  cv::waitKey(0);
}