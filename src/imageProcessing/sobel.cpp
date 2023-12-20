//
// Created by mpl on 23-12-18.
//
#include "imageProcessing/sobel.h"
#include "imageProcessing/image_processing.h"
#include "Util.h"
#include <thread>

using namespace CannyEVIT;

void
image_processing::sobelx_ker(Eigen::ArrayXXd &K) {
  K = Eigen::ArrayXXd(3, 3);
  K << -1.0f / 8.0f, 0.0f / 8.0f, 1.0f / 8.0f,
      -2.0f / 8.0f, 0.0f / 8.0f, 2.0f / 8.0f,
      -1.0f / 8.0f, 0.0f / 8.0f, 1.0f / 8.0f;
}

void
image_processing::sobely_ker(Eigen::ArrayXXd &K) {
  K = Eigen::ArrayXXd(3, 3);
  K << -1.0f / 8.0f, -2.0f / 8.0f, -1.0f / 8.0f,
      0.0f / 8.0f, 0.0f / 8.0f, 0.0f / 8.0f,
      1.0f / 8.0f, 2.0f / 8.0f, 1.0f / 8.0f;
}

void
image_processing::sobel(
    const Eigen::ArrayXXd &img,
    Eigen::ArrayXXd &map_x,
    Eigen::ArrayXXd &map_y) {
  size_t rows = img.rows();
  size_t cols = img.cols();

  if (map_x.rows() != rows || map_x.cols() != cols)
    map_x = Eigen::ArrayXXd(rows, cols);
  if (map_y.rows() != rows || map_y.cols() != cols)
    map_y = Eigen::ArrayXXd(rows, cols);

  std::vector<size_t> startIndices;
  std::vector<size_t> stopIndices;
  startIndices.reserve(GNUMBERTHREADS);
  stopIndices.reserve(GNUMBERTHREADS);
  size_t blockWidth = cols / GNUMBERTHREADS;
  for (size_t i = 0; i < GNUMBERTHREADS; i++) {
    startIndices.push_back(i * blockWidth);
    stopIndices.push_back((i + 1) * blockWidth);
  }
  startIndices.front() += 1;
  stopIndices.back() = cols - 1;

  std::vector<SobelJob> jobs(GNUMBERTHREADS);
  for (size_t i = 0; i < GNUMBERTHREADS; i++) {
    jobs[i]._threadIndex = i;
    jobs[i]._startIndices = &startIndices;
    jobs[i]._stopIndices = &stopIndices;
    jobs[i]._img = &img;
    jobs[i]._map = &map_x;
  }

  std::vector<std::thread> sobelFilters_x;
  for (size_t i = 0; i < GNUMBERTHREADS; i++)
    sobelFilters_x.emplace_back(std::bind(sobel_x_thread, jobs[i]));

  for (auto &thread : sobelFilters_x) {
    if (thread.joinable())
      thread.join();
  }

  for (size_t i = 0; i < GNUMBERTHREADS; i++)
    jobs[i]._map = &map_y;

  std::vector<std::thread> sobelFilters_y;
  for (size_t i = 0; i < GNUMBERTHREADS; i++)
    sobelFilters_y.emplace_back(std::bind(sobel_y_thread, jobs[i]));

  for (auto &thread : sobelFilters_y) {
    if (thread.joinable())
      thread.join();
  }

  cleanUpMap(map_x, 1);
  cleanUpMap(map_y, 1);
}

void
image_processing::sobel_mag(
    const Eigen::ArrayXXd &img,
    Eigen::ArrayXXd &map_x,
    Eigen::ArrayXXd &map_y,
    Eigen::ArrayXXd &map_mag) {
  size_t rows = img.rows();
  size_t cols = img.cols();

  if (map_x.rows() != rows || map_x.cols() != cols)
    map_x = Eigen::ArrayXXd(rows, cols);
  if (map_y.rows() != rows || map_y.cols() != cols)
    map_y = Eigen::ArrayXXd(rows, cols);
  if (map_mag.rows() != rows || map_mag.cols() != cols)
    map_mag = Eigen::ArrayXXd(rows, cols);

  std::vector<size_t> startIndices;
  std::vector<size_t> stopIndices;
  startIndices.reserve(GNUMBERTHREADS);
  stopIndices.reserve(GNUMBERTHREADS);
  size_t blockWidth = cols / GNUMBERTHREADS;
  for (size_t i = 0; i < GNUMBERTHREADS; i++) {
    startIndices.push_back(i * blockWidth);
    stopIndices.push_back((i + 1) * blockWidth);
  }
  startIndices.front() += 1;
  stopIndices.back() = cols - 1;

  std::vector<SobelJob> jobs(GNUMBERTHREADS);
  for (size_t i = 0; i < GNUMBERTHREADS; i++) {
    jobs[i]._threadIndex = i;
    jobs[i]._startIndices = &startIndices;
    jobs[i]._stopIndices = &stopIndices;
    jobs[i]._img = &img;
    jobs[i]._map = &map_x;
  }

  std::vector<std::thread> sobelFilters_x;
  for (size_t i = 0; i < GNUMBERTHREADS; i++)
    sobelFilters_x.emplace_back(std::bind(sobel_x_thread, jobs[i]));

  for (auto &thread : sobelFilters_x) {
    if (thread.joinable())
      thread.join();
  }

  std::vector<SobelJob2> jobs2(GNUMBERTHREADS);
  for (size_t i = 0; i < GNUMBERTHREADS; i++) {
    jobs2[i]._threadIndex = i;
    jobs2[i]._startIndices = &startIndices;
    jobs2[i]._stopIndices = &stopIndices;
    jobs2[i]._img = &img;
    jobs2[i]._map_x = &map_x;
    jobs2[i]._map_y = &map_y;
    jobs2[i]._map_mag = &map_mag;
  }

  std::vector<std::thread> sobelFilters_y;
  for (size_t i = 0; i < GNUMBERTHREADS; i++)
    sobelFilters_y.emplace_back(std::bind(sobel_y_mag_thread, jobs2[i]));

  for (auto &thread : sobelFilters_y) {
    if (thread.joinable())
      thread.join();
  }

  cleanUpMap(map_x, 1);
  cleanUpMap(map_y, 1);
  cleanUpMap(map_mag, 1);
}

void
image_processing::sobel_x_thread(SobelJob &my_job) {
  size_t &threadIndex = my_job._threadIndex;
  size_t start = my_job._startIndices->at(threadIndex);
  size_t stop = my_job._stopIndices->at(threadIndex);

  //get the img and the map
  const Eigen::ArrayXXd &img = *(my_job._img);
  Eigen::ArrayXXd &map = *(my_job._map);

  //now compute all computable elements, each time replacing the
  //oldest column with a new one, and rotating the buffer
  Eigen::ArrayXXd temp(img.rows(), 1);

  for (size_t c = start; c < stop; c++) {
    temp = img.col(c + 1);
    temp -= img.col(c - 1);
    for (size_t r = 1; r < img.rows() - 1; r++)
      map(r, c) = (temp(r - 1, 0) + temp(r, 0) + temp(r, 0) + temp(r + 1, 0)) / 8.0f;
  }
}

void
image_processing::sobel_y_thread(SobelJob &my_job) {
  size_t &threadIndex = my_job._threadIndex;
  size_t start = my_job._startIndices->at(threadIndex);
  size_t stop = my_job._stopIndices->at(threadIndex);

  //get the img and the map
  const Eigen::ArrayXXd &img = *(my_job._img);
  Eigen::ArrayXXd &map = *(my_job._map);

  //now compute all computable elements, each time replacing the
  //oldest column with a new one, and rotating the buffer
  Eigen::ArrayXXd temp(img.rows(), 1);

  for (size_t c = start; c < stop; c++) {
    temp = img.col(c);
    temp += img.col(c);
    temp += img.col(c - 1);
    temp += img.col(c + 1);
    for (size_t r = 1; r < img.rows() - 1; r++)
      map(r, c) = (temp(r + 1, 0) - temp(r - 1, 0)) / 8.0f;
  }
}

void
image_processing::sobel_y_mag_thread(SobelJob2 &my_job) {
  size_t &threadIndex = my_job._threadIndex;
  size_t start = my_job._startIndices->at(threadIndex);
  size_t stop = my_job._stopIndices->at(threadIndex);

  //get the img, the x-map and the map
  const Eigen::ArrayXXd &img = *(my_job._img);
  const Eigen::ArrayXXd &map_x = *(my_job._map_x);
  Eigen::ArrayXXd &map_y = *(my_job._map_y);
  Eigen::ArrayXXd &map_mag = *(my_job._map_mag);

  //now compute all computable elements, each time replacing the
  //oldest column with a new one, and rotating the buffer
  Eigen::ArrayXXd temp(img.rows(), 1);
  Eigen::ArrayXXd temp2(img.rows(), 1);

  for (size_t c = start; c < stop; c++) {
    temp = img.col(c);
    temp += img.col(c);
    temp += img.col(c - 1);
    temp += img.col(c + 1);

    temp2 = map_x.col(c);
    temp2 *= map_x.col(c);
    for (size_t r = 1; r < img.rows() - 1; r++) {
      map_y(r, c) = (temp(r + 1, 0) - temp(r - 1, 0)) / 8.0f;
      map_mag(r, c) = sqrt(temp2(r, 0) + map_y(r, c) * map_y(r, c));
    }
  }
}