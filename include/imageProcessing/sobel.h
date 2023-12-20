//
// Created by mpl on 23-12-18.
// THIS IS INTEGRATED FROM POLYVIEW
//
#ifndef CANNYEVIT_SOBEL_H
#define CANNYEVIT_SOBEL_H

#include <stdlib.h>
#include <Eigen/Eigen>
#include <vector>

namespace CannyEVIT::image_processing {

void sobelx_ker(Eigen::ArrayXXd &K);
void sobely_ker(Eigen::ArrayXXd &K);

struct SobelJob {
  size_t _threadIndex;
  std::vector<size_t> *_startIndices;
  std::vector<size_t> *_stopIndices;
  const Eigen::ArrayXXd *_img;
  Eigen::ArrayXXd *_map;
};

struct SobelJob2 {
  size_t _threadIndex;
  std::vector<size_t> *_startIndices;
  std::vector<size_t> *_stopIndices;
  const Eigen::ArrayXXd *_img;
  const Eigen::ArrayXXd *_map_x;
  Eigen::ArrayXXd *_map_y;
  Eigen::ArrayXXd *_map_mag;
};

void sobel(const Eigen::ArrayXXd &img, Eigen::ArrayXXd &map_x, Eigen::ArrayXXd &map_y);
void sobel_mag(const Eigen::ArrayXXd &img, Eigen::ArrayXXd &map_x, Eigen::ArrayXXd &map_y, Eigen::ArrayXXd &map_mag);

void sobel_x_thread(SobelJob &my_job);
void sobel_y_thread(SobelJob &my_job);
void sobel_y_mag_thread(SobelJob2 &my_job);

void sobel_old(const Eigen::ArrayXXd &img, Eigen::ArrayXXd &map_x, Eigen::ArrayXXd &map_y, bool multithreaded = true);

template<typename T>
void sobel_sparse(const Eigen::ArrayXXd &img, T &loci, Eigen::ArrayXXd &map_x, Eigen::ArrayXXd &map_y);

}

#endif //CANNYEVIT_SOBEL_H
