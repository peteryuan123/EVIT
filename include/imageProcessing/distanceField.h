
// THIS IS INTEGRATED FROM POLYVIEW
#ifndef CANNYEVIT_DISTANCEFIELD
#define CANNYEVIT_DISTANCEFIELD

#include <stdlib.h>
#include <Eigen/Eigen>

namespace CannyEVIT::image_processing {

struct TiedPoint {
  int r;
  int c;
  int r_origin;
  int c_origin;

  TiedPoint(int r, int c, int r_origin, int c_origin) :
      r(r), c(c), r_origin(r_origin), c_origin(c_origin) {}
};

void chebychevDistanceField(
    size_t rows, size_t cols,
    const std::vector<std::pair<int, int>> &uv_edge,
    Eigen::ArrayXXd &df,
    size_t radius = 20);

void chamferDistanceField(
    size_t rows, size_t cols,
    const std::vector<std::pair<int, int>> &uv_edge,
    Eigen::ArrayXXd &df,
    size_t radius = 20);

void euclideanDistanceField(
    size_t rows, size_t cols,
    const std::vector<std::pair<int, int>> &uv_edge,
    Eigen::ArrayXXd &df,
    size_t radius = 16);

struct DFjob {
  Eigen::ArrayXXd *_df;
  std::vector<TiedPoint> *_neighbours;
  std::vector<std::pair<int, int> > *_n8;
  std::vector<double> *_distanceLUT;
  size_t _width;
  size_t _rows;
  size_t _cols;
  size_t _radius;
};

void orientedEuclideanDistanceFields(
    size_t rows, size_t cols,
    const std::vector<std::pair<int, int>> &uv_edge,
    const Eigen::ArrayXXd &grad_x,
    const Eigen::ArrayXXd &grad_y,
    std::vector<Eigen::ArrayXXd> &dfs,
    size_t radius = 16);

void orientedEuclideanDistanceFields_thread(
    DFjob &job);

}

#endif /* CANNYEVIT_DISTANCEFIELD */
