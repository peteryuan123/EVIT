//
// Created by mpl on 23-12-18.
//
#include "imageProcessing/image_processing.h"

using namespace CannyEVIT;

void
image_processing::cleanUpMap(
    Eigen::ArrayXXd &map, size_t clipping, double value) {
  size_t rows = map.rows();
  size_t cols = map.cols();

  map.block(0, 0, rows, clipping).fill(0.0f);
  map.block(0, 0, clipping, cols).fill(0.0f);
  map.block(0, cols - clipping, rows, clipping).fill(0.0f);
  map.block(rows - clipping, 0, clipping, cols).fill(0.0f);
}

size_t image_processing::computeOrientationBin(const Eigen::Vector2f &v) {
  int orient = 0;

  if (fabs(v[0]) < 1e-10f) {
    orient = 2;
    if (v[1] < 0)
      orient = 6;
    return orient;
  }

  float tantheta = v[1] / v[0];

  if (tantheta < -0.4142f) {
    if (tantheta < -2.4142f) {
      orient = 2;
      if (v[1] < 0)
        orient = 6;
    } else {
      orient = 3;
      if (v[1] < 0)
        orient = 7;
    }
  } else {
    if (tantheta > 0.4142f) {
      if (tantheta > 2.4142f) {
        orient = 2;
        if (v[1] < 0)
          orient = 6;
      } else {
        orient = 1;
        if (v[1] < 0)
          orient = 5;
      }
    }
  }

  if (orient == 0 && v[0] < 0.0f)
    orient = 4;

  return orient;
}