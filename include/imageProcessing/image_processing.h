//
// Created by mpl on 23-12-18.
// THIS IS INTEGRATED FROM POLYVIEW
//

#ifndef CANNYEVIT_INCLUDE_IMAGEPROCESSING_IMAGE_PROCESSING_H_
#define CANNYEVIT_INCLUDE_IMAGEPROCESSING_IMAGE_PROCESSING_H_

#include <stdlib.h>
#include <Eigen/Eigen>

namespace CannyEVIT::image_processing {

void cleanUpMap(Eigen::ArrayXXd &map, size_t clipping, double value = 0.0f);

size_t computeOrientationBin(const Eigen::Vector2f &v);

}

#endif //CANNYEVIT_INCLUDE_IMAGEPROCESSING_IMAGE_PROCESSING_H_
