#include <thread>

#include "imageProcessing/canny.h"
#include "Util.h"
using namespace CannyEVIT;


void CannyEVIT::image_processing::canny(const Eigen::ArrayXXd &grad_mag,
                                        const Eigen::ArrayXXd &grad_x,
                                        const Eigen::ArrayXXd &grad_y,
                                        std::vector<std::pair<int, int>> &hl,
                                        double threshold) {
  // step 1: detection of weak edges (with non-maximum suppression)
  size_t rows = grad_mag.rows();
  size_t cols = grad_mag.cols();
  Eigen::ArrayXXi mask(rows, cols);
  mask.fill(0);

  std::vector<std::list<std::pair<int, int> > > weakEdgesLists;
  weakEdges(grad_mag, grad_x, grad_y, mask, threshold, weakEdgesLists);

  std::vector<std::pair<int, int> > neighbours;
  neighbours.reserve(8);
  neighbours.push_back(std::pair<int, int>(-1, -1));
  neighbours.push_back(std::pair<int, int>(-1, 0));
  neighbours.push_back(std::pair<int, int>(-1, 1));
  neighbours.push_back(std::pair<int, int>(0, -1));
  neighbours.push_back(std::pair<int, int>(0, 1));
  neighbours.push_back(std::pair<int, int>(1, -1));
  neighbours.push_back(std::pair<int, int>(1, 0));
  neighbours.push_back(std::pair<int, int>(1, 1));

  // step 2: we grow regions and accept them if they are big enough
  for (size_t i = 0; i < weakEdgesLists.size(); i++) {
    std::list<std::pair<int, int> >::iterator it = weakEdgesLists[i].begin();
    while (it != weakEdgesLists[i].end()) {
      int r = it->first;
      int c = it->second;

      if (mask(r, c) == 1) {
        std::shared_ptr<std::list<std::pair<int, int> > > region(new std::list<std::pair<int, int> >());

        std::list<std::pair<int, int> > newElements;
        newElements.push_back(*it);
        mask(r, c) = 0;

        while (newElements.size() > 0) {
          bool wasEmpty = region->empty();
          std::list<std::pair<int, int> >::iterator lastElement = region->end();
          lastElement--;
          region->splice(region->end(), newElements);

          // go through neighbours and add to newElements
          std::list<std::pair<int, int> >::iterator it2;
          if (wasEmpty)
            it2 = region->begin();
          else {
            it2 = lastElement;
            it2++;
          }

          while (it2 != region->end()) {
            for (size_t j = 0; j < neighbours.size(); j++) {
              int r2 = it2->first + neighbours[j].first;
              int c2 = it2->second + neighbours[j].second;

              if (mask(r2, c2) == 1) {
                newElements.push_back(std::pair<int, int>(r2, c2));
                mask(r2, c2) = 0;
              }
            }
            it2++;
          }
        }

        // now check the size of the region
         if( region->size() > 20 )
         {
           for (auto& pos: *region)
             hl.emplace_back(pos);
         }
      }

      it++;
    }
  }
}

void CannyEVIT::image_processing::weakEdges(const Eigen::ArrayXXd &grad_mag, const Eigen::ArrayXXd &grad_x,
                                            const Eigen::ArrayXXd &grad_y, Eigen::ArrayXXi &mask, double threshold,
                                            std::vector<std::list<std::pair<int, int> > > &weakEdgesLists) {
  size_t cols = grad_mag.cols();
  size_t border = 20;

  std::vector<size_t> startIndices;
  std::vector<size_t> stopIndices;
  startIndices.reserve(GNUMBERTHREADS);
  stopIndices.reserve(GNUMBERTHREADS);
  size_t blockWidth = cols / GNUMBERTHREADS;
  for (size_t i = 0; i < GNUMBERTHREADS; i++) {
    startIndices.push_back(i * blockWidth);
    stopIndices.push_back((i + 1) * blockWidth);
  }
  startIndices.front() += border;
  stopIndices.back() = cols - border;

  std::vector<WeakEdgesJob> jobs(GNUMBERTHREADS);
  weakEdgesLists.clear();
  weakEdgesLists.resize(GNUMBERTHREADS);
  for (size_t i = 0; i < GNUMBERTHREADS; i++) {
    jobs[i]._startIndex = startIndices[i];
    jobs[i]._stopIndex = stopIndices[i];
    jobs[i]._grad_mag = &grad_mag;
    jobs[i]._grad_x = &grad_x;
    jobs[i]._grad_y = &grad_y;
    jobs[i]._mask = &mask;
    jobs[i]._weakEdges = &weakEdgesLists[i];
    jobs[i]._threshold = threshold;
    jobs[i]._border = border;
  }

  std::vector<std::thread> weakEdgeFilters;
  for (size_t i = 0; i < GNUMBERTHREADS; i++) weakEdgeFilters.emplace_back(std::bind(weakEdges_thread, jobs[i]));

  for (auto &thread : weakEdgeFilters) {
    if (thread.joinable()) thread.join();
  }
}

void CannyEVIT::image_processing::weakEdges_thread(WeakEdgesJob &my_job) {
  const Eigen::ArrayXXd &grad_mag = *my_job._grad_mag;
  const Eigen::ArrayXXd &grad_x = *my_job._grad_x;
  const Eigen::ArrayXXd &grad_y = *my_job._grad_y;
  Eigen::ArrayXXi &mask = *my_job._mask;
  size_t border = my_job._border;
  double threshold = my_job._threshold;

  size_t rows = grad_mag.rows();
  std::list<std::pair<int, int> > &weakEdges = *my_job._weakEdges;

  for (size_t c = my_job._startIndex; c < my_job._stopIndex; c++) {
    for (size_t r = border; r < rows - border; r++) {
      double gradient = grad_mag(r, c);

      // check if the gradient is above the threshold
      if (gradient > threshold) {
        int orient = 0;
        if (fabs(grad_x(r, c)) < 0.000001f)
          orient = 2;
        else {
          double tantheta = grad_y(r, c) / grad_x(r, c);
          if (tantheta < -0.4142f) {
            if (tantheta < -2.4142f)
              orient = 2;
            else
              orient = 3;
          } else {
            if (tantheta > 0.4142f) {
              if (tantheta > 2.4142f)
                orient = 2;
              else
                orient = 1;
            }
          }
        }

        switch (orient) {
          case 0: {
            if (gradient >= grad_mag(r, c - 1) && gradient >= grad_mag(r, c + 1)) {
              weakEdges.push_back(std::pair<int, int>(r, c));
              mask(r, c) = 1;
            }
            break;
          }
          case 1: {
            if (gradient >= grad_mag(r - 1, c - 1) && gradient >= grad_mag(r + 1, c + 1)) {
              weakEdges.push_back(std::pair<int, int>(r, c));
              mask(r, c) = 1;
            }
            break;
          }
          case 2: {
            if (gradient >= grad_mag(r - 1, c) && gradient >= grad_mag(r + 1, c)) {
              weakEdges.push_back(std::pair<int, int>(r, c));
              mask(r, c) = 1;
            }
            break;
          }
          case 3: {
            if (gradient >= grad_mag(r + 1, c - 1) && gradient >= grad_mag(r - 1, c + 1)) {
              weakEdges.push_back(std::pair<int, int>(r, c));
              mask(r, c) = 1;
            }
            break;
          }
        }
      }
    }
  }
}