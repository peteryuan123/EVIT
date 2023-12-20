// THIS IS INTEGRATED FROM POLYVIEW

#ifndef CANNYEVIT_CANNY
#define CANNYEVIT_CANNY

#include <Eigen/Eigen>
#include <list>
#include <vector>

namespace CannyEVIT::image_processing {

void canny(
    const Eigen::ArrayXXd &grad_mag,
    const Eigen::ArrayXXd &grad_x,
    const Eigen::ArrayXXd &grad_y,
    std::vector<std::pair<int, int>> &hl,
    double threshold);

struct WeakEdgesJob {
  size_t _startIndex;
  size_t _stopIndex;
  const Eigen::ArrayXXd *_grad_mag;
  const Eigen::ArrayXXd *_grad_x;
  const Eigen::ArrayXXd *_grad_y;
  Eigen::ArrayXXi *_mask;
  std::list<std::pair<int, int> > *_weakEdges;
  double _threshold;
  size_t _border;
};

void weakEdges(const Eigen::ArrayXXd &grad_mag, const Eigen::ArrayXXd &grad_x, const Eigen::ArrayXXd &grad_y,
               Eigen::ArrayXXi &mask, double threshold, std::vector<std::list<std::pair<int, int> > > &weakEdgesLists);

void weakEdges_thread(WeakEdgesJob &my_job);

} // namespace CannyEVIT::image_processing


#endif /* CANNYEVIT_CANNY */
