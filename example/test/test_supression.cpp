//
// Created by mpl on 24-1-12.
//

#include <Eigen/Core>
#include <deque>
#include <fstream>
#include <opencv2/opencv.hpp>

class EventMap {
 public:
  size_t window_size_;
  Eigen::MatrixX<std::deque<std::pair<double, bool>>> event_map_;

  EventMap(size_t window_size) : window_size_(window_size) {
    event_map_.resize(480, 640);
  }

  void push(double time_stamp, size_t x, size_t y, bool polar) {
    if (event_map_(x, y).size() >= window_size_)
      event_map_(x, y).pop_front();
    event_map_(x, y).emplace_back(time_stamp, polar);
  }

  void suppress(){
    for (size_t i = 0; i < 480; i++) {
      for (size_t j = 0; j < 640; j++) {
        if (event_map_(i, j).size() < 3)
          continue;
      }
    }


  }

};




void constructTimeSurface(const EventMap &event_map, double time, cv::Mat mat) {
  Eigen::MatrixXd time_surface;
  time_surface.resize(480, 640);
  time_surface.setZero();
  for (size_t i = 0; i < 480; i++) {
    for (size_t j = 0; j < 640; j++) {
      if (event_map.event_map_(i, j).empty())
        continue;
      for (const auto& pair: event_map.event_map_(i, j)){

      }

    }
  }

}

int main(int argc, char **argv) {
  std::string event_file = "/home/mpl/data/EVIT/offline/events.txt";
  std::ifstream src;
  src.open(event_file);
  size_t weight, height;
  src >> weight >> height;

  double time_stamp;
  size_t x, y;
  bool polar;

  size_t counter = 0;
  EventMap event_map(10);
  std::cout << std::fixed;
  while (src >> time_stamp >> x >> y >> polar) {
    if (x == 347 && y == 148)
      std::cout << time_stamp << " " << polar << std::endl;
//    event_map.push(time_stamp, x, y, polar);

  }

}