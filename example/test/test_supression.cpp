//
// Created by mpl on 24-1-12.
//

#include <Eigen/Core>
#include <deque>
#include <fstream>
#include <opencv2/opencv.hpp>

struct EventType {

  EventType(double time_stamp, size_t x, size_t y, bool polar) : time_stamp(time_stamp), x(x), y(y), polar(polar) {}

  double time_stamp;
  size_t x, y;
  bool polar;
};

class EventMap {
 public:
  size_t window_size_;
  Eigen::MatrixX<std::deque<std::pair<double, bool>>> event_map_;

  EventMap(size_t window_size) : window_size_(window_size) {
    event_map_.resize(480, 640);
  }

  void push(double time_stamp, size_t x, size_t y, bool polar) {
//    if (event_map_(x, y).size() >= window_size_)
//      event_map_(x, y).pop_front();
    event_map_(x, y).emplace_back(time_stamp, polar);
  }

  void suppress() {
    for (size_t i = 0; i < 480; i++) {
      for (size_t j = 0; j < 640; j++) {
        auto &cur_deque = event_map_(i, j);
        if (cur_deque.size() < 3)
          continue;

        double t_first = cur_deque[1].first;
        for (size_t n = 2; n < cur_deque.size(); n++) {
          if (cur_deque[n].second == cur_deque[n - 1].second &&
              cur_deque[n].first - cur_deque[n - 1].first > cur_deque[n - 1].first - cur_deque[n - 2].first &&
              cur_deque[n].first - cur_deque[n - 1].first < 1) {
            cur_deque[n].first = t_first + 1e-6;
          }
          t_first = cur_deque[n].first;
        }
      }
    }
  }

  void print2File(std::string file_path) {

    auto cmp = [](const EventType &first,
                  const EventType &second) {
      return first.time_stamp > second.time_stamp;
    };
    std::priority_queue<EventType, std::vector<EventType>, decltype(cmp)> p_queue(cmp);

    for (size_t i = 0; i < 480; i++) {
      std::cout << i << "/480" << "\r";
      for (size_t j = 0; j < 640; j++) {
        auto &cur_deque = event_map_(i, j);
        for (size_t n = 0; n < cur_deque.size(); n++) {
          p_queue.emplace(cur_deque[n].first, i, j, cur_deque[n].second);
        }
      }
    }

    std::ofstream dest;
    dest.open(file_path);
    dest << std::fixed;
    while (!p_queue.empty()) {
      std::cout << p_queue.size() << "\r";
      dest << p_queue.top().time_stamp << " " << p_queue.top().x << " " << p_queue.top().y << " " << p_queue.top().polar
           << std::endl;
      p_queue.pop();
    }
    dest.close();
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
      for (const auto &pair : event_map.event_map_(i, j)) {

      }

    }
  }

}

int main(int argc, char **argv) {
  std::string event_file = "/home/mpl/data/EVIT/offline/events.txt";
  std::ifstream src;
  src.open(event_file);
  size_t width, height;
  src >> width >> height;

  double time_stamp;
  size_t x, y;
  bool polar;

  size_t counter = 0;
  EventMap event_map(10);
  std::cout << std::fixed;

  std::cout << "pushing..." << std::endl;
  for (int i = 0; i < 10000000; i++) {
    std::cout << i << "/1000000\r" ;
    src >> time_stamp >> x >> y >> polar;
    event_map.push(time_stamp, x, y, polar);
  }

  std::cout << "supressing..." << std::endl;
  event_map.suppress();

  std::cout << "writing..." << std::endl;
  event_map.print2File("/home/mpl/data/EVIT/offline/events_filtered.txt");

}