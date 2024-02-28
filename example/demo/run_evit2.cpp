//
// Created by mpl on 24-1-18.
//

#include <Eigen/Core>
#include <fstream>
#include <deque>

#include "EventCamera.h"

Eigen::MatrixX<std::deque<std::pair<double, bool>>> EventMap;

void readEventMap(std::string file_name){
  std::ifstream src;
  src.open(file_name);

  double time_stamp;
  int x, y;
  bool polar;

  while(src >> time_stamp >> x >> y >> polar){
    EventMap(x, y).emplace_back(time_stamp, polar);
  }
  src.close();
}


int main(){
  EventMap.resize(480, 640);
  CannyEVIT::EventCamera::Ptr cam(new CannyEVIT::EventCamera("/home/mpl/data/EVIT/seq2/config_test.yaml"));

}
