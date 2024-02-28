//
// Created by mpl on 24-1-17.
//

#include "System.h"


void readEventFile(CannyEVIT::System* system, std::string event_path){
  std::ifstream src;
  src.open(event_path);

//  int dummy;
//  src >> dummy >> dummy;

  double time_stamp;
  int x, y;
  bool polar;

  while(src >> time_stamp >> x >> y >> polar){
    system->GrabEventMsg(time_stamp, x, y, polar);
  }
  src.close();
}

void readImuFile(CannyEVIT::System* system, std::string imu_path){
  std::ifstream src;
  src.open(imu_path);

  double time_stamp;
  double accX, accY, accZ;
  double gyrX, gyrY, gyrZ;

  while(src >> time_stamp >> accX >> accY >> accZ >> gyrX >> gyrY >> gyrZ){
    system->GrabImuMsg(time_stamp, accX, accY, accZ, gyrX, gyrY, gyrZ);
  }
  src.close();
}


int main(int argc, char** argv){

  if (argc != 4)
    std::cout << "Usage: ./run_evit [config_path] [event_path】[imu_path]" << std::endl;

  CannyEVIT::System::Ptr system = std::make_shared<CannyEVIT::System>(argv[1]);

  std::thread event_read_thread(&readEventFile, system.get(), argv[2]);
  std::thread imu_read_thread(&readImuFile, system.get(), argv[3]);

  event_read_thread.join();
  imu_read_thread.join();

  while(1);
}