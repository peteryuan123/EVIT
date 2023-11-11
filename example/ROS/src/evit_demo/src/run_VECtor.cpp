#include "System.h"

#include <iostream>
#include <string>
#include <thread>

#include <ros/ros.h>
#include <rosbag/bag.h>
#include <rosbag/view.h>

#include <prophesee_event_msgs/Event.h>
#include <prophesee_event_msgs/EventArray.h>
#include <image_transport/image_transport.h>
#include <sensor_msgs/Imu.h>

using Event          = prophesee_event_msgs::Event;               //for prophess
using EventArray     = prophesee_event_msgs::EventArray;
//using Event          = dvs_msgs::Event;                              //for dvs
//using EventArray     = dvs_msgs::EventArray;

std::shared_ptr<CannyEVIT::System> pSystem;
std::string bag_path;
double start_time;

void PubImuData(std::string &imu_topic, std::shared_ptr<CannyEVIT::System> pSystem){
  //处理IMU数据至统一格式交给System处理
  rosbag::Bag bag;

  bag.open(bag_path, rosbag::bagmode::Read );

  std::vector<std::string> topics{ imu_topic };

  rosbag::View view( bag, rosbag::TopicQuery( topics ) );

  double timestamp;
  Eigen::Vector3d vAcc;
  Eigen::Vector3d vGyr;

  for ( rosbag::MessageInstance const m : rosbag::View( bag ) )
    {
      std::string topic = m.getTopic();

      if ( topic == imu_topic )
        {
          /* TODO： make the queue_imu_ac, queue_imu_ve */
          sensor_msgs::ImuConstPtr imu = m.instantiate<sensor_msgs::Imu>();

          timestamp = imu->header.stamp.toSec();

          double angular_velocity_x = imu->angular_velocity.x;
          double angular_velocity_y = imu->angular_velocity.y;
          double angular_velocity_z = imu->angular_velocity.z;
          vGyr << angular_velocity_x, angular_velocity_y, angular_velocity_z;

          double linear_acceleration_x = imu->linear_acceleration.x;
          double linear_acceleration_y = imu->linear_acceleration.y;
          double linear_acceleration_z = imu->linear_acceleration.z;
          vAcc<<linear_acceleration_x, linear_acceleration_y, linear_acceleration_z;
          // if (timestamp >= start_time)
               pSystem->GrabImuData(timestamp, vGyr, vAcc);
         
          usleep(5000);//单位：微秒
        }
    }

  bag.close();

}

void PubEventData(std::string &event_topic, std::shared_ptr<CannyEVIT::System> pSystem){
  //处理event数据至统一格式交给System处理
  rosbag::Bag bag;

  bag.open( bag_path, rosbag::bagmode::Read );

  std::vector<std::string> topics{ event_topic };
  //std::vector<std::string> topics{ event_topic, depth_topic, rgb_topic, gt_topic };

  rosbag::View view( bag, rosbag::TopicQuery( topics ) );

  for ( rosbag::MessageInstance const m : rosbag::View( bag ) )
    {
      std::string topic = m.getTopic();

      if ( topic == event_topic )
        {
          EventArray::ConstPtr eap = m.instantiate<EventArray>();
          for ( Event e : eap->events ){
            size_t x = e.x;
            uint16_t y = e.y;
            double   ts = e.ts.toSec();
            bool     p = e.polarity;
            // std::cout <<ts << std::endl;
            // if (ts >= start_time)
              pSystem->GrabEventData( x, y, ts, p );
          } 
          usleep(1000);
        }
    }

  bag.close();
}

// void GrabImu(const sensor_msgs::ImuConstPtr &imu){
  
//   timestamp = imu->header.stamp.toSec();

//   double angular_velocity_x = imu.angular_velocity.x;
//   double angular_velocity_y = imu.angular_velocity.y;
//   double angular_velocity_z = imu.angular_velocity.z;
//   vGyr << angular_velocity_x, angular_velocity_y, angular_velocity_z;

//   double linear_acceleration_x = imu.linear_acceleration.x;
//   double linear_acceleration_y = imu.linear_acceleration.y;
//   double linear_acceleration_z = imu.linear_acceleration.z;
//   vAcc<<linear_acceleration_x, linear_acceleration_y, linear_acceleration_z;

//   pSystem->GrabImuData(timestamp, vGyr, vAcc);
// }

// void GrabEvent(const EventArray::ConstPtr &e){
//   for ( Event e : eap->events )
//     pSystem->GrabEventData( e );
// }


int main( int argc, char **argv )
{
  ros::init( argc, argv, "CannyEVIT_node" );
  ros::NodeHandle nh;

  image_transport::ImageTransport it( nh );
//  std::string config_path( argv[1] );
//
//  INFO("[main]", "Read config file at %s", config_path.c_str());
//  std::cout << std::fixed << std::setprecision(10);
//
//  pSystem.reset( new CannyEVIT::System(config_path ));
//
//  cv::FileStorage fs( config_path, cv::FileStorage::READ );
//  std::string event_topic = fs["event_topic"].string();
//  std::string imu_topic   = fs["imu_topic"].string();
//  bag_path = fs["bag_path"].string();
//  start_time = fs["start_time"];
//
//  //开启三个线程
//  std::thread thd_BackEnd(&CannyEVIT::System::ProcessBackEnd, pSystem);
//
//  //从rosbag中读取数据
//  std::thread thd_PubImuData(PubImuData, std::ref(imu_topic), pSystem);
//
//  std::thread thd_PubEventData(PubEventData, std::ref(event_topic), pSystem);
//
//  std::thread thd_Draw(&CannyEVIT::System::Draw, pSystem);
//
//  thd_BackEnd.join();
//  thd_PubImuData.join();
//  thd_PubEventData.join();
//  thd_Draw.join();

  //直接订阅topic, 无论哪种方式，最后都是统一格式发给system类

  // ros::Subscriber sub_imu = nh.subscribe( imu_topic, 2000, GrabImu);

  // ros::Subscriber sub_event = nh.subscribe( event_topic, 500, GrabImu);


  ROS_INFO( "finished..." );

  ros::shutdown();

  return 0;





  // esvo_plus::offline::setVerbose();

  // esvo_plus::offline::setVisualize();

  //esvo_plus::offline::prelude( config_path );

  //esvo_plus::offline::init( &nh, &it );

  // esvo_plus::offline::dataloading();

  // esvo_plus::offline::main();

  // ROS_INFO( "finished..." );

  // ros::shutdown();

  // return 0;
}