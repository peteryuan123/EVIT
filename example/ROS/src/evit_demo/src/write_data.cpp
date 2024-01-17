//
// Created by mpl on 24-1-12.
//

#include <iostream>
#include <string>
#include <fstream>

#include <ros/ros.h>

#include <prophesee_event_msgs/Event.h>
#include <prophesee_event_msgs/EventArray.h>
#include <image_transport/image_transport.h>
#include <sensor_msgs/Imu.h>

using Event          = prophesee_event_msgs::Event;               //for prophess
using EventArray     = prophesee_event_msgs::EventArray;
//using Event          = dvs_msgs::Event;                              //for dvs
//using EventArray     = dvs_msgs::EventArray;

std::ofstream event_dest;
std::ofstream imu_dest;

void EventCallBack(const EventArray& events_msg)
{
  for (Event e: events_msg.events)
  {
    uint16_t x = e.x;
    uint16_t y = e.y;
    bool p = e.polarity;
    double ts = e.ts.toSec();
    event_dest << ts << " " << x << " " << y << " " << p << std::endl;
  }
  std::cout << "event done" << std::endl;
}

void ImuCallBack(const sensor_msgs::ImuConstPtr& imu_msg)
{

  double timestamp = imu_msg->header.stamp.toSec();

  double gyrX = imu_msg->angular_velocity.x;
  double gyrY = imu_msg->angular_velocity.y;
  double gyrZ = imu_msg->angular_velocity.z;

  double accX = imu_msg->linear_acceleration.x;
  double accY = imu_msg->linear_acceleration.y;
  double accZ = imu_msg->linear_acceleration.z;
  imu_dest << timestamp << " " << accX << " " << accY << " " << accZ << " " << gyrX << " " << gyrY << " " << gyrZ << std::endl;
  std::cout << "imu done" << std::endl;

}


int main( int argc, char **argv)
{
  ros::init( argc, argv, "write_data_node" );
  ros::NodeHandle nh("~");

  std::string imu_topic = "/imu/data";
  std::string event_topic = "/prophesee/left/events";

  event_dest.open("/home/mpl/data/EVIT/offline/events.txt");
  event_dest << std::fixed;
  event_dest << "480 640\n";
  imu_dest.open("/home/mpl/data/EVIT/offline/imu.txt");
  imu_dest << std::fixed;
  ros::Subscriber imu_subscriber = nh.subscribe(imu_topic, 10000, &ImuCallBack, ros::TransportHints().tcpNoDelay());
  ros::Subscriber event_subscriber = nh.subscribe(event_topic, 10000, &EventCallBack, ros::TransportHints().tcpNoDelay());
  ros::spin();
}


