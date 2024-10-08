
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

std::shared_ptr<CannyEVIT::System> mSystem;

void EventCallBack(const EventArray& events_msg)
{
  for (Event e: events_msg.events)
  {
    uint16_t x = e.x;
    uint16_t y = e.y;
    bool p = e.polarity;
    double ts = e.ts.toSec();
    mSystem->GrabEventMsg(ts, x, y, p);
  }
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
  mSystem->GrabImuMsg(timestamp, accX, accY, accZ, gyrX, gyrY, gyrZ);
}


int main( int argc, char **argv)
{
  ros::init( argc, argv, "CannyEVIT_node" );
  ros::NodeHandle nh("~");

  std::string config_path;
  nh.getParam("config_path", config_path);
  std::string imu_topic;
  nh.getParam("imu_topic", imu_topic);
  std::string event_topic;
  nh.getParam("event_topic", event_topic);

  mSystem.reset(new CannyEVIT::System(config_path));

  ros::Subscriber imu_subscriber = nh.subscribe(imu_topic, 200000, &ImuCallBack, ros::TransportHints().tcpNoDelay());
  ros::Subscriber event_subscriber = nh.subscribe(event_topic, 200000, &EventCallBack, ros::TransportHints().tcpNoDelay());
  ros::spin();
}


