
cmake_minimum_required(VERSION 3.0.2)
project(evit_demo)

# To be consistent with the configuration in esvo_core
set(CMAKE_CXX_STANDARD 17)
# SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O3")
set(CMAKE_CXX_FLAGS "-O3 -fopenmp")
add_definitions(-D_LOG_ALL)
find_package(OpenCV 4.2 QUIET)
include_directories(${OpenCV_INCLUDE_DIRS})
if(OpenCV_FOUND)
    add_definitions(-DOPENCV4_FOUND)
else()
    find_package(OpenCV 3 REQUIRED)
    add_definitions(-DOPENCV3_FOUND)
endif()

catkin_package(
 LIBRARIES src
 CATKIN_DEPENDS rosbag roscpp std_msgs sensor_msgs geometry_msgs prophesee_event_msgs image_transport
# DEPENDS system_lib
)

find_package(Pangolin REQUIRED)
include_directories(${Pangolin_INCLUDE_DIRS})

find_package(catkin REQUIRED COMPONENTS
        rosbag
        # tf2_ros
#        cv_bridge
        # pcl_conversions
        roscpp
        std_msgs
        sensor_msgs
        geometry_msgs
        prophesee_event_msgs
        image_transport
        )

include_directories(${catkin_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/../../../../include)
include_directories(${PROJECT_SOURCE_DIR}/../../devel/include)


# add_executable(run_VECtor src/run_VECtor.cpp)
# target_link_libraries(run_VECtor
#         ${catkin_LIBRARIES}
#         ${OpenCV_LIBRARIES}
#         ${PROJECT_SOURCE_DIR}/../../../../lib/libEVIT.so)\

add_executable(write_data src/write_data.cpp)
target_link_libraries(write_data
        ${catkin_LIBRARIES}
)

add_executable(run_vector_new src/run_vector_new.cpp)
target_link_libraries(run_vector_new
        ${catkin_LIBRARIES}
        ${Pangolin_LIBRARIES}
        ${PROJECT_SOURCE_DIR}/../../../../lib/libEVIT.so)