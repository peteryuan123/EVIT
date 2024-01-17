#!/bin/zsh

rm -f ./lib/libEVIT.so
mkdir -p build_debug
cd build_debug
cmake -DCMAKE_BUILD_TYPE=Debug ..
make -j6

cd ../example/ROS
catkin_make -DCMAKE_BUILD_TYPE=Debug
source devel/setup.zsh

cd ../..