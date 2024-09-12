#!/bin/zsh

# build EVIT
rm -f ./lib/libEVIT.so
mkdir -p build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j6

# build demo
cd ../example/ROS
catkin_make -DCMAKE_BUILD_TYPE=Release
source devel/setup.zsh

cd ../..