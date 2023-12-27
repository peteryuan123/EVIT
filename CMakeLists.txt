cmake_minimum_required(VERSION 3.1)
project(CannyEVIT VERSION 1.0)

set(CMAKE_CXX_STANDARD 17)
find_package(OpenCV 4.2 QUIET)
include_directories(${OpenCV_INCLUDE_DIRS})
if(OpenCV_FOUND)
    add_definitions(-DOPENCV4_FOUND)
else()
    find_package(OpenCV 3 REQUIRED)
    add_definitions(-DOPENCV3_FOUND)
endif()

add_definitions(-D_LOG_ALL)
set(CMAKE_CXX_FLAGS "-Wall -Wextra -Wno-sign-compare")

find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIRS})

find_package(Pangolin REQUIRED)
include_directories(${Pangolin_INCLUDE_DIRS})

find_package(Ceres REQUIRED)
include_directories(${CERES_INCLUDE_DIRS})

#find_package(g2o REQUIRED)
#include_directories(${G2O_INCLUDE_DIRS})

include_directories(${PROJECT_SOURCE_DIR}/include)

set(SRC_FILE
        src/ImuIntegration.cpp
        src/System.cpp
        src/EventCamera.cpp
        src/TimeSurface.cpp
        src/Frame.cpp
        src/Util.cpp
        src/Optimzer.cpp

        src/CeresFactor/ImuFactor.cpp
        src/CeresFactor/EventFactor.cpp
        src/CeresFactor/EventAutoDIff.cpp

        src/Manifold/PoseLocalParameterization.cpp
        src/Manifold/PoseLocalManifold.cpp

        src/EigenOptimization/Factor/ResidualEventItem.cpp
        src/EigenOptimization/Problem/EventProblem.cpp

        src/imageProcessing/distanceField.cpp
        src/imageProcessing/canny.cpp
        src/imageProcessing/sobel.cpp
        src/imageProcessing/image_processing.cpp
        src/EigenOptimization/Factor/ResidualImuItem.cpp
        src/EigenOptimization/Problem/SlidingWindowProblem.cpp
)


set(INCLUDE_FILE
        include/ImuIntegration.h
        include/Type.h
        include/Util.h
        include/System.h
        include/EventCamera.h
        include/TimeSurface.h
        include/Frame.h
        include/Optimizer.h

        include/CeresFactor/ImuFactor.h
        include/CeresFactor/EventFactor.h
        include/CeresFactor/EventAutoDiffFactor.h

        include/Manifold/PoseLocalParameterization.h
        include/Manifold/PoseLocalManifold.h

        include/EigenOptimization/Factor/ResidualEventItem.h
        include/EigenOptimization/Factor/ResidualImuItem.h
        include/EigenOptimization/Problem/EventProblem.h

        include/imageProcessing/canny.h
        include/imageProcessing/distanceField.h
        include/imageProcessing/sobel.h
        include/imageProcessing/image_processing.h
        include/EigenOptimization/Problem/SlidingWindowProblem.h
)

# library
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)
add_library(EVIT SHARED ${SRC_FILE} ${INCLUDE_FILE})
target_link_libraries(EVIT
        ${OpenCV_LIBRARIES}
        ${PCL_LIBRARIES}
        ${Pangolin_LIBRARIES}
        ${CERES_LIBRARIES}
        )
link_directories(${CMAKE_LIBRARY_OUTPUT_DIRECTORY})


# test
add_executable(test_system
        example/test/test_system_read_param.cpp
)
target_link_libraries(test_system
        EVIT
        )

add_executable(test_normal
        example/test/test_normal.cpp
)
target_link_libraries(test_normal
        EVIT
)

add_executable(test_cam
        example/test/test_cam.cpp
)
target_link_libraries(test_cam
        EVIT
)

add_executable(test_canny
        example/test/test_canny.cpp
)
target_link_libraries(test_canny
        EVIT
)


