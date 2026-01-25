cmake_minimum_required(VERSION 3.10)
project(map_test)

set(CMAKE_CXX_STANDARD 11)

# Manual include paths - point to opencv source and contrib
set(OPENCV_SRC_DIR "C:/Users/vedshiva/OneDrive/Documents/gsoc/opencv")
set(OPENCV_CONTRIB_DIR "C:/Users/vedshiva/OneDrive/Documents/gsoc/opencv_contrib")

include_directories(
    ${OPENCV_SRC_DIR}/include
    ${OPENCV_SRC_DIR}/modules/core/include
    ${OPENCV_SRC_DIR}/modules/imgproc/include
    ${OPENCV_SRC_DIR}/modules/imgcodecs/include
    ${OPENCV_SRC_DIR}/modules/highgui/include
    ${OPENCV_CONTRIB_DIR}/modules/reg/include
)

set(SOURCES map_test.cpp)
add_definitions(-DCOMPARE_FEATURES)
add_executable(map_test ${SOURCES})

message(STATUS "Build system: Headers configured (linking requires prebuilt OpenCV libraries)")
