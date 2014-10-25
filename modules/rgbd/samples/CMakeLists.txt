cmake_minimum_required(VERSION 2.8)
project(map_test)
find_package(OpenCV REQUIRED)

set(SOURCES odometry_evaluation.cpp)

include_directories(${OpenCV_INCLUDE_DIRS})
add_executable(odometry_evaluation ${SOURCES} ${HEADERS})
target_link_libraries(odometry_evaluation ${OpenCV_LIBS})
