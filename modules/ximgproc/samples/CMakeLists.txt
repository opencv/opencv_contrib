cmake_minimum_required(VERSION 2.8)
project(live_demo)
find_package(OpenCV 3.0 REQUIRED)

set(SOURCES live_demo.cpp)

include_directories(${OpenCV_INCLUDE_DIRS})
add_executable(live_demo ${SOURCES} ${HEADERS})
target_link_libraries(live_demo ${OpenCV_LIBS})
