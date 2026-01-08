cmake_minimum_required(VERSION 2.8)
project(example)
find_package(OpenCV REQUIRED)

set(SOURCES example.cpp)

include_directories(${OpenCV_INCLUDE_DIRS})
add_executable(example ${SOURCES} ${HEADERS})
target_link_libraries(example ${OpenCV_LIBS})
