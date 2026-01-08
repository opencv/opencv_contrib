cmake_minimum_required(VERSION 2.8)
project(map_test)
find_package(OpenCV REQUIRED)

set(SOURCES map_test.cpp)

add_definitions(-DCOMPARE_FEATURES)
include_directories(${OpenCV_INCLUDE_DIRS})
add_executable(map_test ${SOURCES} ${HEADERS})
target_link_libraries(map_test ${OpenCV_LIBS})
