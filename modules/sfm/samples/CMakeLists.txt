cmake_minimum_required(VERSION 2.8)
project(reconv2v_test)
find_package(OpenCV REQUIRED)

set(SOURCES recon2v.cpp)

include_directories(${OpenCV_INCLUDE_DIRS})
add_executable(recon2v ${SOURCES} ${HEADERS})
target_link_libraries(recon2v ${OpenCV_LIBS})

include_directories(${OpenCV_INCLUDE_DIRS})
add_executable(simple_reconstruction simple_reconstruction.cpp)
target_link_libraries(simple_reconstruction ${OpenCV_LIBS})