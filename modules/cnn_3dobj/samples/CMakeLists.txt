cmake_minimum_required(VERSION 2.8)
project(cnn_3dobj_test)
find_package(OpenCV REQUIRED)

set(SOURCES cnn_3dobj_demo.cpp)

include_directories(${OpenCV_INCLUDE_DIRS})
add_executable(cnn_3dobj_demo ${SOURCES} ${HEADERS})
target_link_libraries(cnn_3dobj_demo ${OpenCV_LIBS})
