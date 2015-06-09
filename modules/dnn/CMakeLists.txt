cmake_minimum_required(VERSION 2.8)

macro(_dnn_do_protobuf_actions)
option(WITH_PROTOBUF "Build with libprotobuf" ON)

if(NOT WITH_PROTOBUF)
  message(FATAL_ERROR "libprotobuf is required for dnn module")
endif()

find_package( Protobuf REQUIRED )

if(EXISTS ${PROTOBUF_PROTOC_EXECUTABLE})
  message(STATUS "Found PROTOBUF Compiler: ${PROTOBUF_PROTOC_EXECUTABLE}")
else()
  message(FATAL_ERROR "Could not find PROTOBUF Compiler")
endif()

file(GLOB PROTO_FILES src/*.proto)
PROTOBUF_GENERATE_CPP(PROTO_HDRS PROTO_SRCS ${PROTO_FILES})
endmacro(_dnn_do_protobuf_actions)

if(BUILD_opencv_core)#it's build for OpenCV module
_dnn_do_protobuf_actions()
set(the_description "Deep neural network module. Allow load models and make forward pass")
set(OPENCV_MODULE_IS_PART_OF_WORLD OFF)

ocv_add_module(dnn opencv_imgproc opencv_core opencv_highgui WRAP python matlab)
ocv_warnings_disable(CMAKE_CXX_FLAGS -Wno-shadow -Wno-parentheses)
ocv_glob_module_sources(${PROTO_SRCS} ${PROTO_HDRS})
ocv_source_group("Src\\protobuf" FILES ${PROTO_SRCS} ${PROTO_HDRS})
ocv_module_include_directories(${PROTOBUF_INCLUDE_DIR} include src/caffe)

ocv_create_module(${PROTOBUF_LIBRARIES})

ocv_add_accuracy_tests()
ocv_add_perf_tests()
ocv_add_samples()

else()#it's standalone module (development purposes)
project(dnn_standalone)

_dnn_do_protobuf_actions()
find_package( OpenCV 3.0 REQUIRED )

file(GLOB_RECURSE MODULE_SOURCES "src/*.cpp")
file(GLOB_RECURSE MODULE_HEADERS "src/*.hpp" "src/*.h")
file(GLOB_RECURSE MODULE_TEST_SOURCES "test/*.cpp" "test/*.hpp" "test/*.h")

include_directories(include ${OpenCV_INCLUDE_DIRS})
add_library(_opencv_dnn SHARED ${MODULE_SOURCES} ${MODULE_HEADERS} ${PROTO_FILES} ${PROTO_HDRS})
add_executable(_opencv_test_dnn ${MODULE_TEST_SOURCES})
target_link_libraries(_opencv_dnn ${OpenCV_LIBS} ${PROTOBUF_LIBRARIES})
target_link_libraries(_opencv_test_dnn ${OpenCV_LIBS})
target_include_directories(_opencv_dnn PRIVATE src src/caffe ${PROTOBUF_INCLUDE_DIR} ${CMAKE_CURRENT_BINARY_DIR})

endif()