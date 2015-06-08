set(the_description "Deep neural network module. Allow load models and make forward pass")
set(OPENCV_MODULE_IS_PART_OF_WORLD OFF)
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

include_directories(include src/caffe)

ocv_add_module(dnn opencv_imgproc opencv_core opencv_highgui WRAP python matlab)
ocv_glob_module_sources(${PROTO_SRCS} ${PROTO_HDRS})
ocv_source_group("Src\\protobuf" FILES ${PROTO_SRCS} ${PROTO_HDRS})
ocv_module_include_directories(${PROTOBUF_INCLUDE_DIR})

ocv_create_module()

target_link_libraries(opencv_dnn ${PROTOBUF_LIBRARIES})
ocv_add_accuracy_tests()
ocv_add_perf_tests()
ocv_add_samples()
