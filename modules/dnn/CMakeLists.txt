cmake_minimum_required(VERSION 2.8)

macro(_dnn_find_protobuf)

find_package( Protobuf )

if(${PROTOBUF_FOUND} AND EXISTS ${PROTOBUF_PROTOC_EXECUTABLE})
  message(STATUS "Found PROTOBUF Compiler: ${PROTOBUF_PROTOC_EXECUTABLE}")
  file(GLOB PROTO_FILES src/*.proto)
  PROTOBUF_GENERATE_CPP(PROTO_HDRS PROTO_SRCS ${PROTO_FILES})
  set(HAVE_PROTOBUF ON)
  add_definitions(-DHAVE_PROTOBUF=1)
else()
  message(STATUS "PROTOBUF not found. Caffe import function will be disabled.")
  set(HAVE_PROTOBUF OFF)
  set(PROTOBUF_LIBRARIES "")
  add_definitions(-DHAVE_PROTOBUF=0)
endif()

endmacro(_dnn_find_protobuf)

if(${CMAKE_PROJECT_NAME} STREQUAL "OpenCV")#build as OpenCV module

set(the_description "Deep neural network module. It allows to load models and to make forward pass")
set(OPENCV_MODULE_IS_PART_OF_WORLD OFF)
_dnn_find_protobuf()

ocv_add_module(dnn opencv_imgproc opencv_core opencv_highgui WRAP python matlab)
ocv_warnings_disable(CMAKE_CXX_FLAGS -Wno-shadow -Wno-parentheses -Wmaybe-uninitialized -Wsign-promo -Wmissing-declarations -Wmissing-prototypes)
ocv_glob_module_sources(${PROTO_SRCS} ${PROTO_HDRS})
ocv_source_group("Src\\protobuf" FILES ${PROTO_SRCS} ${PROTO_HDRS})
ocv_module_include_directories(include src/caffe ${PROTOBUF_INCLUDE_DIR})

ocv_create_module(${PROTOBUF_LIBRARIES})

ocv_add_accuracy_tests()
ocv_add_perf_tests()
ocv_add_samples()

else()#build as standalone module (for development purposes)

project(dnn_standalone)

_dnn_find_protobuf()
find_package( OpenCV 3.0 REQUIRED )

file(GLOB_RECURSE MODULE_SOURCES "src/*.cpp")
file(GLOB_RECURSE MODULE_HEADERS "src/*.hpp" "src/*.h" "include/*.h*")
file(GLOB_RECURSE MODULE_TEST_SOURCES "test/*.cpp" "test/*.hpp" "test/*.h")

include_directories(include ${OpenCV_INCLUDE_DIRS})
add_library(_opencv_dnn SHARED ${MODULE_SOURCES} ${MODULE_HEADERS} ${PROTO_FILES} ${PROTO_HDRS})
add_executable(_opencv_test_dnn ${MODULE_TEST_SOURCES})
target_link_libraries(_opencv_dnn ${OpenCV_LIBS} ${PROTOBUF_LIBRARIES})
target_link_libraries(_opencv_test_dnn _opencv_dnn opencv_ts)
target_include_directories(_opencv_dnn PRIVATE src src/caffe ${PROTOBUF_INCLUDE_DIR} ${CMAKE_CURRENT_BINARY_DIR})

endif()
