# By default, we use built-in protobuf sources and pre-generated .proto files
# Note: In case of .proto model updates these variables should be used:
# - PROTOBUF_PROTOC_EXECUTABLE (required)
# - PROTOBUF_INCLUDE_DIR
# - PROTOBUF_LIBRARIES or PROTOBUF_LIBRARY / PROTOBUF_LIBRARY_DEBUG for find_package()
OCV_OPTION(BUILD_PROTOBUF "Force to build libprotobuf from sources" ON)
OCV_OPTION(UPDATE_PROTO_FILES "Force to rebuild .proto files" OFF)

if(UPDATE_PROTO_FILES)
  if(NOT DEFINED PROTOBUF_PROTOC_EXECUTABLE)
    find_package(Protobuf QUIET)
  endif()
  if(DEFINED PROTOBUF_PROTOC_EXECUTABLE AND EXISTS ${PROTOBUF_PROTOC_EXECUTABLE})
    message(STATUS "The protocol buffer compiler is found (${PROTOBUF_PROTOC_EXECUTABLE})")
    file(GLOB proto_files src/tensorflow/*.proto)
    list(APPEND proto_files src/caffe/caffe.proto)
    PROTOBUF_GENERATE_CPP(PROTOBUF_HDRS PROTOBUF_SRCS ${proto_files})
  else()
    message(FATAL_ERROR "The protocol buffer compiler is not found (PROTOBUF_PROTOC_EXECUTABLE='${PROTOBUF_PROTOC_EXECUTABLE}')")
  endif()
endif()

if(NOT BUILD_PROTOBUF AND NOT (DEFINED PROTOBUF_INCLUDE_DIR AND DEFINED PROTOBUF_LIBRARIES))
  find_package(Protobuf QUIET)
endif()

if(PROTOBUF_FOUND)
  # nothing
else()
  set(PROTOBUF_CPP_PATH "${OpenCV_BINARY_DIR}/3rdparty/protobuf")
  set(PROTOBUF_CPP_ROOT "${PROTOBUF_CPP_PATH}/protobuf-3.1.0")
  ocv_download(FILENAME "protobuf-cpp-3.1.0.tar.gz"
               HASH "bd5e3eed635a8d32e2b99658633815ef"
               URL
                 "${OPENCV_PROTOBUF_URL}"
                 "$ENV{OPENCV_PROTOBUF_URL}"
                 "https://github.com/google/protobuf/releases/download/v3.1.0/"
               DESTINATION_DIR "${PROTOBUF_CPP_PATH}"
               ID PROTOBUF
               STATUS res
               UNPACK RELATIVE_URL)
  if(NOT res)
    return()
  endif()
  set(PROTOBUF_LIBRARIES libprotobuf)
  set(PROTOBUF_INCLUDE_DIR "${PROTOBUF_CPP_ROOT}/src")
endif()

if(NOT UPDATE_PROTO_FILES)
  file(GLOB fw_srcs ${CMAKE_CURRENT_SOURCE_DIR}/misc/tensorflow/*.cc)
  file(GLOB fw_hdrs ${CMAKE_CURRENT_SOURCE_DIR}/misc/tensorflow/*.h)
  list(APPEND fw_srcs ${CMAKE_CURRENT_SOURCE_DIR}/misc/caffe/caffe.pb.cc)
  list(APPEND fw_hdrs ${CMAKE_CURRENT_SOURCE_DIR}/misc/caffe/caffe.pb.h)
  list(APPEND PROTOBUF_SRCS ${fw_srcs})
  list(APPEND PROTOBUF_HDRS ${fw_hdrs})
  list(APPEND PROTOBUF_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/misc/caffe)
  list(APPEND PROTOBUF_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/misc/tensorflow)
endif()
