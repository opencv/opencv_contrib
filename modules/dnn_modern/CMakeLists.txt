set(the_description "Modern Deep Learning module")

if(${CMAKE_VERSION} VERSION_LESS 3.2)
  message(STATUS "Module opencv_dnn_modern disabled because CMake version is less than 3.2")
  ocv_module_disable(dnn_modern)
  return()
endif()

# Using cmake scripts and modules
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

# ----------------------------------------------------------------------------
# MODULE REQUIREMENTS
# ----------------------------------------------------------------------------

set(TINY_DNN_CPP_PATH "${OpenCV_BINARY_DIR}/3rdparty/tinydnn")
set(TINY_DNN_CPP_ROOT "${TINY_DNN_CPP_PATH}/tiny-dnn-1.0.0a3")
ocv_download(FILENAME "v1.0.0a3.tar.gz"
               HASH "adb1c512e09ca2c7a6faef36f9c53e59"
               URL
                 "${OPENCV_TINY_DNN_URL}"
                 "$ENV{OPENCV_TINY_DNN_URL}"
                 "https://github.com/tiny-dnn/tiny-dnn/archive/"
               DESTINATION_DIR "${TINY_DNN_CPP_PATH}"
               STATUS TINY_DNN_DOWNLOAD_SUCCESS
               ID "tiny-dnn"
               UNPACK RELATIVE_URL)

if(NOT TINY_DNN_DOWNLOAD_SUCCESS)
  message(STATUS "Failed to download tiny-dnn sources")
endif()

find_package(TinyDNN QUIET)

include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)

if(NOT TinyDNN_FOUND)
    message(STATUS "Module opencv_dnn_modern disabled because tiny-dnn was not found")
    ocv_module_disable(dnn_modern)
    return()
elseif(NOT COMPILER_SUPPORTS_CXX11)
    message(STATUS "Module opencv_dnn_modern disabled because your compiler does not support C++11")
    ocv_module_disable(dnn_modern)
    return()
elseif(APPLE_FRAMEWORK OR ANDROID)
    message(STATUS "Module opencv_dnn_modern disabled because you are not under Linux or Win")
    ocv_module_disable(dnn_modern)
    return()
endif()

# ----------------------------------------------------------------------------
# OPTIMIZATION OPTIONS
# ----------------------------------------------------------------------------

option(TINYDNN_USE_SSE    "Build tiny-dnn with SSE library support" ON)
option(TINYDNN_USE_AVX    "Build tiny-dnn with AVX library support" ON)
option(TINYDNN_USE_TBB    "Build tiny-dnn with TBB library support" OFF)
option(TINYDNN_USE_OMP    "Build tiny-dnn with OMP library support" OFF)
option(TINYDNN_USE_NNPACK "Build tiny-dnn with NNPACK library support" OFF)

if(TINYDNN_USE_TBB AND HAVE_TBB)
    add_definitions(-DCNN_USE_TBB)
elseif(NOT TINYDNN_USE_TBB AND
	   TINYDNN_USE_OMP AND HAVE_OPENMP)
    add_definitions(-DCNN_USE_OMP)
endif()

if(TINYDNN_USE_NNPACK)
   find_package(NNPACK REQUIRED)
   add_definitions(-DCNN_USE_NNPACK)
   include_directories(SYSTEM ${NNPACK_INCLUDE_DIR})
   include_directories(SYSTEM ${NNPACK_INCLUDE_DIR}/../third-party/pthreadpool/include)
   list(APPEND REQUIRED_LIBRARIES ${NNPACK_LIB})
endif()

# we need to disable seializer unless we import cereal and we gonna use caffe converter
add_definitions(-DCNN_NO_SERIALIZATION -DCNN_USE_CAFFE_CONVERTER)

# NOTE: In case that proto files already exist,
#       this is not needed anymore.
if(NOT BUILD_PROTOBUF)
  find_package(Protobuf QUIET)
endif()

if(NOT ${Protobuf_FOUND})
  message(STATUS "Module opencv_dnn_modern disabled because Protobuf is not found")
  ocv_module_disable(dnn_modern)
  return()
endif()

if(DEFINED PROTOBUF_PROTOC_EXECUTABLE AND EXISTS ${PROTOBUF_PROTOC_EXECUTABLE})
  execute_process(COMMAND ${PROTOBUF_PROTOC_EXECUTABLE} caffe.proto --cpp_out=./
                  WORKING_DIRECTORY ${TINYDNN_INCLUDE_DIRS}/tiny_dnn/io/caffe)
else()
  message(STATUS "The protocol buffer compiler is not found (PROTOBUF_PROTOC_EXECUTABLE='${PROTOBUF_PROTOC_EXECUTABLE}')")
  ocv_module_disable(dnn_modern)
  return()
endif()

list(APPEND REQUIRED_LIBRARIES ${PROTOBUF_LIBRARIES})
include_directories(SYSTEM ${PROTOBUF_INCLUDE_DIRS})

####
# Setup the compiler options

# set c++ standard to c++11.
# Note: not working on CMake 2.8. We assume that user has
#       a compiler with C++11 support.

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Unix
if(CMAKE_COMPILER_IS_GNUCXX OR MINGW OR
   CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    include(CheckCXXCompilerFlag)
    check_cxx_compiler_flag("-msse3" COMPILER_HAS_SSE_FLAG)
    check_cxx_compiler_flag("-mavx"  COMPILER_HAS_AVX_FLAG)
    check_cxx_compiler_flag("-mavx2" COMPILER_HAS_AVX2_FLAG)
    check_cxx_compiler_flag("-mfma" COMPILER_HAS_AVX2_FLAG)

    # set Streaming SIMD Extension (SSE) instructions
    if(USE_SSE AND COMPILER_HAS_SSE_FLAG)
        add_definitions(-DCNN_USE_SSE)
        set(EXTRA_C_FLAGS "${EXTRA_C_FLAGS} -msse3")
    endif(USE_SSE AND COMPILER_HAS_SSE_FLAG)
    # set Advanced Vector Extensions (AVX)
    if(USE_AVX AND COMPILER_HAS_AVX_FLAG)
        add_definitions(-DCNN_USE_AVX)
        set(EXTRA_C_FLAGS "${EXTRA_C_FLAGS} -mavx")
    endif(USE_AVX AND COMPILER_HAS_AVX_FLAG)
    # set Advanced Vector Extensions 2 (AVX2)
    if(USE_AVX2 AND COMPILER_HAS_AVX2_FLAG)
        add_definitions(-DCNN_USE_AVX2)
        set(EXTRA_C_FLAGS "${EXTRA_C_FLAGS} -mavx2 -mfma -march=core-avx2")
    endif(USE_AVX2 AND COMPILER_HAS_AVX2_FLAG)

    # include extra flags to the compiler
    # TODO: add info about those flags.
    set(EXTRA_C_FLAGS "${EXTRA_C_FLAGS} -Wall -Wpedantic -Wno-narrowing")
    set(EXTRA_C_FLAGS_RELEASE "${EXTRA_C_FLAGS_RELEASE} -O3")
    set(EXTRA_C_FLAGS_DEBUG   "${EXTRA_C_FLAGS_DEBUG} -g3 -pthread")
elseif(MSVC)
    if(USE_SSE)
        add_definitions(-DCNN_USE_SSE)
        set(EXTRA_C_FLAGS "${EXTRA_C_FLAGS} /arch:SSE2")
    endif(USE_SSE)
    if(USE_AVX)
        add_definitions(-DCNN_USE_AVX)
        set(EXTRA_C_FLAGS "${EXTRA_C_FLAGS} /arch:AVX")
    endif(USE_AVX)
    if(USE_AVX2)
        add_definitions(-DCNN_USE_AVX2)
        set(EXTRA_C_FLAGS "${EXTRA_C_FLAGS} /arch:AVX2")
    endif(USE_AVX2)
    # include specific flags for release and debug modes.
    set(EXTRA_C_FLAGS_RELEASE "${EXTRA_C_FLAGS_RELEASE}
        /Ox /Oi /Ot /Oy /GL /fp:fast /GS-")
    set(CMAKE_EXE_LINKER_FLAGS_RELEASE "${CMAKE_EXE_LINKER_FLAGS_RELEASE} /LTCG")
    set(EXTRA_C_FLAGS_DEBUG "${EXTRA_C_FLAGS_DEBUG}")
    set(EXTRA_C_FLAGS "${EXTRA_C_FLAGS} /W4 /bigobj")
    # this is fine
    add_definitions(-D _CRT_SECURE_NO_WARNINGS)
    add_definitions(-D _SCL_SECURE_NO_WARNINGS)
    add_definitions(-D NO_STRICT)
    # prolly powerless with header-only project
    set(EXTRA_C_FLAGS "${EXTRA_C_FLAGS} /MP")
endif()

# ----------------------------------------------------------------------------
# DNN-MODERN MODULE
# ----------------------------------------------------------------------------

ocv_define_module(dnn_modern opencv_core opencv_imgproc WRAP python)
ocv_target_link_libraries(${the_module} ${REQUIRED_LIBRARIES})
ocv_target_include_directories(${the_module} ${TINYDNN_INCLUDE_DIRS})
ocv_warnings_disable(CMAKE_CXX_FLAGS
  -Wnon-virtual-dtor -Wunused-parameter -Wshadow -Wundef
  /wd4265 /wd4100 /wd4458 /wd4244 /wd4456)
