set(the_description "Modern Deep Learning module")

# Using cmake scripts and modules
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

# ----------------------------------------------------------------------------
# MODULE REQUIREMENTS
# ----------------------------------------------------------------------------

find_package(TinyCNN)

include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)

if(NOT TinyCNN_FOUND)
    message(STATUS "Module opencv_dnn_modern disabled because tiny-cnn was not found")
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

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

# ----------------------------------------------------------------------------
# OPTIMIZATION OPTIONS
# ----------------------------------------------------------------------------

option(TINYCNN_USE_SSE    "Build tiny-cnn with SSE library support" ON)
option(TINYCNN_USE_AVX    "Build tiny-cnn with AVX library support" ON)
option(TINYCNN_USE_TBB    "Build tiny-cnn with TBB library support" OFF)
option(TINYCNN_USE_OMP    "Build tiny-cnn with OMP library support" OFF)
option(TINYCNN_USE_NNPACK "Build tiny-cnn with NNPACK library support" OFF)

if(TINYCNN_USE_TBB)
    find_package(TBB REQUIRED)
    add_definitions(-DCNN_USE_TBB)
    include_directories(${TBB_INCLUDE_DIRS})
    link_directories(${TBB_LIBRARY_DIRS})
    set(REQUIRED_LIBRARIES ${REQUIRED_LIBRARIES} ${TBB_LIBRARIES})
elseif(TINYCNN_USE_OMP)
    find_package(OpenMP REQUIRED)
    add_definitions(-DCNN_USE_OMP)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

if(TINYCNN_USE_NNPACK)
   find_package(NNPACK REQUIRED)
   add_definitions(-DCNN_USE_NNPACK)
   include_directories(SYSTEM ${NNPACK_INCLUDE_DIR})
   include_directories(SYSTEM ${NNPACK_INCLUDE_DIR}/../third-party/pthreadpool/include)
   set(REQUIRED_LIBRARIES ${REQUIRED_LIBRARIES} ${NNPACK_LIB})
endif()

find_package(Protobuf)
find_package(Caffe)
set(REQUIRED_LIBRARIES ${REQUIRED_LIBRARIES}
                       ${PROTOBUF_LIBRARIES}
                       ${Caffe_LIBRARIES})


# ----------------------------------------------------------------------------
# COMPILER OPTIONS
# ----------------------------------------------------------------------------

if(CMAKE_COMPILER_IS_GNUCXX OR MINGW OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    if(TINYCNN_USE_SSE)
        add_definitions(-DCNN_USE_SSE)
        set(EXTRA_C_FLAGS  " ${EXTRA_C_FLAGS} -msse3 ")
    endif()
    if(TINYCNN_USE_AVX)
        add_definitions(-DCNN_USE_AVX)
        set(EXTRA_C_FLAGS  "${EXTRA_C_FLAGS} -mavx ")
    endif()
    if((NOT TINYCNN_USE_TBB) AND (NOT TINYCNN_USE_OMP))
        set(EXTRA_C_FLAGS " ${EXTRA_C_FLAGS} -pthread ")
    endif()
    set(EXTRA_C_FLAGS " ${EXTRA_C_FLAGS} -std=c++11 -Wall -Wpedantic")
    set(EXTRA_C_FLAGS_RELEASE " -O3 ")
    set(EXTRA_C_FLAGS_DEBUG " -g3 ")
else() # MSVC
    if(TINYCNN_USE_SSE)
        add_definitions(-DCNN_USE_SSE)
        set(EXTRA_C_FLAGS  " ${EXTRA_C_FLAGS} /arch:SSE2 ")
    endif()
    if(TINYCNN_USE_AVX)
        add_definitions(-DCNN_USE_AVX)
        set(EXTRA_C_FLAGS  "${EXTRA_C_FLAGS}  /arch:AVX ")
    endif()
    set(EXTRA_C_FLAGS_RELEASE " /Ox /bigobj ")
    add_definitions(-D _CRT_SECURE_NO_WARNINGS)
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${EXTRA_C_FLAGS}")
set(CMAKE_CXX_FLAGS_RELEASE "${EXTRA_C_FLAGS_RELEASE}")
set(CMAKE_CXX_FLAGS_DEBUG "${EXTRA_C_FLAGS_DEBUG}")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${EXTRA_EXE_LINKER_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS_RELEASE "${CMAKE_EXE_LINKER_FLAGS_RELEASE} ${EXTRA_EXE_LINKER_FLAGS_RELEASE}")
set(CMAKE_EXE_LINKER_FLAGS_DEBUG "${CMAKE_EXE_LINKER_FLAGS_DEBUG} ${EXTRA_EXE_LINKER_FLAGS_DEBUG}")

# ----------------------------------------------------------------------------
# DNN-MODERN MODULE
# ----------------------------------------------------------------------------

ocv_include_directories(${TINYCNN_INCLUDE_DIRS})
ocv_define_module(dnn_modern opencv_core opencv_imgproc opencv_imgcodecs WRAP python)
ocv_target_link_libraries(${the_module} ${REQUIRED_LIBRARIES})