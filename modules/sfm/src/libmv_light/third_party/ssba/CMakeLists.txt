cmake_minimum_required(VERSION 2.4)

project(V3D)

set (V3D_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set (V3D_INCLUDE_DIRS ${V3D_DIR}/.)

# TODO(keir): This may not work on Windows.
add_definitions(-DV3DLIB_ENABLE_SUITESPARSE)

include_directories(${V3D_INCLUDE_DIRS} ${EXTRA_INC_DIRS})
include_directories(../ldl/Include)
include_directories(../colamd/Include)
include_directories(../ufconfig)

IF(CMAKE_COMPILER_IS_GNUCXX)
  SET(CMAKE_CXX_FLAGS "")
ENDIF(CMAKE_COMPILER_IS_GNUCXX)

add_library(V3D
    Math/v3d_linear.h
    Math/v3d_mathutilities.h
    Math/v3d_optimization.h
    Math/v3d_optimization.cpp
    Geometry/v3d_cameramatrix.h
    Geometry/v3d_metricbundle.h
    Geometry/v3d_metricbundle.cpp)
target_link_libraries(V3D ldl colamd)

#add_subdirectory(Apps)

LIBMV_INSTALL_THIRD_PARTY_LIB(V3D)
