cmake_minimum_required (VERSION 3.0)
project (myproject)
set (CMAKE_CXX_STANDARD 11)

ocv_define_module(alphamat opencv_core
    opencv_imgproc
    opencv_objdetect
    opencv_imgcodecs
    opencv_photo
)

# ----------------------------------------------------------------------------
#  Detect 3rd-party libraries
# ----------------------------------------------------------------------------

# --- Eigen ---
if(WITH_EIGEN AND NOT HAVE_EIGEN)
  message("here")
  find_package(Eigen3 QUIET)

  if(Eigen3_FOUND)
    if(TARGET Eigen3::Eigen)
      # Use Eigen3 imported target if possible
      list(APPEND OPENCV_LINKER_LIBS Eigen3::Eigen)
      set(HAVE_EIGEN 1)
    else()
      if(DEFINED EIGEN3_INCLUDE_DIRS)
        set(EIGEN_INCLUDE_PATH ${EIGEN3_INCLUDE_DIRS})
        set(HAVE_EIGEN 1)
      elseif(DEFINED EIGEN3_INCLUDE_DIR)
        set(EIGEN_INCLUDE_PATH ${EIGEN3_INCLUDE_DIR})
        set(HAVE_EIGEN 1)
      endif()
    endif()
    if(HAVE_EIGEN)
      if(DEFINED EIGEN3_WORLD_VERSION)  # CMake module
        set(EIGEN_WORLD_VERSION ${EIGEN3_WORLD_VERSION})
        set(EIGEN_MAJOR_VERSION ${EIGEN3_MAJOR_VERSION})
        set(EIGEN_MINOR_VERSION ${EIGEN3_MINOR_VERSION})
      else()  # Eigen config file
        set(EIGEN_WORLD_VERSION ${EIGEN3_VERSION_MAJOR})
        set(EIGEN_MAJOR_VERSION ${EIGEN3_VERSION_MINOR})
        set(EIGEN_MINOR_VERSION ${EIGEN3_VERSION_PATCH})
      endif()
    endif()
  endif()

  if(NOT HAVE_EIGEN)
    if(NOT EIGEN_INCLUDE_PATH OR NOT EXISTS "${EIGEN_INCLUDE_PATH}")
      set(__find_paths "")
      set(__find_path_extra_options "")
      if(NOT CMAKE_CROSSCOMPILING)
        list(APPEND __find_paths /opt)
      endif()
      if(DEFINED ENV{EIGEN_ROOT})
        set(__find_paths "$ENV{EIGEN_ROOT}/include")
        list(APPEND __find_path_extra_options NO_DEFAULT_PATH)
      else()
        set(__find_paths ENV ProgramFiles ENV ProgramW6432)
      endif()
      find_path(EIGEN_INCLUDE_PATH "Eigen/Core"
                PATHS ${__find_paths}
                PATH_SUFFIXES include/eigen3 include/eigen2 Eigen/include/eigen3 Eigen/include/eigen2
                DOC "The path to Eigen3/Eigen2 headers"
                ${__find_path_extra_options}
      )
    endif()
    if(EIGEN_INCLUDE_PATH AND EXISTS "${EIGEN_INCLUDE_PATH}")
      ocv_parse_header("${EIGEN_INCLUDE_PATH}/Eigen/src/Core/util/Macros.h" EIGEN_VERSION_LINES EIGEN_WORLD_VERSION EIGEN_MAJOR_VERSION EIGEN_MINOR_VERSION)
      set(HAVE_EIGEN 1)
    endif()
  endif()
endif()


if(HAVE_EIGEN)
  if(EIGEN_INCLUDE_PATH AND EXISTS "${EIGEN_INCLUDE_PATH}")
    ocv_include_directories(SYSTEM ${EIGEN_INCLUDE_PATH})
  endif()
endif()
