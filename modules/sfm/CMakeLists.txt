set(OPENCV_MODULE_IS_PART_OF_WORLD FALSE)
set(the_description "SFM algorithms")


### LIBMV LIGHT EXTERNAL DEPENDENCIES ###
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")
find_package(Gflags QUIET)
find_package(Ceres QUIET)
if(NOT Ceres_FOUND)  # Looks like Ceres find glog on the own, so separate search isn't necessary
  find_package(Glog QUIET)
endif()

if((gflags_FOUND OR GFLAGS_FOUND OR GFLAGS_INCLUDE_DIRS) AND (glog_FOUND OR GLOG_FOUND OR GLOG_INCLUDE_DIRS))
  set(_fname "${CMAKE_CURRENT_BINARY_DIR}/test_sfm_deps.cpp")
  file(WRITE "${_fname}" "#include <glog/logging.h>\n#include <gflags/gflags.h>\nint main() { (void)(0); return 0; }\n")
  try_compile(SFM_DEPS_OK "${CMAKE_BINARY_DIR}" "${_fname}"
      CMAKE_FLAGS "-DINCLUDE_DIRECTORIES:STRING=${GLOG_INCLUDE_DIRS};${GFLAGS_INCLUDE_DIRS}"
      LINK_LIBRARIES ${GLOG_LIBRARIES} ${GFLAGS_LIBRARIES}
      OUTPUT_VARIABLE OUTPUT
  )
  file(REMOVE "${_fname}")
  message(STATUS "Checking SFM deps... ${SFM_DEPS_OK}")
else()
  set(SFM_DEPS_OK FALSE)
endif()

if(NOT HAVE_EIGEN OR NOT SFM_DEPS_OK)
  set(DISABLE_MSG "Module opencv_sfm disabled because the following dependencies are not found:")
  if(NOT HAVE_EIGEN)
    set(DISABLE_MSG "${DISABLE_MSG} Eigen")
  endif()
  if(NOT SFM_DEPS_OK)
    set(DISABLE_MSG "${DISABLE_MSG} Glog/Gflags")
  endif()
  message(STATUS ${DISABLE_MSG})
  ocv_module_disable(sfm)
endif()


### LIBMV LIGHT DEFINITIONS ###

set(LIBMV_LIGHT_INCLUDES
  "${CMAKE_CURRENT_LIST_DIR}/src/libmv_light"
  "${OpenCV_SOURCE_DIR}/include/opencv"
  "${GLOG_INCLUDE_DIRS}"
  "${GFLAGS_INCLUDE_DIRS}"
)

set(LIBMV_LIGHT_LIBS
  correspondence
  multiview
  numeric
  ${GLOG_LIBRARIES}
  ${GFLAGS_LIBRARIES}
)

if(Ceres_FOUND)
  add_definitions("-DCERES_FOUND=1")
  list(APPEND LIBMV_LIGHT_LIBS simple_pipeline)
  list(APPEND LIBMV_LIGHT_INCLUDES "${CERES_INCLUDE_DIR}")
else()
  add_definitions("-DCERES_FOUND=0")
  message(STATUS "CERES support is disabled. Ceres Solver for reconstruction API is required.")
endif()

### COMPILE WITH C++11 IF CERES WAS COMPILED WITH C++11

if(Ceres_FOUND)
  list (FIND CERES_COMPILED_COMPONENTS "C++11" _index)
  if (${_index} GREATER -1)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
  endif()
endif()

### DEFINE OPENCV SFM MODULE DEPENDENCIES ###

### CREATE OPENCV SFM MODULE ###

ocv_add_module(sfm
  opencv_core
  opencv_calib3d
  opencv_features2d
  opencv_xfeatures2d
  WRAP python
)


ocv_warnings_disable(CMAKE_CXX_FLAGS
  -Wundef
  -Wshadow
  -Wsign-compare
  -Wmissing-declarations
  -Wunused-but-set-variable
  -Wunused-parameter
  -Wunused-function
)

if(UNIX)
  if(CMAKE_COMPILER_IS_GNUCXX OR CV_ICC)
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
  endif()
endif()

ocv_include_directories( ${LIBMV_LIGHT_INCLUDES} )
ocv_module_include_directories()

# source files
FILE(GLOB OPENCV_SFM_SRC src/*.cpp)

# define the header files (make the headers appear in IDEs.)
FILE(GLOB OPENCV_SFM_HDRS include/opencv2/sfm.hpp include/opencv2/sfm/*.hpp)

ocv_set_module_sources(HEADERS ${OPENCV_SFM_HDRS}
                       SOURCES ${OPENCV_SFM_SRC})

ocv_create_module()

# build libmv_light
if(NOT CMAKE_VERSION VERSION_LESS 2.8.11) # See ocv_target_include_directories() implementation
  if(TARGET ${the_module})
    get_target_property(__include_dirs ${the_module} INCLUDE_DIRECTORIES)
    include_directories(${__include_dirs})
  endif()
endif()
include_directories(${OCV_TARGET_INCLUDE_DIRS_${the_module}})
add_subdirectory("${CMAKE_CURRENT_LIST_DIR}/src/libmv_light" "${CMAKE_CURRENT_BINARY_DIR}/src/libmv")

ocv_target_link_libraries(${the_module} ${LIBMV_LIGHT_LIBS})


### CREATE OPENCV SFM TESTS ###

ocv_add_accuracy_tests()


### CREATE OPENCV SFM SAMPLES ###

if(Ceres_FOUND)
  ocv_add_samples(opencv_viz)
endif ()
