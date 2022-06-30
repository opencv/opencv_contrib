set(OPENCV_MODULE_IS_PART_OF_WORLD FALSE)
set(the_description "SFM algorithms")


### LIBMV LIGHT EXTERNAL DEPENDENCIES ###

find_package(Ceres QUIET)

if(NOT Gflags_FOUND)  # Ceres find gflags on the own, so separate search isn't necessary
  find_package(Gflags QUIET)
endif()
if(NOT (Glog_FOUND OR glog_FOUND))  # Ceres find glog on the own, so separate search isn't necessary
  find_package(Glog QUIET)
endif()

if(NOT Gflags_FOUND OR NOT (Glog_FOUND OR glog_FOUND))
  # try local search scripts
  list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")
  if(NOT Gflags_FOUND)
    find_package(Gflags QUIET)
  endif()
  if(NOT (Glog_FOUND OR glog_FOUND))
    find_package(Glog QUIET)
  endif()
endif()

if(NOT DEFINED GFLAGS_INCLUDE_DIRS AND DEFINED GFLAGS_INCLUDE_DIR)
  set(GFLAGS_INCLUDE_DIRS "${GFLAGS_INCLUDE_DIR}")
endif()
if(NOT GFLAGS_LIBRARIES AND TARGET gflags::gflags)
  set(GFLAGS_LIBRARIES gflags::gflags)
elseif(NOT GFLAGS_LIBRARIES AND TARGET gflags)
  set(GFLAGS_LIBRARIES gflags)
endif()
if(NOT DEFINED GLOG_INCLUDE_DIRS AND DEFINED GLOG_INCLUDE_DIR)
  set(GLOG_INCLUDE_DIRS "${GLOG_INCLUDE_DIR}")
endif()
if(NOT GLOG_LIBRARIES AND TARGET glog::glog)
  set(GLOG_LIBRARIES glog::glog)
endif()

if((gflags_FOUND OR Gflags_FOUND OR GFLAGS_FOUND OR GFLAGS_INCLUDE_DIRS) AND (glog_FOUND OR Glog_FOUND OR GLOG_FOUND OR GLOG_INCLUDE_DIRS))
  set(__cache_key "${GLOG_INCLUDE_DIRS} ~ ${GFLAGS_INCLUDE_DIRS} ~ ${GLOG_LIBRARIES} ~ ${GFLAGS_LIBRARIES}")
  if(NOT DEFINED SFM_GLOG_GFLAGS_TEST_CACHE_KEY OR NOT (SFM_GLOG_GFLAGS_TEST_CACHE_KEY STREQUAL __cache_key))
    set(__fname "${CMAKE_CURRENT_LIST_DIR}/cmake/checks/check_glog_gflags.cpp")
    try_compile(
        SFM_GLOG_GFLAGS_TEST "${CMAKE_BINARY_DIR}" "${__fname}"
        CMAKE_FLAGS "-DINCLUDE_DIRECTORIES:STRING=${GLOG_INCLUDE_DIRS};${GFLAGS_INCLUDE_DIRS}"
        LINK_LIBRARIES ${GLOG_LIBRARIES} ${GFLAGS_LIBRARIES}
        OUTPUT_VARIABLE __output
    )
    if(NOT SFM_GLOG_GFLAGS_TEST)
      file(APPEND ${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/CMakeError.log
          "Failed compilation check: ${__fname}\n"
          "${__output}\n\n"
      )
    endif()
    set(SFM_GLOG_GFLAGS_TEST "${SFM_GLOG_GFLAGS_TEST}" CACHE INTERNAL "")
    set(SFM_GLOG_GFLAGS_TEST_CACHE_KEY "${__cache_key}" CACHE INTERNAL "")
    message(STATUS "Checking SFM glog/gflags deps... ${SFM_GLOG_GFLAGS_TEST}")
  endif()
  unset(__cache_key)
  set(SFM_DEPS_OK "${SFM_GLOG_GFLAGS_TEST}")
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
  if(Ceres_VERSION VERSION_LESS 2.0.0)
    list(APPEND LIBMV_LIGHT_INCLUDES "${CERES_INCLUDE_DIRS}")
  endif()
else()
  add_definitions("-DCERES_FOUND=0")
  message(STATUS "CERES support is disabled. Ceres Solver for reconstruction API is required.")
endif()

### CREATE OPENCV SFM MODULE ###

ocv_add_module(sfm
  opencv_core
  opencv_calib3d
  opencv_features2d
  opencv_xfeatures2d
  opencv_imgcodecs
  WRAP python
)

add_definitions(/DGLOG_NO_ABBREVIATED_SEVERITIES)  # avoid ERROR macro conflict in glog (ceres dependency)

if(WIN32)
  # Avoid error due to min/max being already defined as a macro
  add_definitions(-DNOMINMAX)
endif(WIN32)

ocv_warnings_disable(CMAKE_CXX_FLAGS
  -Wundef
  -Wshadow
  -Wsign-compare
  -Wmissing-declarations
  -Wunused-but-set-variable
  -Wunused-parameter
  -Wunused-function
  -Wsuggest-override
)

ocv_include_directories( ${LIBMV_LIGHT_INCLUDES} )
ocv_module_include_directories()

# source files
FILE(GLOB OPENCV_SFM_SRC src/*.cpp)

# define the header files (make the headers appear in IDEs.)
FILE(GLOB OPENCV_SFM_HDRS include/opencv2/sfm.hpp include/opencv2/sfm/*.hpp)

ocv_set_module_sources(HEADERS ${OPENCV_SFM_HDRS}
                       SOURCES ${OPENCV_SFM_SRC})

ocv_create_module()


### BUILD libmv_light ###

if(NOT CMAKE_VERSION VERSION_LESS 2.8.11) # See ocv_target_include_directories() implementation
  if(TARGET ${the_module})
    get_target_property(__include_dirs ${the_module} INCLUDE_DIRECTORIES)
    include_directories(${__include_dirs})
  endif()
endif()
#include_directories(${OCV_TARGET_INCLUDE_DIRS_${the_module}})
add_subdirectory("${CMAKE_CURRENT_LIST_DIR}/src/libmv_light" "${CMAKE_CURRENT_BINARY_DIR}/src/libmv")

ocv_target_link_libraries(${the_module} ${LIBMV_LIGHT_LIBS})


### CREATE OPENCV SFM TESTS ###

ocv_add_accuracy_tests()
if(Ceres_FOUND AND TARGET opencv_test_sfm)
  ocv_target_link_libraries(opencv_test_sfm ${CERES_LIBRARIES})
endif ()


### CREATE OPENCV SFM SAMPLES ###

if(Ceres_FOUND)
  ocv_add_samples(opencv_viz)
endif ()
