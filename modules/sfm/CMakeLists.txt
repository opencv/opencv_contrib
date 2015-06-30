set(the_description "SFM algorithms")

ocv_warnings_disable(CMAKE_CXX_FLAGS -Wundef -Winvalid-pch)

# needed since we are linking static libs
set(CMAKE_CXX_FLAGS ${CMAKE_CSS_FLAGS} "-fPIC")


### LIBMV LIGHT DEFINITIONS ###

set(LIBMV_LIGHT_INCLUDES
  src/libmv_light
  src/libmv_light/third_party/gflags
  src/libmv_light/third_party/glog/src
  src/libmv_light/third_party/ssba
  src/libmv_light/third_party/ceres/include
)

set(LIBMV_LIGHT_LIBS
  reconstruction
  correspondence
  camera
  multiview
  multiview_test_data
  numeric
  simple_pipeline
)


### CHECK OPENCV SFM MODULE DEPENDENCIES ###

set(OPENCV_SFM_DEPS opencv_core opencv_ts opencv_calib3d opencv_features2d opencv_viz)

ocv_check_dependencies(${OPENCV_SFM_DEPS})
ocv_include_directories("${OpenCV_SOURCE_DIR}/include/opencv")
ocv_include_modules_recurse(${OPENCV_SFM_DEPS})


### CREATE OPENCV SFM MODULE ###

ocv_add_module(sfm ${OPENCV_SFM_DEPS})
ocv_include_directories( ${LIBMV_LIGHT_INCLUDES})

# build libmv_light
add_subdirectory(src/libmv_light)

# source files
FILE(GLOB OPENCV_SFM_SRC src/*.cpp)

# define the header files (make the headers appear in IDEs.)
FILE(GLOB OPENCV_SFM_HDRS include/opencv2/sfm/*.hpp)

ocv_set_module_sources( HEADERS ${OPENCV_SFM_HDRS} SOURCES ${OPENCV_SFM_SRC} )

ocv_module_include_directories()
ocv_create_module( ${LIBMV_LIGHT_LIBS} )


### OPENCV SFM TESTS ###

ADD_DEFINITIONS(-DTEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/testdata/cv/sfm/")

ocv_add_accuracy_tests()


### OPENCV SFM SAMPLES ###

ocv_add_samples(opencv_viz)