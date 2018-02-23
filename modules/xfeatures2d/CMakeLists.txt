set(the_description "Contributed/Experimental Algorithms for Salient 2D Features Detection")

ocv_define_module(xfeatures2d opencv_core opencv_imgproc opencv_features2d opencv_calib3d OPTIONAL opencv_shape opencv_ml opencv_cudaarithm WRAP python java)

include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/download_vgg.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/download_boostdesc.cmake)
set(DOWNLOAD_DIR "${OpenCV_BINARY_DIR}/downloads/xfeatures2d")
download_boost_descriptors("${DOWNLOAD_DIR}" boost_status)
download_vgg_descriptors("${DOWNLOAD_DIR}" vgg_status)
if(NOT boost_status OR NOT vgg_status)
  ocv_module_disable(xfeatures2d)
endif()

ocv_module_include_directories("${DOWNLOAD_DIR}")
