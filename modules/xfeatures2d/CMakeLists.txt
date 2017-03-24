set(the_description "Contributed/Experimental Algorithms for Salient 2D Features Detection")
ocv_define_module(xfeatures2d opencv_core opencv_imgproc opencv_features2d opencv_calib3d OPTIONAL opencv_shape opencv_cudaarithm WRAP python java)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/download_vgg.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/download_boostdesc.cmake)
