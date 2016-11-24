set(the_description "Contributed/Experimental Algorithms for Salient 2D Features Detection")
ocv_define_module(xfeatures2d opencv_core opencv_imgproc opencv_features2d opencv_calib3d opencv_shape opencv_highgui opencv_videoio opencv_ml
                  OPTIONAL opencv_cudaarithm WRAP python java)
include(cmake/download_vgg.cmake)
include(cmake/download_boostdesc.cmake)
