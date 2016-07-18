set(the_description "RGBD algorithms")
ocv_warnings_disable(CMAKE_CXX_FLAGS -Wundef)
ocv_define_module(rgbd opencv_core opencv_calib3d opencv_highgui opencv_imgproc WRAP python)
