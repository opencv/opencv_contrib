set(the_description "Advanced edge-detection algorithms")
ocv_warnings_disable(CMAKE_CXX_FLAGS -Wundef)
ocv_define_module(ximpgroc opencv_core opencv_imgproc OPTIONAL opencv_highgui)