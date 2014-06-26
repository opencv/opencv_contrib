set(the_description "Color balance algorithms")
ocv_warnings_disable(CMAKE_CXX_FLAGS -Wundef)
ocv_define_module(colorbalance opencv_core opencv_imgproc OPTIONAL opencv_highgui)