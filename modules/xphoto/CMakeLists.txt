set(the_description "Addon to basic photo module")
ocv_warnings_disable(CMAKE_CXX_FLAGS -Wundef)
ocv_define_module(xphoto opencv_core opencv_imgproc OPTIONAL opencv_highgui)