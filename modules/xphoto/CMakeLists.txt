set(the_description "Addon to basic photo module")
set(CMAKE_CXX_FLAGS "-std=c++11")
ocv_define_module(xphoto opencv_core opencv_imgproc opencv_photo WRAP python java)
