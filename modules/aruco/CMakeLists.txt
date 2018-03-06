set(the_description "ArUco Marker Detection")
ocv_define_module(aruco opencv_core opencv_imgproc opencv_calib3d WRAP python java)
ocv_warnings_disable(CMAKE_CXX_FLAGS /wd4244)