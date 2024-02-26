set(the_description "ArUco Marker Detection")
ocv_define_module(aruco opencv_core opencv_imgproc opencv_3d opencv_calib opencv_objdetect WRAP python java objc js)
ocv_include_directories(${CMAKE_CURRENT_BINARY_DIR})

