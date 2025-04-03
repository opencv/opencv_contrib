set(the_description "ArUco Marker Detection")
ocv_define_module(aruco opencv_core opencv_imgproc opencv_calib3d opencv_objdetect WRAP python java objc js)

# Fractal Markers Tests
if(BUILD_TESTS AND HAVE_OPENCV_VIDEOIO)
  add_subdirectory(test)
endif()
