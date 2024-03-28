set(the_description "ArUco Marker Detection")
ocv_define_module(aruco opencv_core opencv_imgproc opencv_calib3d opencv_objdetect WRAP python java objc js)

# add test data from samples and tutorials dir to contrib/aruco
ocv_add_testdata(samples/ contrib/aruco FILES_MATCHING PATTERN "*.yml")
ocv_add_testdata(tutorials/ contrib/aruco FILES_MATCHING PATTERN "*.yml")
ocv_add_testdata(tutorials/images/ contrib/aruco FILES_MATCHING PATTERN "*.jpg")
ocv_add_testdata(tutorials/charuco_detection/images contrib/aruco
  FILES_MATCHING PATTERN "*.jpg")
ocv_add_testdata(tutorials/charuco_detection/images contrib/aruco
  FILES_MATCHING PATTERN "*.png")
ocv_add_testdata(tutorials/charuco_diamond_detection/images contrib/aruco
  FILES_MATCHING PATTERN "*.png")
