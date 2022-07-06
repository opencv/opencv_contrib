set(the_description "ArUco Marker Detection")
if(ARM AND CV_GCC AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 7.0)
  # suppress warnings from GCC only on 7.1 and later
  ocv_warnings_disable(CMAKE_CXX_FLAGS -Wno-psabi)
endif()
ocv_define_module(aruco opencv_core opencv_imgproc opencv_calib3d WRAP python java)
ocv_include_directories(${CMAKE_CURRENT_BINARY_DIR})

ocv_add_testdata(samples/ contrib/aruco
    FILES_MATCHING PATTERN "*yml"
)

ocv_add_testdata(tutorials/aruco_detection/images/ contrib/aruco
    FILES_MATCHING PATTERN "singlemarkersoriginal.jpg"
)

ocv_add_testdata(tutorials/aruco_board_detection/images/ contrib/aruco
    FILES_MATCHING PATTERN "gboriginal.png"
)

ocv_add_testdata(tutorials/charuco_detection/images/ contrib/aruco
    FILES_MATCHING PATTERN "choriginal.jpg"
)

ocv_add_testdata(tutorials/charuco_detection/images/ contrib/aruco
    FILES_MATCHING PATTERN "chocclusion_original.jpg"
)

ocv_add_testdata(tutorials/charuco_diamond_detection/images/ contrib/aruco
    FILES_MATCHING PATTERN "diamondmarkers.png"
)

ocv_add_testdata(tutorials/aruco_calibration/images/ contrib/aruco
    FILES_MATCHING REGEX "img_[0-9]+.jpg"
)
