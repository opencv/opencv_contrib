set(the_description "Extended image processing module. It includes edge-aware filters and etc.")
ocv_define_module(ximgproc opencv_core opencv_imgproc opencv_calib3d opencv_imgcodecs WRAP python)

file(COPY samples/peilin_plane.png DESTINATION ${OpenCV_BINARY_DIR}/bin)
file(COPY samples/peilin_shape.png DESTINATION ${OpenCV_BINARY_DIR}/bin)