set(the_description "Extended image processing module. It includes edge-aware filters and etc.")
set(OPENCV_MODULE_IS_PART_OF_WORLD OFF)
ocv_define_module(ximgproc opencv_imgproc opencv_core opencv_highgui WRAP python)

target_link_libraries(opencv_ximgproc)

file(COPY samples/peilin_plane.png DESTINATION ${OpenCV_BINARY_DIR}/bin)
file(COPY samples/peilin_shape.png DESTINATION ${OpenCV_BINARY_DIR}/bin)
