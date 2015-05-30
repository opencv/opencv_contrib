set(the_description "Deep neural netork module. Allow load models and make forward pass")
set(OPENCV_MODULE_IS_PART_OF_WORLD OFF)
ocv_define_module(dnn opencv_imgproc opencv_core opencv_highgui WRAP python)

target_link_libraries(opencv_dnn)
