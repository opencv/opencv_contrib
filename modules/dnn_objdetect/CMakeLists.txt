set(the_description "Object Detection using CNNs")
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

ocv_define_module(dnn_objdetect opencv_imgcodecs opencv_core opencv_imgproc opencv_dnn)

set(OPENCV_DNN_LIBRARIES opencv_core opencv_imgproc opencv_core opencv_imgcodecs opencv_dnn)
ocv_check_dependencies(${OPENCV_DNN_LIBRARIES})

set(TARGET test_detection)
set(SOURCE_FILES test/obj_detect.cpp)
ocv_add_executable(${TARGET} ${SOURCE_FILES})
ocv_target_link_libraries(${TARGET} ${OPENCV_DNN_LIBRARIES})
set_target_properties(${TARGET} PROPERTIES
                      ARCHIVE_OUTPUT_DIRECTORY ${LIBRARY_OUTPUT_PATH}
                      RUNTIME_OUTPUT_DIRECTORY ${EXECUTABLE_OUTPUT_PATH}
                      OUTPUT_NAME "opencv_dnn_obj_detect")
