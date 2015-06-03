set(the_description "Deep neural network module. Allow load models and make forward pass")
set(OPENCV_MODULE_IS_PART_OF_WORLD OFF)
ocv_define_module(dnn opencv_imgproc opencv_core opencv_highgui WRAP python)

option(WITH_PROTOBUF "Build with libprotobuf" ON)


if(NOT WITH_PROTOBUF)
	message(ERROR "libprotobuf required for dnn module")
else()
	message(STATUS "Find protobuf")
	find_package( Protobuf REQUIRED )
	target_include_directories(opencv_dnn PUBLIC ${PROTOBUF_INCLUDE_DIR})
	target_link_libraries(opencv_dnn ${PROTOBUF_LIBRARIES})
	message(STATUS "Protobuf:" ${PROTOBUF_INCLUDE_DIR})
	message(STATUS "Protobuf:" ${PROTOBUF_LIBRARIES})
endif()

target_link_libraries(opencv_dnn)
