set(the_description "Object Detection using CNNs")
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# Caffe
find_package(Caffe REQUIRED)
if(CAFFE_FOUND)
  set(HAVE_CAFFE 1)
  message(STATUS "Caffe:   YES")
else(CAFFE_FOUND)
	message(STATUS "Caffe:   NO")
	message(STATUS "Please set the variable `Caffe_DIR`")
endif(CAFFE_FOUND)
include_directories(${Caffe_INCLUDE_DIRS})
list(APPEND DNN_OBJ_LIBRARIES "${Caffe_LIBRARIES}")

# Protobuf
find_package(Protobuf REQUIRED)
if(PROTOBUF_FOUND)
	set(HAVE_PROTOBUF 1)
	message(STATUS "Protobuf:   YES")
else(PROTOBUF_FOUND)
	message(STATUS "Protobuf:   NO")
	message(STATUS "Please set the variable `Protobuf_DIR`")
endif(PROTOBUF_FOUND)
include_directories(${Protobuf_INCLUDE_DIRS})
list(APPEND DNN_OBJ_LIBRARIES "${Protobuf_LIBRARIES}")

# Glog
find_package(Glog REQUIRED)
if(GLOG_FOUND)
	message(STATUS "Glog:   YES")
	set(HAVE_GLOG 1)
else(GLOG_FOUND)
	message(STATUS "Glog:   NO")
endif(GLOG_FOUND)
list(APPEND DNN_OBJ_LIBRARIES "${GLOB_LIBRARIES}")

set(TARGET train_obj_detect)
set(SOURCE_FILES src/train_model.cpp)
ocv_add_executable(${TARGET} ${SOURCE_FILES})
ocv_target_link_libraries(${TARGET} ${DNN_OBJ_LIBRARIES})
set_target_properties(${TARGET} PROPERTIES
                      ARCHIVE_OUTPUT_DIRECTORY ${LIBRARY_OUTPUT_PATH}
                      RUNTIME_OUTPUT_DIRECTORY ${EXECUTABLE_OUTPUT_PATH}
                      OUTPUT_NAME "opencv_train_obj_detect")
