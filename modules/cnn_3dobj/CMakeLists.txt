set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR})

find_package(Caffe)
if(Caffe_FOUND)
  message(STATUS "Caffe:   YES")
  set(HAVE_CAFFE 1)
else()
  message(STATUS "Caffe:   NO")
endif()

find_package(Protobuf)
if(Protobuf_FOUND)
  message(STATUS "Protobuf:   YES")
  set(HAVE_PROTOBUF 1)
else()
  message(STATUS "Protobuf:   NO")
endif()

find_package(Glog)
if(Glog_FOUND)
  message(STATUS "Glog:   YES")
  set(HAVE_GLOG 1)
else()
  message(STATUS "Glog:   NO")
endif()

find_path(HDF5_INCLUDE_DIR NAMES hdf5.h
    HINTS
    /usr/include/hdf5)
find_library(HDF5_LIBS NAMES hdf5
    HINTS
    /usr/lib)

if(HAVE_CAFFE)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cnn_3dobj_config.hpp.in
               ${CMAKE_CURRENT_SOURCE_DIR}/include/opencv2/cnn_3dobj_config.hpp @ONLY)

include_directories(${CMAKE_CURRENT_BINARY_DIR})

if(${Caffe_FOUND})
  include_directories(${Caffe_INCLUDE_DIR} ${HDF5_INCLUDE_DIR})
endif()
set(the_description "CNN for 3D object recognition and pose estimation including a completed Sphere View on 3D objects")
ocv_define_module(cnn_3dobj opencv_core opencv_imgproc opencv_viz opencv_highgui OPTIONAL WRAP python)

if(${Caffe_FOUND})
  target_link_libraries(opencv_cnn_3dobj ${Caffe_LIBS} ${Glog_LIBS} ${Protobuf_LIBS} ${HDF5_LIBS})
endif()
endif()
