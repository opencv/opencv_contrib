set(the_description "Text Detection and Recognition")
ocv_define_module(text opencv_ml opencv_highgui opencv_imgproc opencv_core opencv_features2d opencv_calib3d WRAP python)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR})

find_package(Tesseract)
if(Tesseract_FOUND)
    message(STATUS "Tesseract:   YES")
    set(HAVE_TESSERACT 1)
else()
    message(STATUS "Tesseract:   NO")
endif()

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/text_config.hpp.in
               ${CMAKE_BINARY_DIR}/text_config.hpp @ONLY)

include_directories(${CMAKE_CURRENT_BINARY_DIR})

if(${Tesseract_FOUND})
include_directories(${Tesseract_INCLUDE_DIR})
endif()

if(${Tesseract_FOUND})
  target_link_libraries(opencv_text ${Tesseract_LIBS})
endif()

#Principal source from which adaptation came is the cnn_3dobj module
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

if(HAVE_CAFFE)
message(STATUS "HAVE CAFFE!!!")
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/text_config.hpp.in
               ${CMAKE_CURRENT_SOURCE_DIR}/include/opencv2/text_config.hpp @ONLY)


include_directories(${CMAKE_CURRENT_BINARY_DIR})

if(${Caffe_FOUND})

  include_directories(${Caffe_INCLUDE_DIR})
  #taken from caffe's cmake
  find_package(HDF5 COMPONENTS HL REQUIRED)
  include_directories(SYSTEM ${HDF5_INCLUDE_DIRS} ${HDF5_HL_INCLUDE_DIR})
  list(APPEND Caffe_LINKER_LIBS ${HDF5_LIBRARIES})
  find_package(Boost 1.46 REQUIRED COMPONENTS system thread filesystem)
  include_directories(SYSTEM ${Boost_INCLUDE_DIR})
  include_directories(SYSTEM /usr/local/cuda-7.5/targets/x86_64-linux/include/)
  list(APPEND Caffe_LINKER_LIBS ${Boost_LIBRARIES})

endif()


if(${Caffe_FOUND})
  #taken from caffe's cmake
  target_link_libraries(opencv_text ${Caffe_LIBS} ${Glog_LIBS} ${Protobuf_LIBS} ${HDF5_LIBRARIES} ${Boost_LIBRARIES})
endif()
endif()

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/text_config.hpp.in
               ${CMAKE_BINARY_DIR}/text_config.hpp @ONLY)

find_package(Qt5Gui)
if(Qt5Gui_FOUND)
    message(STATUS "text module found Qt5Gui:   YES")
    set(HAVE_QT5GUI 1)
  #foreach(dt5_dep Core Gui Widgets Test Concurrent)
  foreach(dt5_dep Gui)
    add_definitions(${Qt5${dt5_dep}_DEFINITIONS})
    include_directories(${Qt5${dt5_dep}_INCLUDE_DIRS})
    #list(APPEND HIGHGUI_LIBRARIES ${Qt5${dt5_dep}_LIBRARIES})
    target_link_libraries(opencv_text ${Qt5${dt5_dep}_LIBRARIES})
  endforeach()

else()
    message(STATUS "text module found Qt5Gui:   NO")
endif()



