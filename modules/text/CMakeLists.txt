set(the_description "Text Detection and Recognition")
ocv_define_module(text opencv_ml opencv_imgproc opencv_core opencv_features2d OPTIONAL opencv_highgui WRAP python java)

if(NOT CMAKE_CROSSCOMPILING OR OPENCV_FIND_TESSERACT)
  set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
  find_package(Tesseract QUIET)
  if(Tesseract_FOUND)
    message(STATUS "Tesseract:   YES")
    set(HAVE_TESSERACT 1)
    ocv_include_directories(${Tesseract_INCLUDE_DIR})
    ocv_target_link_libraries(${the_module} ${Tesseract_LIBRARIES})
  else()
    message(STATUS "Tesseract:   NO")
  endif()
endif()

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/text_config.hpp.in
               ${CMAKE_BINARY_DIR}/text_config.hpp @ONLY)

ocv_include_directories(${CMAKE_CURRENT_BINARY_DIR})

ocv_add_testdata(samples/ contrib/text
    FILES_MATCHING PATTERN "*.xml" PATTERN "*.xml.gz" REGEX "scenetext[0-9]+.jpg"
)


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