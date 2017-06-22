set(the_description "Text Detection and Recognition")
# Using cmake scripts and modules
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR})

set(TEXT_DEPS opencv_ml opencv_highgui opencv_imgproc opencv_core opencv_features2d opencv_calib3d)

find_package(Caffe)
if(Caffe_FOUND)
  message(STATUS "Caffe:   YES")
  set(HAVE_CAFFE 1)
else()
  message(STATUS "Caffe:   NO")
#  list(APPEND TEXT_DEPS opencv_dnn)
endif()

#internal dependencies
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

ocv_define_module(text opencv_ml opencv_imgproc opencv_core opencv_features2d opencv_calib3d WRAP python)
#ocv_define_module(text ${TEXT_DEPS} WRAP python)

#set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR})

find_package(Tesseract)
if(${Tesseract_FOUND})
  message(STATUS "Tesseract:   YES")
  include_directories(${Tesseract_INCLUDE_DIR})
  target_link_libraries(opencv_text ${Tesseract_LIBS})
  add_definitions(-DHAVE_TESSERACT)
else()
  message(STATUS "Tesseract:   NO")
endif()




if(HAVE_CAFFE AND HAVE_GLOG AND HAVE_PROTOBUF)
  include_directories(${Caffe_INCLUDE_DIR})
  find_package(HDF5 COMPONENTS HL REQUIRED)
  include_directories(SYSTEM ${HDF5_INCLUDE_DIRS} ${HDF5_HL_INCLUDE_DIR})
  list(APPEND Caffe_LINKER_LIBS ${HDF5_LIBRARIES})
  find_package(Boost 1.46 REQUIRED COMPONENTS system thread filesystem)
  include_directories(SYSTEM ${Boost_INCLUDE_DIR})
  include_directories(SYSTEM /usr/local/cuda-8.0/targets/x86_64-linux/include/ usr/local/cuda-8.0/include/ /usr/local/cuda-7.5/targets/x86_64-linux/include/ )
  link_directories(SYSTEM /usr/local/cuda-8.0/targets/x86_64-linux/lib/ usr/local/cuda-8.0/lib/ /usr/local/cuda-7.5/targets/x86_64-linux/lib/ /usr/lib/openblas-base/lib /usr/local/cuda-8.0/lib64)
  list(APPEND Caffe_LINKER_LIBS ${Boost_LIBRARIES})
  target_link_libraries(opencv_text atlas blas ${Caffe_LIBS} ${Glog_LIBS} ${Protobuf_LIBS} ${HDF5_LIBRARIES} ${Boost_LIBRARIES})
  add_definitions(-DHAVE_CAFFE)
endif() #HAVE_CAFFE

message(STATUS "TEXT CAFFE SEARCH")
if()
  message(STATUS "TEXT NO CAFFE CONFLICT")
else()
  message(STATUS "TEXT CAFFE CONFLICT")
endif()

