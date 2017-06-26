# Caffe package for Object Detection using CNNs
set(Caffe_DIR "" CACHE STRING
  "Location of the root directory of Caffe installation")
unset(CAFFE_FOUND)

find_path(Caffe_INCLUDE_DIR
  NAMES caffe.hpp
  PATH_SUFFIXES include/caffe caffe
  PATHS "${Caffe_DIR}"
  DOC "Directory where `caffe.hpp` is found")

find_library(Caffe_LIBRARY
  NAMES caffe
  PATH_SUFFIXES build/lib lib
  PATHS "${Caffe_DIR}"
  DOC "Directory where `caffe` library is found")

if(Caffe_INCLUDE_DIR AND Caffe_LIBRARY)
	set(CAFFE_FOUND 1)
	set(Caffe_LIBRARIES ${Caffe_LIBRARY})
	set(Caffe_INCLUDE_DIRS ${Caffe_INCLUDE_DIR})
endif(Caffe_INCLUDE_DIR AND Caffe_LIBRARY)