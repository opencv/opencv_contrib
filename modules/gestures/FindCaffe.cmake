unset(Caffe_FOUND)
unset(Caffe_INCLUDE_DIR)
unset(Caffe_LIBRARIES)

find_path(Caffe_INCLUDE_DIR
    NAMES caffe/caffe.hpp caffe/common.hpp caffe/net.hpp caffe/data_transformers.hpp caffe/proto/caffe.pb.h
    PATHS /usr/local/include $ENV{HOME}/caffe/include)

find_library(Caffe_LIBRARIES
    NAMES caffe
    PATHS /usr/local/lib $ENV{HOME}/caffe/build/lib)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Caffe DEFAULT_MSG Caffe_INCLUDE_DIR Caffe_LIBRARIES)
