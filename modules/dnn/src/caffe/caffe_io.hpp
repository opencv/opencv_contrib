#ifndef __OPENCV_DNN_CAFFE_IO_HPP__
#define __OPENCV_DNN_CAFFE_IO_HPP__
#if HAVE_PROTOBUF

#include "caffe.pb.h"

namespace cv {
namespace dnn {

// Read parameters from a file into a NetParameter proto message.
void ReadNetParamsFromTextFileOrDie(const char* param_file,
                                    caffe::NetParameter* param);
void ReadNetParamsFromBinaryFileOrDie(const char* param_file,
                                      caffe::NetParameter* param);

}
}
#endif
#endif
