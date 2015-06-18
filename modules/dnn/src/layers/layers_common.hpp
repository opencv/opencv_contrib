#ifndef __OPENCV_DNN_LAYERS_LAYERS_COMMON_HPP__
#define __OPENCV_DNN_LAYERS_LAYERS_COMMON_HPP__
#include <opencv2/dnn.hpp>

namespace cv
{
namespace dnn
{

void getKernelParams(LayerParams &params, int &kernelH, int &kernelW, int &padH, int &padW, int &strideH, int &strideW);

}
}

#endif