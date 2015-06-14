#include "precomp.hpp"
#include "layers.hpp"
#include <math.h>

namespace cv
{
namespace dnn
{

struct ReLUFunctor
{
    float negative_slope;

    ReLUFunctor(LayerParams &params)
    {
        if (params.has("negative_slope"))
            negative_slope = params.get<float>("negative_slope");
        else
            negative_slope = 0.f;
    }

    inline float operator()(float x)
    {
        return (x >= 0) ? x : negative_slope * x;
    }
};

struct TanHFunctor
{
    TanHFunctor(LayerParams &params) {}

    inline float operator()(float x)
    {
        return tanh(x);
    }
};

REGISTER_LAYER_CLASS(ReLU, ElementWiseLayer<ReLUFunctor>)
REGISTER_LAYER_CLASS(TanH, ElementWiseLayer<TanHFunctor>)
REGISTER_LAYER_CLASS(Convolution, ConvolutionLayer)
REGISTER_LAYER_CLASS(Pooling, PoolingLayer)
REGISTER_LAYER_CLASS(InnerProduct, FullyConnectedLayer)

//////////////////////////////////////////////////////////////////////////

PoolingLayer::PoolingLayer(LayerParams &params)
{

}

void PoolingLayer::allocate(const std::vector<Blob*> &inputs, std::vector<Blob> &outputs)
{

}

void PoolingLayer::forward(std::vector<Blob*> &inputs, std::vector<Blob> &outputs)
{

}

//////////////////////////////////////////////////////////////////////////

ConvolutionLayer::ConvolutionLayer(LayerParams &params)
{

}

void ConvolutionLayer::allocate(const std::vector<Blob*> &inputs, std::vector<Blob> &outputs)
{

}


template <typename Dtype>
void im2col_cpu(const Dtype* data_im, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    Dtype* data_col)
{
  int height_col = (height + 2 * pad_h - kernel_h) / stride_h + 1;
  int width_col = (width + 2 * pad_w - kernel_w) / stride_w + 1;
  int channels_col = channels * kernel_h * kernel_w;
  for (int c = 0; c < channels_col; ++c) {
    int w_offset = c % kernel_w;
    int h_offset = (c / kernel_w) % kernel_h;
    int c_im = c / kernel_h / kernel_w;
    for (int h = 0; h < height_col; ++h) {
      for (int w = 0; w < width_col; ++w) {
        int h_pad = h * stride_h - pad_h + h_offset;
        int w_pad = w * stride_w - pad_w + w_offset;
        if (h_pad >= 0 && h_pad < height && w_pad >= 0 && w_pad < width)
          data_col[(c * height_col + h) * width_col + w] =
            data_im[(c_im * height + h_pad) * width + w_pad];
        else
          data_col[(c * height_col + h) * width_col + w] = 0;
      }
    }
  }
}

void ConvolutionLayer::forward(std::vector<Blob*> &inputs, std::vector<Blob> &outputs)
{
    CV_Assert(inputs.size() == outputs.size());
    
    for (size_t i = 0; i < outputs.size(); i++)
    {
        
    }
}

//////////////////////////////////////////////////////////////////////////


FullyConnectedLayer::FullyConnectedLayer(LayerParams &params)
{

}

void FullyConnectedLayer::allocate(const std::vector<Blob*> &inputs, std::vector<Blob> &outputs)
{

}

void FullyConnectedLayer::forward(std::vector<Blob*> &inputs, std::vector<Blob> &outputs)
{

}

}
}