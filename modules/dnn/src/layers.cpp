#include "precomp.hpp"
#include "layers.hpp"
#include <math.h>
#include <float.h>
#include <iostream>
#include <algorithm>
using std::max;
using std::min;

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


static void getKernelParams(LayerParams &params, int &kernelH, int &kernelW, int &padH, int &padW, int &strideH, int &strideW)
{
    if (params.has("kernel_h") && params.has("kernel_w"))
    {
        kernelH = params.get<int>("kernel_h");
        kernelW = params.get<int>("kernel_w");
    }
    else if (params.has("kernel_size"))
    {
        kernelH = kernelW = params.get<int>("kernel_size");
    }
    else
    {
        CV_Error(cv::Error::StsBadArg, "kernel_size (or kernel_h and kernel_w) not specified");
    }

    if (params.has("pad_h") && params.has("pad_w"))
    {
        padH = params.get<int>("pad_h");
        padW = params.get<int>("pad_w");
    }
    else
    {
        padH = padW = params.get<int>("pad", 0);
    }

    if (params.has("stride_h") && params.has("stride_w"))
    {
        strideH = params.get<int>("stride_h");
        strideW = params.get<int>("stride_w");
    }
    else
    {
        strideH = strideW = params.get<int>("stride", 1);
    }

    CV_Assert(kernelH > 0 && kernelW > 0 && padH >= 0 && padW >= 0 && strideH > 0 & strideW > 0);
}

PoolingLayer::PoolingLayer(LayerParams &params)
{
    if (params.has("pool"))
    {
        String pool = params.get<String>("pool").toLowerCase();
        if (pool == "max")
            type = MAX;
        else if (pool == "ave")
            type = AVE;
        else if (pool == "stochastic")
            type = STOCHASTIC;
        else
            CV_Error(cv::Error::StsBadArg, "Unknown pooling type \"" + pool + "\"");
    }
    else
    {
        type = MAX;
    }

    getKernelParams(params, kernelH, kernelW, padH, padW, strideH, strideW);
}

void PoolingLayer::allocate(const std::vector<Blob*> &inputs, std::vector<Blob> &outputs)
{
    CV_Assert(inputs.size() > 0);

    inH = inputs[0]->cols();
    inW = inputs[0]->rows();
    computeOutputShape(inH, inW);

    outputs.resize(inputs.size());
    for (size_t i = 0; i < inputs.size(); i++)
    {
        CV_Assert(inputs[i]->rows() == inH && inputs[i]->cols() == inW);
        outputs[i].create(inputs[i]->num(), inputs[i]->channels(), pooledH, pooledW);
    }
}

void PoolingLayer::forward(std::vector<Blob*> &inputs, std::vector<Blob> &outputs)
{
    for (size_t ii = 0; ii < inputs.size(); ii++)
    {
        switch (type)
        {
        case MAX:
            maxPooling(*inputs[ii], outputs[ii]);
            break;
        default:
            CV_Error(cv::Error::StsNotImplemented, "Not implemented");
            break;
        }
    }
}

void PoolingLayer::maxPooling(Blob &input, Blob &output)
{
    CV_DbgAssert(output.rows() == pooledH && output.cols() == pooledW);

    for (int n = 0; n < input.num(); ++n) 
    {
        for (int c = 0; c < input.channels(); ++c)
        {
            float *srcData = input.ptr<float>(n, c);
            float *dstData = output.ptr<float>(n, c);

            for (int ph = 0; ph < pooledH; ++ph)
            {
                for (int pw = 0; pw < pooledW; ++pw)
                {
                    int hstart = ph * strideH - padH;
                    int wstart = pw * strideW - padW;
                    int hend = min(hstart + kernelH, inH);
                    int wend = min(wstart + kernelW, inW);
                    hstart = max(hstart, 0);
                    wstart = max(wstart, 0);
                    const int pool_index = ph * pooledW + pw;
                    float max_val = -FLT_MAX;

                    for (int h = hstart; h < hend; ++h)
                        for (int w = wstart; w < wend; ++w) 
                        {
                            const int index = h * inW + w;
                            if (srcData[index] > max_val)
                                max_val = srcData[index];
                        }

                    dstData[pool_index] = max_val;
                }
            }
        }
    }
}

void PoolingLayer::computeOutputShape(int inH, int inW)
{
    //Yeah something strange Caffe scheme-)
    pooledH = static_cast<int>(ceil(static_cast<float>(inH + 2 * padH - kernelH) / strideH)) + 1;
    pooledW = static_cast<int>(ceil(static_cast<float>(inW + 2 * padW - kernelW) / strideW)) + 1;

    if (padH || padW)
    {
        // If we have padding, ensure that the last pooling starts strictly
        // inside the image (instead of at the padding); otherwise clip the last.
        if ((pooledH - 1) * strideH >= inH + padH)
            --pooledH;
        if ((pooledW - 1) * strideW >= inW + padW)
            --pooledW;
        CV_Assert((pooledH - 1) * strideH < inH + padH);
        CV_Assert((pooledW - 1) * strideW < inW + padW);
    }
}

//////////////////////////////////////////////////////////////////////////

ConvolutionLayer::ConvolutionLayer(LayerParams &params)
{
    getKernelParams(params, kernelH, kernelW, padH, padW, strideH, strideW);

    numOutput = params.get<int>("num_output");
    bias = params.get<bool>("bias_term", true);
    group = params.get<int>("group", 1);
    CV_Assert(numOutput % group == 0);

    CV_Assert(params.learnedBlobs.size() >= 1 && (!bias || params.learnedBlobs.size() >= 2));
    learnedParams.assign(params.learnedBlobs.begin(), params.learnedBlobs.begin() + (bias ? 2 : 1));
    
    Blob &weightBlob = learnedParams[0];
    CV_Assert(weightBlob.cols() == kernelW && weightBlob.rows() == kernelH && weightBlob.num() == numOutput);

    if (bias)
    {
        Blob &biasBlob = learnedParams[1];
        CV_Assert(biasBlob.total() == numOutput);
    }
}

void ConvolutionLayer::allocate(const std::vector<Blob*> &inputs, std::vector<Blob> &outputs)
{
    CV_Assert(inputs.size() > 0);
    
    Blob &weightBlob = learnedParams[0];
    
    inCn = inputs[0]->channels();
    CV_Assert(inCn % group == 0 && weightBlob.channels() == inCn);

    inH = inputs[0]->rows();
    inW = inputs[0]->cols();
    computeOutputShape(inH, inW);

    outputs.resize(inputs.size());
    for (size_t i = 0; i < inputs.size(); i++)
    {
        CV_Assert(inputs[i]->rows() == inH && inputs[i]->cols() == inW && inputs[i]->channels() == inCn);
        int num = inputs[i]->num();
        
        outputs[i].create(num, numOutput, outH, outW);
    }

    colCn = kernelH * kernelW * inCn;
    imColsMat.create(colCn, outH * outW, CV_32F);

    if (bias)
    {
        biasOnesMat = Mat::ones(1, outH * outW, CV_32F);
    }
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

    float *colPtr = imColsMat.ptr<float>();
    float *weigtsPtr = learnedParams[0].ptr<float>();
    float *biasPtr = (bias) ? learnedParams[1].ptr<float>() : NULL;

    CV_Assert(group == 1);
    
    for (size_t ii = 0; ii < outputs.size(); ii++)
    {
        int num = inputs[ii]->num();

        for (int n = 0; n < num; n++)
        {
            float *srcImPtr = inputs[ii]->ptr<float>(n);
            float *dstImPtr = outputs[ii].ptr<float>(n);

            im2col_cpu(srcImPtr, inCn, inH, inW, kernelH, kernelW, padH, padW, strideH, strideW, colPtr);

            Mat weightsMat(numOutput, colCn, CV_32F, weigtsPtr);
            Mat dstIm(numOutput, outH*outW, CV_32F, dstImPtr);

            cv::gemm(weightsMat, imColsMat, 1, noArray(), 0, dstIm);

            if (bias)
            {
                Mat biasMat(numOutput, 1, CV_32F, biasPtr);
                cv::gemm(biasMat, biasOnesMat, 1, dstIm, 1, dstIm);
            }
        }
    }
}

void ConvolutionLayer::computeOutputShape(int inH, int inW)
{
    outH = (inH + 2 * padH - kernelH) / strideH + 1;
    outW = (inW + 2 * padW - kernelW) / strideW + 1;
}

//////////////////////////////////////////////////////////////////////////

FullyConnectedLayer::FullyConnectedLayer(LayerParams &params)
{
    numOutputs = params.get<int>("num_output");
    bias = params.get<bool>("bias_term", true);
    
    CV_Assert(params.learnedBlobs.size() >= 1);
    CV_Assert(!bias || (params.learnedBlobs.size() >= 2 && params.learnedBlobs[1].total() == numOutputs));

    learnedParams.resize(bias ? 2 : 1);
    learnedParams[0] = params.learnedBlobs[0];
    if (bias)
    {
        learnedParams[1] = params.learnedBlobs[1];
    }
}

void FullyConnectedLayer::allocate(const std::vector<Blob*> &inputs, std::vector<Blob> &outputs)
{
    CV_Assert(inputs.size() > 0);

    inC = inputs[0]->channels();
    inH = inputs[0]->rows();
    inW = inputs[0]->cols();
    inSize = inC * inH * inW;

    CV_Assert(inSize * numOutputs == learnedParams[0].total());

    outputs.resize(inputs.size());
    for (size_t i = 0; i < inputs.size(); i++)
    {
        if (i != 0)
            CV_Assert(inputs[i]->channels() == inC && inputs[i]->rows() == inH && inputs[i]->cols() == inW);

        outputs[i].create(inputs[i]->num(), numOutputs, 1, 1);
    }
}

void FullyConnectedLayer::forward(std::vector<Blob*> &inputs, std::vector<Blob> &outputs)
{
    for (size_t i = 0; i < inputs.size(); i++)
    {
        int M = inputs[i]->num();
        int N = numOutputs;
        int K = inSize;

        Mat srcMat(M, K, CV_32F, inputs[i]->ptr<float>());
        Mat weights(K, N, CV_32F, learnedParams[0].ptr<float>());
        Mat dstMat(M, N, CV_32F, outputs[i].ptr<float>());

        cv::gemm(srcMat, weights, 1, noArray(), 0, dstMat);

        if (bias)
        {
            Mat biasOnesMat = Mat::ones(M, 1, CV_32F);
            Mat biasMat(1, N, CV_32F, learnedParams[1].ptr<float>());
            cv::gemm(biasOnesMat, biasMat, 1, dstMat, 1, dstMat);
        }
    }
}

}
}