/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2013, OpenCV Foundation, all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#include "../precomp.hpp"
#include "layers_common.hpp"
#include "op_im2col.hpp"
#include "op_blas.hpp"
#include <opencv2/dnn/shape_utils.hpp>
#include <iostream>

namespace cv
{
namespace dnn
{

class BaseConvolutionLayerImpl : public ConvolutionLayer
{
public:
    BaseConvolutionLayerImpl();
    virtual void allocate(const std::vector<Mat*> &inputs, std::vector<Mat> &outputs);

    void init();
    virtual void computeInpOutShape(const Mat &inpBlob) = 0;
    bool is1x1() const;

    int numOutput, group;
    int inpH, inpW, inpCn;
    int outH, outW, outCn;
    int inpGroupCn, outGroupCn;
    int ksize;
    std::vector<int> colRowBlobShape;

    bool bias;
    Mat colRowBlob, biasOnesBlob;
};

//TODO: simultaneously convolution and bias addition for cache optimization
class ConvolutionLayerImpl : public BaseConvolutionLayerImpl
{
public:
    virtual void forward(std::vector<Mat*> &inputs, std::vector<Mat> &outputs);
    virtual void computeInpOutShape(const Mat &inpBlob);

    void im2col(const  Mat &srcImg,  Mat &dstCol);
    void im2row(const  Mat &srcImg,  Mat &dstRow);
};

class DeConvolutionLayerImpl : public BaseConvolutionLayerImpl
{
public:
    virtual void forward(std::vector<Mat*> &inputs, std::vector<Mat> &outputs);

    virtual void computeInpOutShape(const Mat &inpBlob);
    void col2im(const  Mat &colMat, Mat  &dstImg);
};


BaseConvolutionLayerImpl::BaseConvolutionLayerImpl():
    numOutput(-1), group(-1),
    inpH(0), inpW(0), inpCn(0),
    outH(0), outW(0), outCn(0),
    inpGroupCn(0), outGroupCn(0),
    ksize(0), bias(false)
{
#ifdef HAVE_LAPACK
    if (getBlasThreads() != cv::getThreadNum())
    {
        setBlasThreads(cv::getThreadNum());
    }
#endif
}

void BaseConvolutionLayerImpl::init()
{
    CV_Assert(blobs.size() >= 1 && blobs.size() <= 2);
    CV_Assert(blobs[0].dims == 4 && blobs[0].size[3] == kernel.width && blobs[0].size[2] == kernel.height);

    bias = (blobs.size() >= 2);
}

void BaseConvolutionLayerImpl::allocate(const std::vector<Mat*> &inputs, std::vector<Mat> &outputs)
{
    CV_Assert(inputs.size() > 0);

    init();

    const Mat &input = *inputs[0];
    CV_Assert(input.dims == 4 && (input.type() == CV_32F || input.type() == CV_64F));
    for (size_t i = 0; i < inputs.size(); i++)
    {
        CV_Assert(inputs[i]->type() == input.type());
        CV_Assert(inputs[i]->dims == 4 && inputs[i]->size[1] == input.size[1]);
        CV_Assert(inputs[i]->size[2] == input.size[2] && inputs[i]->size[3] == input.size[3]);
    }

    computeInpOutShape(input);

    if (bias)
    {
        biasOnesBlob.create(1, outH * outW, input.type());
        biasOnesBlob.setTo(1);
    }

    outputs.resize(inputs.size());
    for (size_t i = 0; i < inputs.size(); i++)
    {
        int sz[] = { inputs[i]->size[0], outCn, outH, outW };
        outputs[i].create(4, sz, input.type());
    }

    if (!is1x1())
    {
        colRowBlob.create((int)colRowBlobShape.size(), &colRowBlobShape[0], input.type());
        colRowBlob.setTo(0);
    }
}

bool BaseConvolutionLayerImpl::is1x1() const
{
    return (kernel.height == 1 && kernel.width == 1) &&
           (stride.height == 1 && stride.width == 1) &&
           (dilation.height == 1 && dilation.width == 1);
}

void ConvolutionLayerImpl::computeInpOutShape(const Mat &input)
{
    CV_Assert(!bias || blobs[1].total() == (size_t)blobs[0].size[0]);

    numOutput = blobs[0].size[0];

    inpH = input.size[2];
    inpW = input.size[3];
    inpCn = input.size[1];
    outCn = numOutput;

    if (padMode.empty())
    {
        outH = (inpH + 2 * pad.height - (dilation.height * (kernel.height - 1) + 1)) / stride.height + 1;
        outW = (inpW + 2 * pad.width - (dilation.width * (kernel.width - 1) + 1)) / stride.width + 1;
    }
    else
    {
        getConvPoolOutParams(inpH, inpW, kernel, stride, pad, padMode, outH, outW);
    }

    group = inpCn / blobs[0].size[1];

    CV_Assert(inpCn % group == 0 && outCn % group == 0);
    CV_Assert(blobs[0].size[0] == outCn && blobs[0].size[1] == inpCn / group);

    outGroupCn = outCn / group;
    inpGroupCn = inpCn / group;
    ksize = inpGroupCn * kernel.height * kernel.width;

    colRowBlobShape.clear();
    colRowBlobShape.push_back(outH*outW);
    colRowBlobShape.push_back(ksize);
}

void ConvolutionLayerImpl::forward(std::vector<Mat*> &inputs, std::vector<Mat> &outputs)
{
    CV_Assert(inputs.size() > 0);

    Mat weightsMat = blobs[0].reshape(1, outCn);
    Mat biasesMat  = bias ? blobs[1].reshape(1, outCn) : Mat();

    for (size_t ii = 0; ii < outputs.size(); ii++)
    {
        int numImg = inputs[ii]->size[0];
        Mat inpMat = *inputs[ii];
        Mat outMat = outputs[ii].reshape(1, numImg*group*outGroupCn);

        for (int n = 0; n < numImg; n++)
        {
            for (int g = 0; g < group; g++)
            {
                Mat colMat, curInp = slice(inpMat, n, _Range(g * inpGroupCn, inpGroupCn));

                im2row(curInp, colMat);

                _Range kerRange(g * outGroupCn, outGroupCn);
                Mat kerMat = weightsMat.rowRange(kerRange);

                _Range outRange((g + n * group) * outGroupCn, outGroupCn);
                Mat dstMat = outMat.rowRange(outRange);

                dnn::gemm(kerMat, colMat, 1, dstMat, 0, GEMM_2_T);

                if (bias)
                {
                    dnn::gemm(biasesMat.rowRange(kerRange), biasOnesBlob, 1, dstMat, 1);
                }
            }
        }
    }
}

void ConvolutionLayerImpl::im2col(const Mat &srcImg, Mat &dstCol)
{
    if (is1x1())
    {
        dstCol = srcImg.reshape(1, ksize);
        return;
    }

    Mat &colMat = colRowBlob;
    if (srcImg.type() == CV_32F)
        im2col_CpuPBody<float>::run(srcImg.ptr<float>(), inpGroupCn, inpH, inpW, kernel.height,
                                    kernel.width, pad.height, pad.width, stride.height, stride.width,
                                    dilation.height, dilation.width, outH, outW, colMat.ptr<float>());
    if (srcImg.type() == CV_64F)
        im2col_CpuPBody<double>::run(srcImg.ptr<double>(), inpGroupCn, inpH, inpW, kernel.height,
                                     kernel.width, pad.height, pad.width, stride.height, stride.width,
                                     dilation.height, dilation.width, outH, outW, colMat.ptr<double>());

    dstCol = colMat;
}

void ConvolutionLayerImpl::im2row(const  Mat &srcImg,  Mat &dstRow)
{
    if (is1x1())
    {
        dstRow = srcImg.reshape(1, ksize).t();
        return;
    }

    Mat &colMat = colRowBlob;
    if (srcImg.type() == CV_32F)
        im2row_CpuPBody<float>::run(srcImg.ptr<float>(), inpGroupCn, inpH, inpW, kernel.height,
                                    kernel.width, pad.height, pad.width, stride.height, stride.width,
                                    dilation.height, dilation.width, outH, outW, colMat.ptr<float>());
    if (srcImg.type() == CV_64F)
        im2row_CpuPBody<double>::run(srcImg.ptr<double>(), inpGroupCn, inpH, inpW, kernel.height,
                                     kernel.width, pad.height, pad.width, stride.height, stride.width,
                                     dilation.height, dilation.width, outH, outW, colMat.ptr<double>());

    dstRow = colMat;
}

//Deconvolution

void DeConvolutionLayerImpl::computeInpOutShape(const Mat &inpBlob)
{
    CV_Assert(!bias || blobs[1].total() == (size_t)blobs[0].size[0]);

    numOutput = blobs[0].size[0];

    inpH = inpBlob.size[2];
    inpW = inpBlob.size[3];
    inpCn = inpBlob.size[1];

    outH = stride.height * (inpH - 1) + kernel.height - 2 * pad.height + adjustPad.height;
    outW = stride.width * (inpW - 1) + kernel.width - 2 * pad.width + adjustPad.width;
    outCn = numOutput;

    group = inpCn / blobs[0].size[1];
    outGroupCn = outCn / group;
    inpGroupCn = inpCn / group;
    ksize = outGroupCn * kernel.height * kernel.width;

    CV_Assert(inpCn % group == 0 && outCn % group == 0);
    CV_Assert(blobs[0].size[0] == outCn && blobs[0].size[1] == inpCn / group);

    colRowBlobShape.clear();
    colRowBlobShape.push_back(ksize);
    colRowBlobShape.push_back(inpH * inpW);
}

void DeConvolutionLayerImpl::forward(std::vector<Mat *> &inputs, std::vector<Mat> &outputs)
{
    Mat weightsMat = blobs[0].reshape(1, inpCn);
    Mat biasesMat  = bias ? blobs[1].reshape(1, outCn) : Mat();

    for (size_t ii = 0; ii < outputs.size(); ii++)
    {
        int numImg = inputs[ii]->size[0];
        Mat convBlob = inputs[ii]->reshape(1, numImg*inpCn);
        Mat decnBlob = outputs[ii].reshape(1, numImg*outCn);

        for (int n = 0; n < numImg; n++)
        {
            for (int g = 0; g < group; g++)
            {
                Mat dstMat = decnBlob.rowRange(_Range((g + n * group) * outGroupCn, outGroupCn));
                Mat &colMat = (is1x1()) ? dstMat : colRowBlob;

                Mat convMat = convBlob.rowRange(_Range((g + n * group) * inpGroupCn, inpGroupCn));
                Mat wghtMat = weightsMat.rowRange(_Range(g * inpGroupCn, inpGroupCn));

                dnn::gemm(wghtMat, convMat, 1, colMat, 0, GEMM_1_T);

                if (!is1x1())
                    col2im(colMat, dstMat);

                if (bias)
                {
                    Mat curBiasMat = biasesMat.rowRange(_Range(g * outGroupCn, outGroupCn));
                    dnn::gemm(curBiasMat, biasOnesBlob, 1, dstMat, 1);
                }
            }
        }
    }
}

void DeConvolutionLayerImpl::col2im(const Mat &colMat, Mat &dstImg)
{
    if (is1x1())
    {
        dstImg = colMat;
        return;
    }
    if (dstImg.type() == CV_32F)
        col2im_CpuPBody<float>::run(colMat.ptr<float>(), outGroupCn, outH, outW, kernel.height, kernel.width, pad.height, pad.width, stride.height, stride.width, dstImg.ptr<float>());
    if (dstImg.type() == CV_64F)
        col2im_CpuPBody<double>::run(colMat.ptr<double>(), inpGroupCn, inpH, inpW, kernel.height, kernel.width, pad.height, pad.width, stride.height, stride.width, dstImg.ptr<double>());
}

//Initializers

/*Ptr<BaseConvolutionLayer> ConvolutionLayer::create(Size kernel, Size stride, Size pad, Size dilation)
{
    ConvolutionLayerImpl *l = new ConvolutionLayerImpl();
    l->kernel = kernel;
    l->pad = pad;
    l->stride = stride;
    l->dilation = dilation;
    return Ptr<BaseConvolutionLayer>(l);
}

Ptr<BaseConvolutionLayer> DeconvolutionLayer::create(Size kernel, Size stride, Size pad, Size dilation, Size adjustPad)
{
    DeConvolutionLayerImpl *l = new DeConvolutionLayerImpl();
    l->kernel = kernel;
    l->pad = pad;
    l->stride = stride;
    l->dilation = dilation;
    l->adjustPad = adjustPad;

    return Ptr<BaseConvolutionLayer>(l);
}*/

//Convolution and Deconvolution
static void initConvDeconvLayerFromCaffe(Ptr<BaseConvolutionLayer> l, const LayerParams &params)
{
    l->setParamsFrom(params);
    getConvolutionKernelParams(params, l->kernel.height, l->kernel.width, l->pad.height,
                               l->pad.width, l->stride.height, l->stride.width, l->dilation.height,
                               l->dilation.width, l->padMode);

    bool bias = params.get<bool>("bias_term", true);
    int numOutput = params.get<int>("num_output");
    int group = params.get<int>("group", 1);

    l->adjustPad.height = params.get<int>("adj_h", 0);
    l->adjustPad.width = params.get<int>("adj_w", 0);

    CV_Assert(numOutput % group == 0);
    CV_Assert((bias && l->blobs.size() == 2) || (!bias && l->blobs.size() == 1));
}

Ptr<BaseConvolutionLayer> ConvolutionLayer::create(const LayerParams &params)
{
    Ptr<BaseConvolutionLayer> l(new ConvolutionLayerImpl);
    initConvDeconvLayerFromCaffe(l, params);
    return l;
}

Ptr<BaseConvolutionLayer> DeconvolutionLayer::create(const LayerParams &params)
{
    Ptr<BaseConvolutionLayer> l(new DeConvolutionLayerImpl);
    initConvDeconvLayerFromCaffe(l, params);

    return l;
}

}
}
