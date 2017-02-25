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
#include <opencv2/core/ocl.hpp>
#include "layers_common.hpp"
#include "convolution_layer.hpp"
#include "op_im2col.hpp"
#include "op_blas.hpp"
#include <opencv2/dnn/shape_utils.hpp>
#include <iostream>

namespace cv
{
namespace dnn
{

ConvolutionLayerImpl::ConvolutionLayerImpl()
{
    tryUseOpenCL = false; //true;
    numOutput = -1;
    group = -1;

    #if HAVE_CBLAS
        if (getBlasThreads() != cv::getThreadNum())
        {
            setBlasThreads(cv::getThreadNum());
        }
    #endif
}

void ConvolutionLayerImpl::init()
{
    CV_Assert(1 <= blobs.size() && blobs.size() <= 2);

    bias = (blobs.size() >= 2);
    numOutput = blobs[0].num();

    CV_Assert(blobs[0].dims() == 4 && blobs[0].cols() == kernel.width && blobs[0].rows() == kernel.height);
    CV_Assert(!bias || blobs[1].total() == (size_t)blobs[0].num());

    //TODO: dilation in OCL mode
    useOpenCL = ocl::useOpenCL() && tryUseOpenCL && dilation == Size(1, 1);
}

void ConvolutionLayerImpl::allocate(const std::vector<Blob*> &inputs, std::vector<Blob> &outputs)
{
    init();

    CV_Assert(inputs.size() > 0);
    const Blob &input = *inputs[0];
    CV_Assert(input.dims() == 4 && (input.type() == CV_32F || input.type() == CV_64F));
    computeInpOutShape(input);

    group = inpCn / blobs[0].channels();
    CV_Assert(inpCn % group == 0 && outCn % group == 0);
    CV_Assert(blobs[0].num() == outCn && blobs[0].channels() == inpCn / group);

    outGroupCn = outCn / group;
    inpGroupCn = inpCn / group;
    ksize = inpGroupCn * kernel.height * kernel.width;

    for (size_t i = 0; i < inputs.size(); i++)
    {
        CV_Assert(inputs[i]->type() == input.type());
        CV_Assert(inputs[i]->dims() == 4 && inputs[i]->channels() == input.channels());
        CV_Assert(inputs[i]->rows() == input.rows() && inputs[i]->cols() == input.cols());
    }

    int allocFlags = useOpenCL ? Blob::ALLOC_UMAT : Blob::ALLOC_MAT;

    if (!is1x1())
    {
        colBlob.create(Shape(ksize, outH * outW), input.type(), allocFlags);
    }

    if (bias)
    {
        biasOnesBlob.create(Shape(1, topH * topW), input.type(), allocFlags);
        biasOnesBlob.setTo(1);
    }

    outputs.resize(inputs.size());
    for (size_t i = 0; i < inputs.size(); i++)
    {
        outputs[i].create(Shape(inputs[i]->num(), topCn, topH, topW), input.type(), allocFlags);
    }
}

bool ConvolutionLayerImpl::is1x1() const
{
    return (kernel.height == 1 && kernel.width == 1) &&
           (stride.height == 1 && stride.width == 1) &&
           (dilation.height == 1 && dilation.width == 1);
}

template<typename XMat>
void ConvolutionLayerImpl::forward_(std::vector<Blob*> &inputs, std::vector<Blob> &outputs)
{
    XMat weightsMat = reshaped(blobs[0].getRefConst<XMat>(), Shape(outCn, ksize));
    XMat biasesMat  = (bias) ? reshaped(blobs[1].getRefConst<XMat>(), Shape(outCn, 1)) : XMat();

    for (size_t ii = 0; ii < outputs.size(); ii++)
    {
        int numImg = inputs[ii]->size(0);
        XMat inpMat = inputs[ii]->getRefConst<XMat>();
        XMat outMat = reshaped(outputs[ii].getRef<XMat>(), Shape(numImg*group*outGroupCn, outH*outW));

        for (int n = 0; n < numImg; n++)
        {
            for (int g = 0; g < group; g++)
            {
                XMat colMat, curInp = slice(inpMat, n, _Range(g * inpGroupCn, inpGroupCn));
                im2col(curInp, colMat);

                _Range kerRange(g * outGroupCn, outGroupCn);
                XMat kerMat = weightsMat.rowRange(kerRange);

                _Range outRange((g + n * group) * outGroupCn, outGroupCn);
                XMat dstMat = outMat.rowRange(outRange);

                dnn::gemm(kerMat, colMat, 1, dstMat, 0);

                if (bias)
                {
                    dnn::gemm(biasesMat.rowRange(kerRange), biasOnesBlob.getRefConst<XMat>(), 1, dstMat, 1);
                }
            }
        }
    }
}

void ConvolutionLayerImpl::forward(std::vector<Blob*> &inputs, std::vector<Blob> &outputs)
{
    if (!useOpenCL)
        forward_<Mat>(inputs, outputs);
    else
        forward_<UMat>(inputs, outputs);
}

void ConvolutionLayerImpl::im2col(const UMat &srcImg, UMat &dstCol)
{
    if (is1x1())
    {
        dstCol = reshaped(srcImg, Shape(ksize, outH*outW));
        return;
    }
#ifdef HAVE_OPENCL
    CV_Assert(im2col_ocl(srcImg, inpGroupCn, inpH, inpW, kernel.height, kernel.width, pad.height, pad.width, stride.height, stride.width, dilation.height, dilation.width, this->colBlob.umatRef()));
    dstCol = this->colBlob.umatRefConst();
#else
    CV_Error(Error::StsInternal, "");
    dstCol = srcImg; //supress warning
#endif
}

void ConvolutionLayerImpl::im2col(const Mat &srcImg, Mat &dstCol)
{
    if (is1x1())
    {
        dstCol = reshaped(srcImg, Shape(ksize, outH*outW));
        return;
    }

    Mat &colMat = colBlob.matRef();
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

void ConvolutionLayerImpl::computeInpOutShape(const Blob &input)
{
    inpH = input.rows();
    inpW = input.cols();
    inpCn = input.channels();
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

    topH = outH; topW = outW; topCn = outCn;
}

//Deconvolution

DeConvolutionLayerImpl::DeConvolutionLayerImpl()
{

}

void DeConvolutionLayerImpl::computeInpOutShape(const Blob &inpBlob)
{
    outH = inpBlob.rows();
    outW = inpBlob.cols();
    outCn = inpBlob.channels();

    inpH = stride.height * (outH - 1) + kernel.height - 2 * pad.height;
    inpW = stride.width * (outW - 1) + kernel.width - 2 * pad.width;
    inpCn = numOutput;

    topH = inpH; topW = inpW; topCn = inpCn;
}

void DeConvolutionLayerImpl::forward(std::vector<Blob*> &inputs, std::vector<Blob> &outputs)
{
    if (!useOpenCL)
        forward_<Mat>(inputs, outputs);
    else
        forward_<UMat>(inputs, outputs);
}

template<typename XMat>
void DeConvolutionLayerImpl::forward_(std::vector<Blob *> &inputs, std::vector<Blob> &outputs)
{
    XMat weightsMat = reshaped(blobs[0].getRefConst<XMat>(), Shape(outCn, ksize));
    XMat biasesMat  = (bias) ? reshaped(blobs[1].getRefConst<XMat>(), Shape(outCn, 1)) : XMat();

    for (size_t ii = 0; ii < outputs.size(); ii++)
    {
        int numImg = inputs[ii]->size(0);
        XMat convBlob = reshaped(inputs[ii]->getRefConst<XMat>(), Shape(numImg*outCn, outH*outW));
        XMat decnBlob = reshaped(outputs[ii].getRef<XMat>(), Shape(numImg*inpCn, inpH*inpW));

        for (int n = 0; n < numImg; n++)
        {
            for (int g = 0; g < group; g++)
            {
                XMat dstMat = decnBlob.rowRange(_Range((g + n * group) * inpGroupCn, inpGroupCn));
                XMat &colMat = (is1x1()) ? dstMat : colBlob.getRef<XMat>();

                XMat convMat = convBlob.rowRange(_Range((g + n * group) * outGroupCn, outGroupCn));
                XMat wghtMat = weightsMat.rowRange(_Range(g * outGroupCn, outGroupCn));

                dnn::gemm(wghtMat, convMat, 1, colMat, 0, GEMM_1_T);

                if (!is1x1())
                    col2im(colMat, dstMat);

                if (bias)
                {
                    XMat curBiasMat = biasesMat.rowRange(_Range(g * outGroupCn, outGroupCn));
                    dnn::gemm(curBiasMat, biasOnesBlob.getRefConst<XMat>(), 1, dstMat, 1);
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
        col2im_CpuPBody<float>::run(colMat.ptr<float>(), inpGroupCn, inpH, inpW, kernel.height, kernel.width, pad.height, pad.width, stride.height, stride.width, dstImg.ptr<float>());
    if (dstImg.type() == CV_64F)
        col2im_CpuPBody<double>::run(colMat.ptr<double>(), inpGroupCn, inpH, inpW, kernel.height, kernel.width, pad.height, pad.width, stride.height, stride.width, dstImg.ptr<double>());
}

void DeConvolutionLayerImpl::col2im(const UMat &colMat, UMat &dstImg)
{
    if (is1x1())
    {
        dstImg = colMat;
        return;
    }
#ifdef HAVE_OPENCL
    CV_Assert(col2im_ocl(colMat, inpGroupCn, inpH, inpW, kernel.height, kernel.width, pad.height, pad.width, stride.height, stride.width, dstImg));
#else
    CV_Error(Error::StsInternal, "");
    dstImg = colMat;
#endif
}

//Initializers

Ptr<BaseConvolutionLayer> ConvolutionLayer::create(Size kernel, Size stride, Size pad, Size dilation)
{
    ConvolutionLayerImpl *l = new ConvolutionLayerImpl();
    l->kernel = kernel;
    l->pad = pad;
    l->stride = stride;
    l->dilation = dilation;
    return Ptr<BaseConvolutionLayer>(l);
}

Ptr<BaseConvolutionLayer> DeconvolutionLayer::create(Size kernel, Size stride, Size pad, Size dilation)
{
    DeConvolutionLayerImpl *l = new DeConvolutionLayerImpl();
    l->kernel = kernel;
    l->pad = pad;
    l->stride = stride;
    l->dilation = dilation;
    return Ptr<BaseConvolutionLayer>(l);
}

}
}
