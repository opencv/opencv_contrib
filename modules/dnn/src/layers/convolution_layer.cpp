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

BaseConvolutionLayerImpl::BaseConvolutionLayerImpl():
    bias(false), tryUseOpenCL(false)
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
    CV_Assert(blobs[0].dims() == 4 && blobs[0].cols() == kernel.width && blobs[0].rows() == kernel.height);

    bias = (blobs.size() >= 2);
    useOpenCL = ocl::useOpenCL() && tryUseOpenCL && dilation == Size(1, 1);
}

void BaseConvolutionLayerImpl::allocate(const std::vector<Blob*> &inputs, std::vector<Blob> &outputs)
{
    CV_Assert(inputs.size() > 0);

    init();

    const Blob &input = *inputs[0];
    CV_Assert(input.dims() == 4 && (input.type() == CV_32F || input.type() == CV_64F));
    for (size_t i = 0; i < inputs.size(); i++)
    {
        CV_Assert(inputs[i]->type() == input.type());
        CV_Assert(inputs[i]->dims() == 4 && inputs[i]->channels() == input.channels());
        CV_Assert(inputs[i]->rows() == input.rows() && inputs[i]->cols() == input.cols());
    }

    getConvPoolPaddings(input.size2(), outputs[0].size2(), kernel, stride, padMode, pad);

    int allocFlags = useOpenCL ? Blob::ALLOC_UMAT : Blob::ALLOC_MAT;

    if (bias)
    {
        biasOnesBlob.create(Shape(1, outputs[0].size2().area()), input.type(), allocFlags);
        biasOnesBlob.setTo(1);
    }

    if (!is1x1())
    {
        colRowBlob.create(computeColRowShape(input.shape(), outputs[0].shape()),
                input.type(), allocFlags);
        colRowBlob.setTo(0);
    }
}

bool BaseConvolutionLayerImpl::is1x1() const
{
    return (kernel.height == 1 && kernel.width == 1) &&
           (stride.height == 1 && stride.width == 1) &&
           (dilation.height == 1 && dilation.width == 1);
}

BlobShape ConvolutionLayerImpl::computeColRowShape(const BlobShape &inpShape, const BlobShape &outShape) const
{
    Size out(outShape[3], outShape[2]);
    int inpGroupCn = blobs[0].channels();
    int ksize = inpGroupCn * kernel.height * kernel.width;
    return BlobShape(out.area(), ksize);
}

void ConvolutionLayerImpl::getOutShapes(const std::vector<BlobShape> &inputs,
                                        std::vector<BlobShape> &outputs, const int requiredOutputs) const
{
    CV_Assert(blobs.size() != 0);
    CV_Assert(!bias || blobs[1].total() == (size_t)blobs[0].num());
    CV_Assert(inputs.size() != 0);

    int inpCn = inputs[0][1];
    int inpH = inputs[0][2];
    int inpW = inputs[0][3];

    int outCn = blobs[0].num();
    Size out;

    if (padMode.empty())
    {
        out.height = (inpH + 2 * pad.height - (dilation.height * (kernel.height - 1) + 1)) / stride.height + 1;
        out.width = (inpW + 2 * pad.width - (dilation.width * (kernel.width - 1) + 1)) / stride.width + 1;
    }
    else
    {
        getConvPoolOutParams(Size(inpH, inpW), kernel, stride, padMode, out);
    }

    int group = inpCn / blobs[0].channels();

    CV_Assert(inpCn % group == 0 && outCn % group == 0);
    CV_Assert(blobs[0].num() == outCn && blobs[0].channels() == inpCn / group);

    BlobShape outShape(inputs[0][0], outCn, out.height, out.width);
    outputs.resize(inputs.size(), outShape);
}

template<typename XMat>
void ConvolutionLayerImpl::forward_(std::vector<Blob*> &inputs, std::vector<Blob> &outputs)
{
    CV_Assert(inputs.size() > 0);

    int outCn = blobs[0].num();
    int inpCn = inputs[0]->size(1);
    int inpGroupCn = blobs[0].channels();
    int ksize = inpGroupCn * kernel.height * kernel.width;

    XMat weightsMat = reshaped(blobs[0].getRefConst<XMat>(), Shape(outCn, ksize));
    XMat biasesMat  = (bias) ? reshaped(blobs[1].getRefConst<XMat>(), Shape(outCn, 1)) : XMat();

    for (size_t ii = 0; ii < outputs.size(); ii++)
    {
        int numImg = inputs[ii]->size(0);
        int outH = outputs[ii].size(2), outW = outputs[ii].size(3);
        int group = inpCn / blobs[0].channels();
        int outGroupCn = outCn / group;

        XMat inpMat = inputs[ii]->getRefConst<XMat>();
        XMat outMat = reshaped(outputs[ii].getRef<XMat>(), Shape(numImg*group*outGroupCn, outH*outW));

        for (int n = 0; n < numImg; n++)
        {
            for (int g = 0; g < group; g++)
            {
                XMat colMat, curInp = slice(inpMat, n, _Range(g * inpGroupCn, inpGroupCn));

                im2row(curInp, colMat, inputs[ii]->shape(), outputs[ii].shape());

                _Range kerRange(g * outGroupCn, outGroupCn);
                XMat kerMat = weightsMat.rowRange(kerRange);

                _Range outRange((g + n * group) * outGroupCn, outGroupCn);
                XMat dstMat = outMat.rowRange(outRange);

                dnn::gemm(kerMat, colMat, 1, dstMat, 0, GEMM_2_T);

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

void ConvolutionLayerImpl::im2col(const UMat &srcImg, UMat &dstCol, const BlobShape& inShape, const BlobShape& outShape)
{
    int outH = outShape[2], outW = outShape[3];
    int inpGroupCn = blobs[0].channels();
    int ksize = inpGroupCn * kernel.height * kernel.width;
    if (is1x1())
    {
        dstCol = reshaped(srcImg, Shape(ksize, outH*outW));
        return;
    }
#ifdef HAVE_OPENCL
    int inpH = inShape[2];
    int inpW = inShape[3];
    CV_Assert(im2col_ocl(srcImg, inpGroupCn, inpH, inpW, kernel.height, kernel.width, pad.height, pad.width, stride.height, stride.width, dilation.height, dilation.width, this->colRowBlob.umatRef()));
    dstCol = this->colRowBlob.umatRefConst();
#else
    CV_Error(Error::StsInternal, "");
    dstCol = srcImg; //supress warning
#endif
}

void ConvolutionLayerImpl::im2col(const Mat &srcImg, Mat &dstCol, const BlobShape& inShape, const BlobShape& outShape)
{
    int inpH = inShape[2];
    int inpW = inShape[3];
    int outH = outShape[2], outW = outShape[3];
    int inpGroupCn = blobs[0].channels();
    int ksize = inpGroupCn * kernel.height * kernel.width;
    if (is1x1())
    {
        dstCol = reshaped(srcImg, Shape(ksize, outH*outW));
        return;
    }

    Mat &colMat = colRowBlob.matRef();
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

void ConvolutionLayerImpl::im2row(const  Mat &srcImg,  Mat &dstRow, const BlobShape& inShape, const BlobShape& outShape)
{
    int inpH = inShape[2];
    int inpW = inShape[3];
    int outH = outShape[2], outW = outShape[3];
    int inpGroupCn = blobs[0].channels();
    int ksize = inpGroupCn * kernel.height * kernel.width;

    if (is1x1())
    {
        dstRow = reshaped(srcImg, Shape(ksize, outH*outW)).t();
        return;
    }

    Mat &colMat = colRowBlob.matRef();
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

void ConvolutionLayerImpl::im2row(const UMat &srcImg, UMat &dstCol, const BlobShape& inShape, const BlobShape& outShape)
{
    CV_Error(cv::Error::StsNotImplemented, "");
}

long ConvolutionLayerImpl::getFLOPS(const std::vector<BlobShape> &inputs,
                                    const std::vector<BlobShape> &outputs) const
{
    CV_Assert(inputs.size() == outputs.size());

    long flops = 0;
    for (int i = 0; i < inputs.size(); i++)
    {
        flops += outputs[i].total()*(2*kernel.area()*inputs[i][1] + 1);
    }

    return flops;
}

//Deconvolution

BlobShape DeConvolutionLayerImpl::computeColRowShape(const BlobShape &inpShape,
                                                     const BlobShape &outShape) const
{
    int inpCn = inpShape[1];
    int inpH = inpShape[2];
    int inpW = inpShape[3];
    int outCn = outShape[1];
    int group = inpCn / blobs[0].channels();
    int outGroupCn = outCn / group;
    int ksize = outGroupCn * kernel.height * kernel.width;
    return BlobShape(ksize, inpH * inpW);
}

void DeConvolutionLayerImpl::getOutShapes(const std::vector<BlobShape> &inputs,
                                          std::vector<BlobShape> &outputs, const int requiredOutputs) const
{
    CV_Assert(!bias || blobs[1].total() == (size_t)blobs[0].num());
    CV_Assert(inputs.size() != 0);

    int inpCn = inputs[0][1];
    int inpH = inputs[0][2];
    int inpW = inputs[0][3];

    int outH = stride.height * (inpH - 1) + kernel.height - 2 * pad.height + adjustPad.height;
    int outW = stride.width * (inpW - 1) + kernel.width - 2 * pad.width + adjustPad.width;
    int outCn = blobs[0].num();

    int group = inpCn / blobs[0].channels();

    CV_Assert(inpCn % group == 0 && outCn % group == 0);
    CV_Assert(blobs[0].num() == outCn && blobs[0].channels() == inpCn / group);

    BlobShape outShape(inputs[0][0], outCn, outH, outW);
    outputs.resize(inputs.size(), outShape);
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
    CV_Assert(inputs.size() == outputs.size());

    int outCn = blobs[0].num();
    int inpCn = inputs[0]->size(1);
    int group = inpCn / blobs[0].channels();
    int outGroupCn = outCn / group;
    int ksize = outGroupCn * kernel.height * kernel.width;

    XMat weightsMat = reshaped(blobs[0].getRefConst<XMat>(), Shape(inpCn, ksize));
    XMat biasesMat  = (bias) ? reshaped(blobs[1].getRefConst<XMat>(), Shape(outCn, 1)) : XMat();

    for (size_t ii = 0; ii < outputs.size(); ii++)
    {
        int inpH = inputs[ii]->size(2);
        int inpW = inputs[ii]->size(3);
        int outH = outputs[ii].size(2), outW = outputs[ii].size(3);
        int group = inpCn / blobs[0].channels();
        int inpGroupCn = blobs[0].channels();
        int outGroupCn = outCn / group;

        int numImg = inputs[ii]->size(0);
        XMat convBlob = reshaped(inputs[ii]->getRefConst<XMat>(), Shape(numImg*inpCn, inpH*inpW));
        XMat decnBlob = reshaped(outputs[ii].getRef<XMat>(), Shape(numImg*outCn, outH*outW));

        for (int n = 0; n < numImg; n++)
        {
            for (int g = 0; g < group; g++)
            {
                XMat dstMat = decnBlob.rowRange(_Range((g + n * group) * outGroupCn, outGroupCn));
                XMat &colMat = (is1x1()) ? dstMat : colRowBlob.getRef<XMat>();

                XMat convMat = convBlob.rowRange(_Range((g + n * group) * inpGroupCn, inpGroupCn));
                XMat wghtMat = weightsMat.rowRange(_Range(g * inpGroupCn, inpGroupCn));

                dnn::gemm(wghtMat, convMat, 1, colMat, 0, GEMM_1_T);

                if (!is1x1())
                    col2im(colMat, dstMat, inputs[ii]->shape(), outputs[ii].shape());

                if (bias)
                {
                    XMat curBiasMat = biasesMat.rowRange(_Range(g * outGroupCn, outGroupCn));
                    dnn::gemm(curBiasMat, biasOnesBlob.getRefConst<XMat>(), 1, dstMat, 1);
                }
            }
        }
    }
}

void DeConvolutionLayerImpl::col2im(const Mat &colMat, Mat &dstImg, const BlobShape& inShape, const BlobShape& outShape)
{
    int outCn = outShape[1], outH = outShape[2], outW = outShape[3];
    int inpCn = inShape[1];
    int group = inpCn / blobs[0].channels();
    int outGroupCn = outCn / group;
    if (is1x1())
    {
        dstImg = colMat;
        return;
    }
    if (dstImg.type() == CV_32F)
        col2im_CpuPBody<float>::run(colMat.ptr<float>(), outGroupCn, outH, outW, kernel.height, kernel.width, pad.height, pad.width, stride.height, stride.width, dstImg.ptr<float>());
    if (dstImg.type() == CV_64F)
        col2im_CpuPBody<double>::run(colMat.ptr<double>(), outGroupCn, outH, outW, kernel.height, kernel.width, pad.height, pad.width, stride.height, stride.width, dstImg.ptr<double>());
}

void DeConvolutionLayerImpl::col2im(const UMat &colMat, UMat &dstImg, const BlobShape& inShape, const BlobShape& outShape)
{
    if (is1x1())
    {
        dstImg = colMat;
        return;
    }
#ifdef HAVE_OPENCL
    int outCn = outShape[1], outH = outShape[2], outW = outShape[3];
    int inpCn = inShape[1];
    int group = inpCn / blobs[0].channels();
    int outGroupCn = outCn / group;
    CV_Assert(col2im_ocl(colMat, outGroupCn, outH, outW, kernel.height, kernel.width, pad.height, pad.width, stride.height, stride.width, dstImg));
#else
    CV_Error(Error::StsInternal, "");
    dstImg = colMat;
#endif
}

long DeConvolutionLayerImpl::getFLOPS(const std::vector<BlobShape> &inputs,
                                      const std::vector<BlobShape> &outputs) const
{
    CV_Assert(inputs.size() == outputs.size());

    float flops = 0;
    int outChannels = blobs[0].num();

    for (int i = 0; i < inputs.size(); i++)
    {
        flops += 2*outChannels*kernel.area()*inputs[i].total();
    }

    return flops;
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

Ptr<BaseConvolutionLayer> DeconvolutionLayer::create(Size kernel, Size stride, Size pad, Size dilation, Size adjustPad)
{
    DeConvolutionLayerImpl *l = new DeConvolutionLayerImpl();
    l->kernel = kernel;
    l->pad = pad;
    l->stride = stride;
    l->dilation = dilation;
    l->adjustPad = adjustPad;

    return Ptr<BaseConvolutionLayer>(l);
}

}
}
