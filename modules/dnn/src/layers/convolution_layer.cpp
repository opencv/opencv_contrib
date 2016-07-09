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
#include <iostream>

namespace cv
{
namespace dnn
{

typedef BlobShape Shape;

template<typename Mat>
void reshape(Mat &m, const BlobShape &shape)
{
    m = m.reshape(1, shape.dims(), shape.ptr());
}

template<typename Mat>
Mat reshaped(const Mat &m, const BlobShape &shape)
{
    return m.reshape(1, shape.dims(), shape.ptr());
}

ConvolutionLayer::ConvolutionLayer(LayerParams &params) : Layer(params)
{
    getKernelParams(params, kerH, kerW, padH, padW, strideH, strideW);

    numOutput = params.get<int>("num_output");
    bias = params.get<bool>("bias_term", true);
    group = params.get<int>("group", 1);
    CV_Assert(numOutput % group == 0);

    CV_Assert(!bias || blobs.size() == 2);
    CV_Assert( bias || blobs.size() == 1);

    const Blob &wgtBlob = blobs[0];
    CV_Assert(wgtBlob.dims() == 4 && wgtBlob.cols() == kerW && wgtBlob.rows() == kerH);

    if (bias)
    {
        Blob &biasBlob = blobs[1];
        CV_Assert(biasBlob.total() == (size_t)numOutput);
    }

    #if HAVE_CBLAS
        if (getBlasThreads() != cv::getThreadNum())
        {
            setBlasThreads(cv::getThreadNum());
        }
    #endif

    tryUseOpenCL = true;
}

void ConvolutionLayer::allocate(const std::vector<Blob*> &inputs, std::vector<Blob> &outputs)
{
    CV_Assert(inputs.size() > 0);

    const Blob &inpBlob = *inputs[0];
    CV_Assert(inpBlob.dims() == 4 && inpBlob.type() == CV_32F);
    computeInpOutShape(inpBlob);

    CV_Assert(inpCn % group == 0 && outCn % group == 0);
    CV_Assert(blobs[0].num() == outCn && blobs[0].channels() == inpCn / group);

    outGroupCn = outCn / group;
    inpGroupCn = inpCn / group;
    ksize = inpGroupCn * kerH * kerW;

    outputs.resize(inputs.size());
    for (size_t i = 0; i < inputs.size(); i++)
    {
        CV_Assert(inputs[i]->type() == inpBlob.type());
        CV_Assert(inputs[i]->dims() == 4 && inputs[i]->channels() == inpBlob.channels());
        CV_Assert(inputs[i]->rows() == inpBlob.rows() && inputs[i]->cols() == inpBlob.cols());

        outputs[i].create(Shape(inputs[i]->num(), topCn, topH, topW));
    }

    #ifdef HAVE_OPENCL
    useOpenCL = ocl::useOpenCL() && tryUseOpenCL;
    #else
    useOpenCL = false;
    #endif

    int allocFlags = useOpenCL ? Blob::ALLOC_BOTH : Blob::ALLOC_MAT;

    if (!is1x1())
    {
        colBlob.create(Shape(ksize, outH * outW), inpBlob.type(), allocFlags);
        colMat = colBlob.matRef();
    }

    if (bias)
    {
        biasOnesBlob.create(Shape(1, topH * topW), inpBlob.type(), allocFlags);
        biasOnesBlob.matRef().setTo(1);
        biasOnesMat = biasOnesBlob.matRefConst();
    }
}

inline bool ConvolutionLayer::is1x1() const
{
    return (kerH == 1 && kerW == 1) && (strideW == 1 && strideH == 1); //hotfix with stride
}

template<typename Mat>
void ConvolutionLayer::forward_(std::vector<Blob*> &inputs, std::vector<Blob> &outputs)
{
    Mat weightsMat = reshaped(blobs[0].getRefConst<Mat>(), Shape(outCn, ksize));
    Mat biasesMat  = reshaped(blobs[1].getRefConst<Mat>(), Shape(outCn, 1));

    for (size_t ii = 0; ii < outputs.size(); ii++)
    {
        Blob &inpBlob = *inputs[ii];
        Blob &outBlob = outputs[ii];
        Mat inpMat = inpBlob.getRefConst<Mat>();
        Mat outMat = reshaped(outBlob.getRef<Mat>(), Shape(inpBlob.num()*group*outGroupCn, outH*outW));

        int outCurrCn = 0;
        for (int n = 0; n < inpBlob.num(); n++)
        {
            int kerCurrCn = 0;
            for (int g = 0; g < group; g++)
            {
                im2col(inpBlob, n, g, colBlob);
                const Mat &colMat = colBlob.getRefConst<Mat>();

                Range kerRange(kerCurrCn, kerCurrCn + outGroupCn);
                Mat kerMat = weightsMat.rowRange(kerRange);

                Range outRange(outCurrCn, outCurrCn + outGroupCn);
                Mat dstMat = outMat.rowRange(outRange);

                dnn::gemm(kerMat, colMat, 1, dstMat, 0);

                if (bias)
                {
                    dnn::gemm(biasesMat.rowRange(kerRange), biasOnesMat, 1, dstMat, 1);
                }

                kerCurrCn += outGroupCn;
                outCurrCn += outGroupCn;
            }
        }
    }
}

void ConvolutionLayer::forward(std::vector<Blob*> &inputs, std::vector<Blob> &outputs)
{
    if (!useOpenCL)
        forward_<Mat>(inputs, outputs);
    else
        forward_<UMat>(inputs, outputs);
}

void ConvolutionLayer::im2col(Blob &inpBlob, int imNum, int cnGroup, Blob &colBlob)
{
#ifdef HAVE_OPENCL
    if (useOpenCL)
    {
        std::vector<Range> ranges(4, Range::all());
        ranges[0] = Range(imNum, imNum+1);
        ranges[1] = Range(cnGroup*inpGroupCn, (cnGroup + 1)*inpGroupCn);

        UMat src = inpBlob.umatRef()(&ranges[0]);
        UMat &dst = colBlob.umatRef();
        im2col_ocl(src, inpGroupCn, inpH, inpW, kerH, kerW, padH, padW, strideH, strideW, dst);
        return;
    }
#endif // HAVE_OPENCL

    Mat &colMat = colBlob.matRef();
    uchar *srcPtr = inpBlob.ptr(imNum, cnGroup*inpGroupCn);

    if (is1x1())
    {
        colMat = Mat(ksize, inpBlob.rows()*inpBlob.cols(), inpBlob.type(), srcPtr);
        return;
    }

    if (inpBlob.type() == CV_32F)
        im2col_CpuPBody<float>::run((float*)srcPtr, inpGroupCn, inpH, inpW, kerH, kerW, padH, padW, strideH, strideW, colMat.ptr<float>());
    if (inpBlob.type() == CV_64F)
        im2col_CpuPBody<double>::run((double*)srcPtr, inpGroupCn, inpH, inpW, kerH, kerW, padH, padW, strideH, strideW, colMat.ptr<double>());
}

void ConvolutionLayer::computeInpOutShape(const Blob &inpBlob)
{
    inpH = inpBlob.rows();
    inpW = inpBlob.cols();
    inpCn = inpBlob.channels();

    outH = (inpH + 2 * padH - kerH) / strideH + 1;
    outW = (inpW + 2 * padW - kerW) / strideW + 1;
    outCn = numOutput;

    topH = outH; topW = outW; topCn = outCn;
}

DeConvolutionLayer::DeConvolutionLayer(LayerParams &params)
    : ConvolutionLayer(params) {}

void DeConvolutionLayer::computeInpOutShape(const Blob &inpBlob)
{
    outH = inpBlob.rows();
    outW = inpBlob.cols();
    outCn = inpBlob.channels();

    inpH = strideH * (outH - 1) + kerH - 2 * padH;
    inpW = strideW * (outW - 1) + kerW - 2 * padW;
    inpCn = numOutput;

    topH = inpH; topW = inpW; topCn = inpCn;
}

void DeConvolutionLayer::forward(std::vector<Blob*> &inputs, std::vector<Blob> &outputs)
{
    Blob &wghtBlob = blobs[0];

    for (size_t ii = 0; ii < outputs.size(); ii++)
    {
        Blob &convBlob = *inputs[ii];
        Blob &decnBlob = outputs[ii];

        for (int n = 0; n < convBlob.num(); n++)
        {
            for (int g = 0; g < group; g++)
            {
                Mat dstMat(inpGroupCn, inpH*inpW, decnBlob.type(), decnBlob.ptr(n, g*inpGroupCn));

                if (is1x1())
                    colMat = dstMat;

                Mat convMat(outGroupCn, outH*outW, convBlob.type(), convBlob.ptr(n, g*outGroupCn));
                Mat wghtMat(outGroupCn, ksize, wghtBlob.type(), wghtBlob.ptr(g*outGroupCn));
                gemmCPU(wghtMat, convMat, 1, colMat, 0, GEMM_1_T);

                col2im(dstMat);

                if (bias)
                {
                    float *biasPtr = blobs[1].ptrf() + g*inpGroupCn;
                    Mat biasMat(inpGroupCn, 1, CV_32F, biasPtr);
                    gemmCPU(biasMat, biasOnesMat, 1, dstMat, 1); //TODO: gemv
                }
            }
        }
    }
}

void DeConvolutionLayer::col2im(Mat &dstMat)
{
    if (is1x1()) return;

    if (dstMat.type() == CV_32F)
        col2im_cpu(colMat.ptr<float>(), inpGroupCn, inpH, inpW, kerH, kerW, padH, padW, strideH, strideW, dstMat.ptr<float>());
    if (dstMat.type() == CV_64F)
        col2im_cpu(colMat.ptr<double>(), inpGroupCn, inpH, inpW, kerH, kerW, padH, padW, strideH, strideW, dstMat.ptr<double>());
}

}
}
