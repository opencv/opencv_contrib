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
#include <iostream>

namespace cv
{
namespace dnn
{

class BaseConvolutionLayerImpl : public ConvolutionLayer
{
public:
    BaseConvolutionLayerImpl()
    {
#ifdef HAVE_LAPACK
        int nthreads = cv::getThreadNum();
        if (getBlasThreads() != nthreads)
        {
            setBlasThreads(nthreads);
        }
#endif
    }
    void finalize(const std::vector<Mat*> &inputs, std::vector<Mat> &outputs)
    {
        CV_Assert(inputs.size() > 0);

        CV_Assert(blobs.size() >= 1 && blobs.size() <= 2);
        CV_Assert(blobs[0].dims == 4 && blobs[0].size[3] == kernel.width && blobs[0].size[2] == kernel.height);

        const Mat &input = *inputs[0];
        CV_Assert(input.dims == 4 && (input.type() == CV_32F || input.type() == CV_64F));
        for (size_t i = 0; i < inputs.size(); i++)
        {
            CV_Assert(inputs[i]->type() == input.type());
            CV_Assert(inputs[i]->dims == 4 && inputs[i]->size[1] == input.size[1]);
            CV_Assert(inputs[i]->size[2] == input.size[2] && inputs[i]->size[3] == input.size[3]);
        }

        Size outSize = Size(outputs[0].size[3], outputs[0].size[2]);
        getConvPoolPaddings(Size(input.size[3], input.size[2]), outSize,
                kernel, stride, padMode, pad);
    }

    bool hasBias() const
    {
        return blobs.size() >= 2;
    }

    virtual MatShape computeColRowShape(const MatShape &inpShape, const MatShape &outShape) const = 0;
    bool is1x1() const
    {
        return (kernel.height == 1 && kernel.width == 1) &&
        (stride.height == 1 && stride.width == 1) &&
        (dilation.height == 1 && dilation.width == 1);
    }
};

//TODO: simultaneously convolution and bias addition for cache optimization
class ConvolutionLayerImpl : public BaseConvolutionLayerImpl
{
public:
    MatShape computeColRowShape(const MatShape &inpShape, const MatShape &outShape) const
    {
        Size out(outShape[3], outShape[2]);
        int inpGroupCn = blobs[0].size[1];
        int ksize = inpGroupCn * kernel.height * kernel.width;
        return shape(out.area(), ksize);
    }

    bool getMemoryShapes(const std::vector<MatShape> &inputs,
                         const int requiredOutputs,
                         std::vector<MatShape> &outputs,
                         std::vector<MatShape> &internals) const
    {
        CV_Assert(blobs.size() != 0);
        CV_Assert(!hasBias() || blobs[1].total() == (size_t)blobs[0].size[0]);
        CV_Assert(inputs.size() != 0);

        internals.clear();

        int inpCn = inputs[0][1];
        int inpH = inputs[0][2];
        int inpW = inputs[0][3];

        int outCn = blobs[0].size[0];
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

        int group = inpCn / blobs[0].size[1];

        CV_Assert(inpCn % group == 0 && outCn % group == 0);
        CV_Assert(blobs[0].size[0] == outCn);

        int dims[] = {inputs[0][0], outCn, out.height, out.width};
        outputs.resize(inputs.size(), shape(dims));

        internals.push_back(MatShape());
        if (!is1x1())
            internals[0] = computeColRowShape(inputs[0], outputs[0]);

        if (hasBias())
            internals.push_back(shape(1, out.area()));

        return false;
    }

    void forward(std::vector<Mat*> &inputs, std::vector<Mat> &outputs, std::vector<Mat> &internals)
    {
        CV_Assert(inputs.size() > 0);

        internals[0].setTo(0);

        if (hasBias())
            internals[1].setTo(1);

        int outCn = blobs[0].size[0];
        int inpCn = inputs[0]->size[1];
        int inpGroupCn = blobs[0].size[1];

        Mat weightsMat = blobs[0].reshape(1, outCn);
        Mat biasesMat  = hasBias() ? blobs[1].reshape(1, outCn) : Mat();

        for (size_t ii = 0; ii < outputs.size(); ii++)
        {
            int numImg = inputs[ii]->size[0];
            int group = inpCn / blobs[0].size[1];
            int outGroupCn = outCn / group;
            Mat inpMat = *inputs[ii];
            Mat outMat = outputs[ii].reshape(1, numImg*group*outGroupCn);

            for (int n = 0; n < numImg; n++)
            {
                for (int g = 0; g < group; g++)
                {
                    Mat curInp = slice(inpMat, n, _Range(g * inpGroupCn, inpGroupCn));

                    im2row(curInp, internals[0], shape(inpMat), shape(outputs[ii]));

                    _Range kerRange(g * outGroupCn, outGroupCn);
                    Mat kerMat = weightsMat.rowRange(kerRange);

                    _Range outRange((g + n * group) * outGroupCn, outGroupCn);
                    Mat dstMat = outMat.rowRange(outRange);

                    dnn::gemm(kerMat, internals[0], 1, dstMat, 0, GEMM_2_T);

                    if (hasBias())
                    {
                        dnn::gemm(biasesMat.rowRange(kerRange), internals[1], 1, dstMat, 1);
                    }
                }
            }
        }
    }

    void im2row(const  Mat &srcImg, Mat &dstRow, const MatShape& inShape, const MatShape& outShape)
    {
        int inpH = inShape[2];
        int inpW = inShape[3];
        int outH = outShape[2], outW = outShape[3];
        int inpGroupCn = blobs[0].size[1];
        int ksize = inpGroupCn * kernel.height * kernel.width;

        if (is1x1())
        {
            transpose(srcImg.reshape(1, ksize), dstRow);
        }
        else
        {
            cv::dnn::im2row(srcImg.ptr<float>(), inpGroupCn, inpH, inpW, kernel.height,
                            kernel.width, pad.height, pad.width, stride.height, stride.width,
                            dilation.height, dilation.width, outH, outW, dstRow.ptr<float>());
        }
    }

    virtual int64 getFLOPS(const std::vector<MatShape> &inputs,
                           const std::vector<MatShape> &outputs) const
    {
        CV_Assert(inputs.size() == outputs.size());

        int64 flops = 0;
        for (int i = 0; i < inputs.size(); i++)
        {
            flops += total(outputs[i])*(2*kernel.area()*inputs[i][1] + 1);
        }

        return flops;
    }
};

class DeConvolutionLayerImpl : public BaseConvolutionLayerImpl
{
public:
    MatShape computeColRowShape(const MatShape &inpShape, const MatShape &outShape) const
    {
        int inpCn = inpShape[1];
        int inpH = inpShape[2];
        int inpW = inpShape[3];
        int outCn = outShape[1];
        int group = inpCn / blobs[0].size[1];
        int outGroupCn = outCn / group;
        int ksize = outGroupCn * kernel.height * kernel.width;
        return shape(ksize, inpH * inpW);
    }

    bool getMemoryShapes(const std::vector<MatShape> &inputs,
                         const int requiredOutputs,
                         std::vector<MatShape> &outputs,
                         std::vector<MatShape> &internals) const
    {
        CV_Assert(!hasBias() || blobs[1].total() == (size_t)blobs[0].size[0]);
        CV_Assert(inputs.size() != 0);

        int inpCn = inputs[0][1];
        int inpH = inputs[0][2];
        int inpW = inputs[0][3];

        int outH = stride.height * (inpH - 1) + kernel.height - 2 * pad.height + adjustPad.height;
        int outW = stride.width * (inpW - 1) + kernel.width - 2 * pad.width + adjustPad.width;
        int outCn = blobs[0].size[0];

        int group = inpCn / blobs[0].size[1];

        CV_Assert(inpCn % group == 0 && outCn % group == 0);
        CV_Assert(blobs[0].size[0] == outCn && blobs[0].size[1] == inpCn / group);

        int dims[] = {inputs[0][0], outCn, outH, outW};
        outputs.resize(inputs.size(), shape(dims));

        internals.push_back(MatShape());
        if (!is1x1())
            internals[0] = computeColRowShape(inputs[0], outputs[0]);

        if (hasBias())
            internals.push_back(shape(1, outH*outW));

        return false;
    }


    void forward(std::vector<Mat *> &inputs, std::vector<Mat> &outputs, std::vector<Mat> &internals)
    {
        internals[0].setTo(0);
        if (hasBias())
            internals[1].setTo(1);

        int outCn = blobs[0].size[0];
        int inpCn = inputs[0]->size[1];
        Mat weightsMat = blobs[0].reshape(1, inpCn);
        Mat biasesMat  = hasBias() ? blobs[1].reshape(1, outCn) : Mat();

        for (size_t ii = 0; ii < outputs.size(); ii++)
        {
            int group = inpCn / blobs[0].size[1];
            int inpGroupCn = blobs[0].size[1];
            int outGroupCn = outCn / group;
            int numImg = inputs[ii]->size[0];

            Mat convBlob = inputs[ii]->reshape(1, numImg*inpCn);
            Mat decnBlob = outputs[ii].reshape(1, numImg*outCn);

            for (int n = 0; n < numImg; n++)
            {
                for (int g = 0; g < group; g++)
                {
                    Mat dstMat = decnBlob.rowRange(_Range((g + n * group) * outGroupCn, outGroupCn));
                    Mat &colMat = (is1x1()) ? dstMat : internals[0];

                    Mat convMat = convBlob.rowRange(_Range((g + n * group) * inpGroupCn, inpGroupCn));
                    Mat wghtMat = weightsMat.rowRange(_Range(g * inpGroupCn, inpGroupCn));

                    dnn::gemm(wghtMat, convMat, 1, colMat, 0, GEMM_1_T);

                    if (!is1x1())
                        col2im(colMat, dstMat, shape(*inputs[ii]), shape(outputs[ii]));

                    if (hasBias())
                    {
                        Mat curBiasMat = biasesMat.rowRange(_Range(g * outGroupCn, outGroupCn));
                        dnn::gemm(curBiasMat, internals[1], 1, dstMat, 1);
                    }
                }
            }
        }
    }

    void col2im(const Mat &colMat, Mat &dstImg, const MatShape& inShape, const MatShape& outShape)
    {
        int outCn = outShape[1], outH = outShape[2], outW = outShape[3];
        int inpCn = inShape[1];
        int group = inpCn / blobs[0].size[1];
        int outGroupCn = outCn / group;

        if (is1x1())
        {
            dstImg = colMat;
            return;
        }
        cv::dnn::col2im(colMat.ptr<float>(), outGroupCn, outH, outW, kernel.height, kernel.width,
                        pad.height, pad.width, stride.height, stride.width,
                        dilation.height, dilation.width, dstImg.ptr<float>(), &ofsbuf[0]);
    }

    virtual int64 getFLOPS(const std::vector<MatShape> &inputs,
                           const std::vector<MatShape> &outputs) const
    {
        CV_Assert(inputs.size() == outputs.size());

        float flops = 0;
        int outChannels = blobs[0].size[0];

        for (int i = 0; i < inputs.size(); i++)
        {
            flops += 2*outChannels*kernel.area()*total(inputs[i]);
        }

        return flops;
    }

    std::vector<int> ofsbuf;
};

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
