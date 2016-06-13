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
#include "recurrent_layers.hpp"
#include "op_blas.hpp"
#include <iostream>

namespace cv
{
namespace dnn
{

class LSTMLayerImpl : public LSTMLayer
{
public:

    LSTMLayerImpl()
    {
        type = "LSTM";
    }

    int nH, nX, nC, numSamples;
    Mat prevH, prevC;
    Mat gates, dummyOnes;

    void setWeights(const Blob &Wh, const Blob &Wx, const Blob &bias)
    {
        CV_Assert(Wh.dims() == 2 && Wx.dims() == 2);
        CV_Assert(Wh.size(0) == Wx.size(0) && Wh.size(0) % 4 == 0);
        CV_Assert(Wh.size(0) == bias.total());

        blobs.resize(3);
        blobs[0] = Wh;
        blobs[1] = Wx;
        blobs[2] = bias;
    }

    void allocate(const std::vector<Blob*> &input, std::vector<Blob> &output)
    {
        CV_Assert(blobs.size() == 3);
        Blob &Wh = blobs[0], &Wx = blobs[1];
        
        nH = Wh.size(1);
        nX = Wx.size(1);
        nC = Wh.size(0) / 4;

        CV_Assert(input.size() >= 1 && input.size() <= 3);
        CV_Assert(input[0]->size(-1) == nX);
        
        BlobShape inpShape = input[0]->shape();
        numSamples = input[0]->total(0, input[0]->dims()-1);

        BlobShape hShape = inpShape;
        hShape[-1] = nH;
        BlobShape cShape = inpShape;
        cShape[-1] = nC;

        output.resize(2);
        output[0].create(hShape, input[0]->type());
        output[1].create(cShape, input[0]->type());

        if (input.size() < 2)
        {
            prevH.create(numSamples, nH, input[0]->type());
            prevH.setTo(0);
        }
        else
            CV_Assert(input[1]->shape() == hShape);

        if (input.size() < 3)
        {
            prevC.create(numSamples, nC, input[0]->type());
            prevC.setTo(0);
        }
        else
            CV_Assert(input[2]->shape() == cShape);

        gates.create(numSamples, 4*nC, input[0]->type());
        dummyOnes.create(numSamples, 1, input[0]->type());
        dummyOnes.setTo(1);
    }

    Mat ep, em;
    void tanh(Mat &x, Mat &d)
    {
        cv::exp(-x, em);
        cv::exp( x, ep);
        cv::divide(ep - em, ep + em, d);
    }

    void sigmoid(Mat &x)
    {
        cv::exp(-x, x);
        cv::pow(1 + x, -1, x);
    }

    void forward(std::vector<Blob*> &input, std::vector<Blob> &output)
    {
        CV_DbgAssert(blobs.size() == 3);
        const Mat &Wh = blobs[0].matRefConst(), &Wx = blobs[1].matRefConst();
        Mat bias = blobs[2].matRefConst().reshape(1, 1);
        CV_DbgAssert(Wh.type() == CV_32F && Wx.type() == CV_32F && bias.type() == CV_32F);

        int szx[] = { numSamples, nX };
        int szc[] = { numSamples, nC };
        Mat xCurr = input[0]->matRefConst().reshape(1, 2, szx);
        Mat hPrev = (input.size() >= 2) ? input[1]->matRefConst().reshape(1, 2, szc) : prevH;
        Mat cPrev = (input.size() >= 3) ? input[2]->matRefConst().reshape(1, 2, szc) : prevC;
        CV_Assert(xCurr.type() == CV_32F && hPrev.type() == CV_32F && cPrev.type() == CV_32F);

        Mat hCurr = output[0].matRef().reshape(1, 2, szc);
        Mat cCurr = output[1].matRef().reshape(1, 2, szc);
        CV_Assert(hCurr.type() == CV_32F && cCurr.type() == CV_32F);

        gemmCPU(xCurr, Wx, 1, gates, 0, GEMM_2_T); // Wx * x_t
        gemmCPU(hPrev, Wh, 1, gates, 1, GEMM_2_T); //+Wh * h_{t-1}
        gemmCPU(dummyOnes, bias, 1, gates, 1);     //+b

        Mat gatesDiv = gates.reshape(1, 4*numSamples);
        Mat getesIFO = gatesDiv(Range(0, 3*numSamples), Range::all());
        Mat gateI = gatesDiv(Range(0*numSamples, 1*numSamples), Range::all());
        Mat gateF = gatesDiv(Range(1*numSamples, 2*numSamples), Range::all());
        Mat gateO = gatesDiv(Range(2*numSamples, 3*numSamples), Range::all());
        Mat gateG = gatesDiv(Range(3*numSamples, 4*numSamples), Range::all());
        
        sigmoid(getesIFO);
        tanh(gateG, gateG);

        cv::add(gateF.mul(cPrev), gateI.mul(gateG), cCurr);
        
        tanh(cCurr, hCurr);
        cv::multiply(gateO, hCurr, hCurr);

        //save answers for next iteration
        if (input.size() <= 2)
            hCurr.copyTo(hPrev);
        if (input.size() <= 3)
            cCurr.copyTo(cPrev);
    }
};

Ptr<LSTMLayer> LSTMLayer::create()
{
    return Ptr<LSTMLayer>(new LSTMLayerImpl());
}

void LSTMLayer::forward(std::vector<Blob*> &input, std::vector<Blob> &output)
{
    CV_Error(Error::StsNotImplemented, "This function should be unreached");
}

class RNNLayerImpl : public RNNLayer
{
    int nX, nH;
    Mat Whh, Wxh, bh;
    Mat Who, bo;

public:

    RNNLayerImpl()
    {
        type = "RNN";
    }

    void setWeights(const Blob &Whh, const Blob &Wxh, const Blob &bh, const Blob &Who, const Blob &bo)
    {
        CV_Assert(Whh.dims() == 2 && Wxh.dims() == 2);
        CV_Assert(Whh.size(0) == Wxh.size(0) && Whh.size(0) == Whh.size(1) && bh.total() == Wxh.size(0));
        CV_Assert(Who.size(0) == bo.total());
        CV_Assert(Who.size(1) == Whh.size(1));

        blobs.reserve(5);
        blobs[0] = Whh;
        blobs[1] = Wxh;
        blobs[2] = bh;
        blobs[3] = Who;
        blobs[4] = bo;
    }

    void allocate(const std::vector<Blob*> &input, std::vector<Blob> &output)
    {

    }

    void tanh(Mat &x)
    {

    }

    void forward(std::vector<Blob*> &input, std::vector<Blob> &output)
    {

    }
};

}
}