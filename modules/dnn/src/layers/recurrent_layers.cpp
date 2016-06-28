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
#include <cmath>

namespace cv
{
namespace dnn
{

template<typename Dtype>
static void tanh(const Mat &src, Mat &dst)
{
    MatConstIterator_<Dtype> itSrc = src.begin<Dtype>();
    MatIterator_<Dtype> itDst = dst.begin<Dtype>();

    for (; itSrc != src.end<Dtype>(); itSrc++, itDst++)
        *itDst = std::tanh(*itSrc);
}

static void tanh(const Mat &src, Mat &dst)
{
    dst.create(src.dims, (const int*)src.size, src.type());

    if (src.type() == CV_32F)
        tanh<float>(src, dst);
    else if (src.type() == CV_64F)
        tanh<double>(src, dst);
    else
        CV_Error(Error::StsUnsupportedFormat, "Functions supports only floating point types");
}

static void sigmoid(const Mat &src, Mat &dst)
{
    cv::exp(-src, dst);
    cv::pow(1 + dst, -1, dst);
}

class LSTMLayerImpl : public LSTMLayer
{
    int numOut, numTimeStamps, numSamples, numInp;
    Mat hInternal, cInternal;
    Mat gates, dummyOnes;
    int dtype;
    bool allocated;

    bool useTimestampDim;
    bool produceCellOutput;

public:

    LSTMLayerImpl()
    {
        type = "LSTM";
        useTimestampDim = true;
        produceCellOutput = false;
        allocated = false;
    }

    void setUseTimstampsDim(bool use)
    {
        CV_Assert(!allocated);
        useTimestampDim = use;
    }

    void setProduceCellOutput(bool produce)
    {
        CV_Assert(!allocated);
        produceCellOutput = produce;
    }

    void setC(const Blob &C)
    {
        CV_Assert(!allocated || C.total() == cInternal.total());
        C.matRefConst().copyTo(cInternal);
    }

    void setH(const Blob &H)
    {
        CV_Assert(!allocated || H.total() == hInternal.total());
        H.matRefConst().copyTo(hInternal);
    }

    Blob getC() const
    {
        CV_Assert(!cInternal.empty());

        //TODO: add convinient Mat -> Blob constructor
        Blob res;
        res.fill(BlobShape::like(cInternal), cInternal.type(), cInternal.data);
        return res;
    }

    Blob getH() const
    {
        CV_Assert(!hInternal.empty());

        Blob res;
        res.fill(BlobShape::like(hInternal), hInternal.type(), hInternal.data);
        return res;
    }

    void setWeights(const Blob &Wh, const Blob &Wx, const Blob &bias)
    {
        CV_Assert(Wh.dims() == 2 && Wx.dims() == 2);
        CV_Assert(Wh.size(0) == Wx.size(0));
        CV_Assert(Wh.size(0) == 4*Wh.size(1));
        CV_Assert(Wh.size(0) == (int)bias.total());
        CV_Assert(Wh.type() == Wx.type() && Wx.type() == bias.type());

        blobs.resize(3);
        blobs[0] = Wh;
        blobs[1] = Wx;
        blobs[2] = bias;
        blobs[2].reshape(BlobShape(1, (int)bias.total()));
    }

    void allocate(const std::vector<Blob*> &input, std::vector<Blob> &output)
    {
        CV_Assert(blobs.size() == 3);
        Blob &Wh = blobs[0], &Wx = blobs[1];

        numOut = Wh.size(1);
        numInp = Wx.size(1);

        CV_Assert(input.size() == 1);
        CV_Assert(input[0]->dims() > 2 && (int)input[0]->total(2) == numInp);

        numTimeStamps = input[0]->size(0);
        numSamples = input[0]->size(1);
        dtype = input[0]->type();

        CV_Assert(dtype == CV_32F || dtype == CV_64F);
        CV_Assert(Wh.type() == dtype);

        BlobShape outShape(numTimeStamps, numSamples, numOut);
        output.resize(2);
        output[0].create(outShape, dtype);
        output[1].create(outShape, dtype);

        hInternal.create(numSamples, numOut, dtype);
        hInternal.setTo(0);

        cInternal.create(numSamples, numOut, dtype);
        cInternal.setTo(0);

        gates.create(numSamples, 4*numOut, dtype);

        dummyOnes.create(numSamples, 1, dtype);
        dummyOnes.setTo(1);

        allocated = true;
    }

    void forward(std::vector<Blob*> &input, std::vector<Blob> &output)
    {
        const Mat &Wh = blobs[0].matRefConst();
        const Mat &Wx = blobs[1].matRefConst();
        const Mat &bias = blobs[2].matRefConst();

        int numSamplesTotal = numTimeStamps*numSamples;
        Mat xTs = input[0]->reshaped(BlobShape(numSamplesTotal, numInp)).matRefConst();

        BlobShape outMatShape(numSamplesTotal, numOut);
        Mat hOutTs = output[0].reshaped(outMatShape).matRef();
        Mat cOutTs = (produceCellOutput) ? output[1].reshaped(outMatShape).matRef() : Mat();

        for (int ts = 0; ts < numTimeStamps; ts++)
        {
            Range curRowRange(ts*numSamples, (ts + 1)*numSamples);
            Mat xCurr = xTs.rowRange(curRowRange);

            gemmCPU(xCurr, Wx, 1, gates, 0, GEMM_2_T);      // Wx * x_t
            gemmCPU(hInternal, Wh, 1, gates, 1, GEMM_2_T);  //+Wh * h_{t-1}
            gemmCPU(dummyOnes, bias, 1, gates, 1);          //+b

            Mat getesIFO = gates.colRange(0, 3*numOut);
            Mat gateI = gates.colRange(0*numOut, 1*numOut);
            Mat gateF = gates.colRange(1*numOut, 2*numOut);
            Mat gateO = gates.colRange(2*numOut, 3*numOut);
            Mat gateG = gates.colRange(3*numOut, 4*numOut);

            sigmoid(getesIFO, getesIFO);
            tanh(gateG, gateG);

            //compute c_t
            cv::multiply(gateF, cInternal, gateF);  // f_t (*) c_{t-1}
            cv::multiply(gateI, gateG, gateI);      // i_t (*) g_t
            cv::add(gateF, gateI, cInternal);       // c_t = f_t (*) c_{t-1} + i_t (*) g_t

            //compute h_t
            tanh(cInternal, hInternal);
            cv::multiply(gateO, hInternal, hInternal);

            //save results in output blobs
            hInternal.copyTo(hOutTs.rowRange(curRowRange));
            if (produceCellOutput)
                cInternal.copyTo(cOutTs.rowRange(curRowRange));
        }
    }
};

Ptr<LSTMLayer> LSTMLayer::create()
{
    return Ptr<LSTMLayer>(new LSTMLayerImpl());
}

void LSTMLayer::forward(std::vector<Blob*>&, std::vector<Blob>&)
{
    CV_Error(Error::StsInternal, "This function should be unreached");
}


class RNNLayerImpl : public RNNLayer
{
    int nX, nH, nO, nSamples;
    int dtype;
    Mat Whh, Wxh, bh;
    Mat Who, bo;
    Mat hPrevInternal, dummyBiasOnes;

public:

    RNNLayerImpl()
    {
        type = "RNN";
    }

    void setWeights(const Blob &W_hh, const Blob &W_xh, const Blob &b_h, const Blob &W_ho, const Blob &b_o)
    {
        CV_Assert(W_hh.dims() == 2 && W_xh.dims() == 2);
        CV_Assert(W_hh.size(0) == W_xh.size(0) && W_hh.size(0) == W_hh.size(1) && (int)b_h.total() == W_xh.size(0));
        CV_Assert(W_ho.size(0) == (int)b_o.total());
        CV_Assert(W_ho.size(1) == W_hh.size(1));

        blobs.resize(5);
        blobs[0] = W_hh;
        blobs[1] = W_xh;
        blobs[2] = b_h;
        blobs[3] = W_ho;
        blobs[4] = b_o;
    }

    void allocate(const std::vector<Blob*> &input, std::vector<Blob> &output)
    {
        CV_Assert(input.size() >= 1 && input.size() <= 2);

        Whh = blobs[0].matRefConst();
        Wxh = blobs[1].matRefConst();
        bh  = blobs[2].matRefConst();
        Who = blobs[3].matRefConst();
        bo  = blobs[4].matRefConst();

        nH = Wxh.rows;
        nX = Wxh.cols;
        nO = Who.rows;

        CV_Assert(input[0]->size(-1) == Wxh.cols);
        nSamples = input[0]->total(0, input[0]->dims() - 1);
        BlobShape xShape = input[0]->shape();
        BlobShape hShape = xShape;
        BlobShape oShape = xShape;
        hShape[-1] = nH;
        oShape[-1] = nO;

        if (input.size() == 2)
        {
            CV_Assert(input[1]->shape() == hShape);
        }
        else
        {
            hPrevInternal.create(nSamples, nH, input[0]->type());
            hPrevInternal.setTo(0);
        }

        output.resize(2);
        output[0].create(oShape, input[0]->type());
        output[1].create(hShape, input[0]->type());

        dummyBiasOnes.create(nSamples, 1, bh.type());
        dummyBiasOnes.setTo(1);
        bh = bh.reshape(1, 1); //is 1 x nH mat
        bo = bo.reshape(1, 1); //is 1 x nO mat
    }

    void forward(std::vector<Blob*> &input, std::vector<Blob> &output)
    {
        Mat xCurr = input[0]->matRefConst();
        Mat hPrev = (input.size() >= 2) ? input[1]->matRefConst() : hPrevInternal;
        Mat oCurr = output[0].matRef();
        Mat hCurr = output[1].matRef();

        //TODO: Check types

        int xsz[] = {nSamples, nX};
        int hsz[] = {nSamples, nH};
        int osz[] = {nSamples, nO};
        if (xCurr.dims != 2) xCurr = xCurr.reshape(1, 2, xsz);
        if (hPrev.dims != 2) hPrev = hPrev.reshape(1, 2, hsz);
        if (oCurr.dims != 2) oCurr = oCurr.reshape(1, 2, osz);
        if (hCurr.dims != 2) hCurr = hCurr.reshape(1, 2, hsz);

        gemmCPU(hPrev, Whh, 1, hCurr, 0, GEMM_2_T); // W_{hh} * h_{prev}
        gemmCPU(xCurr, Wxh, 1, hCurr, 1, GEMM_2_T); //+W_{xh} * x_{curr}
        gemmCPU(dummyBiasOnes, bh, 1, hCurr, 1);    //+bh
        tanh(hCurr, hCurr);

        gemmCPU(hPrev, Who, 1, oCurr, 0, GEMM_2_T); // W_{ho} * h_{prev}
        gemmCPU(dummyBiasOnes, bo, 1, oCurr, 1);    //+b_o
        tanh(oCurr, oCurr);

        if (input.size() < 2) //save h_{prev}
            hCurr.copyTo(hPrevInternal);
    }
};

void RNNLayer::forward(std::vector<Blob*>&, std::vector<Blob>&)
{
    CV_Error(Error::StsInternal, "This function should be unreached");
}

CV_EXPORTS_W Ptr<RNNLayer> RNNLayer::create()
{
    return Ptr<RNNLayer>(new RNNLayerImpl());
}

}
}
