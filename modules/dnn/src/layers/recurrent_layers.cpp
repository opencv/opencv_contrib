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
#include <opencv2/dnn/shape_utils.hpp>

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

//TODO: make utils method
static void tanh(const Mat &src, Mat &dst)
{
    dst.create(src.dims, (const int*)src.size, src.type());

    if (src.type() == CV_32F)
        tanh<float>(src, dst);
    else if (src.type() == CV_64F)
        tanh<double>(src, dst);
    else
        CV_Error(Error::StsUnsupportedFormat, "Function supports only floating point types");
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

    Shape outTailShape;                 //shape of single output sample
    Shape outTsMatShape, outTsShape;    //shape of N output samples
    Shape outResShape;                  //shape of T timestamps and N output samples

    bool useTimestampDim;
    bool produceCellOutput;

public:

    LSTMLayerImpl()
    {
        type = "LSTM";
        useTimestampDim = true;
        produceCellOutput = false;
        allocated = false;
        outTailShape = Shape::empty();
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
        CV_Assert(cInternal.empty() || C.total() == cInternal.total());
        if (!cInternal.empty())
            C.reshaped(Shape::like(cInternal)).matRefConst().copyTo(cInternal);
        else
            C.matRefConst().copyTo(cInternal);
    }

    void setH(const Blob &H)
    {
        CV_Assert(hInternal.empty() || H.total() == hInternal.total());
        if (!hInternal.empty())
            H.reshaped(Shape::like(hInternal)).matRefConst().copyTo(hInternal);
        else
            H.matRefConst().copyTo(hInternal);
    }

    Blob getC() const
    {
        CV_Assert(!cInternal.empty());

        //TODO: add convinient Mat -> Blob constructor
        Blob res(outTsShape, cInternal.type());
        res.fill(res.shape(), res.type(), cInternal.data);
        return res;
    }

    Blob getH() const
    {
        CV_Assert(!hInternal.empty());

        Blob res(outTsShape, hInternal.type());
        res.fill(res.shape(), res.type(), hInternal.data);
        return res;
    }

    void setOutShape(const Shape &outTailShape_)
    {
        CV_Assert(!allocated || outTailShape_.total() == outTailShape.total());
        outTailShape = outTailShape_;
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
        blobs[2].reshape(Shape(1, (int)bias.total()));
    }

    void allocate(const std::vector<Blob*> &input, std::vector<Blob> &output)
    {
        CV_Assert(blobs.size() == 3);
        CV_Assert(input.size() == 1);

        Blob &Wh = blobs[0], &Wx = blobs[1];
        numOut = Wh.size(1);
        numInp = Wx.size(1);

        if (!outTailShape.isEmpty())
            CV_Assert(outTailShape.total() == numOut);
        else
            outTailShape = Shape(numOut);

        if (useTimestampDim)
        {
            CV_Assert(input[0]->dims() >= 2 && (int)input[0]->total(2) == numInp);
            numTimeStamps = input[0]->size(0);
            numSamples = input[0]->size(1);
            outResShape = Shape(numTimeStamps, numSamples) + outTailShape;
        }
        else
        {
            CV_Assert(input[0]->dims() >= 1 && (int)input[0]->total(1) == numInp);
            numTimeStamps = 1;
            numSamples = input[0]->size(0);
            outResShape = Shape(numSamples) + outTailShape;
        }
        outTsMatShape = Shape(numSamples, numOut);
        outTsShape = Shape(numSamples) + outTailShape;

        dtype = input[0]->type();
        CV_Assert(dtype == CV_32F || dtype == CV_64F);
        CV_Assert(Wh.type() == dtype);

        output.resize( (produceCellOutput) ? 2 : 1 );
        output[0].create(outResShape, dtype);
        if (produceCellOutput)
            output[1].create(outResShape, dtype);

        if (hInternal.empty())
        {
            hInternal.create(outTsMatShape.dims(), outTsMatShape.ptr(), dtype);
            hInternal.setTo(0);
        }
        else
        {
            CV_Assert((int)hInternal.total() == numSamples*numOut);
            hInternal = hInternal.reshape(1, outTsMatShape.dims(), outTsMatShape.ptr());
        }

        if (cInternal.empty())
        {
            cInternal.create(outTsMatShape.dims(), outTsMatShape.ptr(), dtype);
            cInternal.setTo(0);
        }
        else
        {
            CV_Assert((int)cInternal.total() == numSamples*numOut);
            cInternal = cInternal.reshape(1, outTsMatShape.dims(), outTsMatShape.ptr());
        }

        gates.create(numSamples, 4*numOut, dtype);

        dummyOnes.create(numSamples, 1, dtype);
        dummyOnes.setTo(1);

        allocated = true;
    }

    void forward(std::vector<Blob*> &input, std::vector<Blob> &output)
    {
        const Mat &Wh = blobs[0].getRefConst<Mat>();
        const Mat &Wx = blobs[1].getRefConst<Mat>();
        const Mat &bias = blobs[2].getRefConst<Mat>();

        int numSamplesTotal = numTimeStamps*numSamples;
        Mat xTs = reshaped(input[0]->getRefConst<Mat>(), Shape(numSamplesTotal, numInp));

        Shape outMatShape(numSamplesTotal, numOut);
        Mat hOutTs = reshaped(output[0].getRef<Mat>(), outMatShape);
        Mat cOutTs = (produceCellOutput) ? reshaped(output[1].getRef<Mat>(), outMatShape) : Mat();

        for (int ts = 0; ts < numTimeStamps; ts++)
        {
            Range curRowRange(ts*numSamples, (ts + 1)*numSamples);
            Mat xCurr = xTs.rowRange(curRowRange);

            dnn::gemm(xCurr, Wx, 1, gates, 0, GEMM_2_T);      // Wx * x_t
            dnn::gemm(hInternal, Wh, 1, gates, 1, GEMM_2_T);  //+Wh * h_{t-1}
            dnn::gemm(dummyOnes, bias, 1, gates, 1);          //+b

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

int LSTMLayer::inputNameToIndex(String inputName)
{
    if (inputName.toLowerCase() == "x")
        return 0;
    return -1;
}

int LSTMLayer::outputNameToIndex(String outputName)
{
    if (outputName.toLowerCase() == "h")
        return 0;
    else if (outputName.toLowerCase() == "c")
        return 1;
    return -1;
}


class RNNLayerImpl : public RNNLayer
{
    int numX, numH, numO;
    int numSamples, numTimestamps, numSamplesTotal;
    int dtype;
    Mat Whh, Wxh, bh;
    Mat Who, bo;
    Mat hCurr, hPrev, dummyBiasOnes;
    bool produceH;

public:

    RNNLayerImpl()
    {
        type = "RNN";
        produceH = false;
    }

    void setProduceHiddenOutput(bool produce = false)
    {
        produceH = produce;
    }

    void setWeights(const Blob &W_xh, const Blob &b_h, const Blob &W_hh, const Blob &W_ho, const Blob &b_o)
    {
        CV_Assert(W_hh.dims() == 2 && W_xh.dims() == 2);
        CV_Assert(W_hh.size(0) == W_xh.size(0) && W_hh.size(0) == W_hh.size(1) && (int)b_h.total() == W_xh.size(0));
        CV_Assert(W_ho.size(0) == (int)b_o.total());
        CV_Assert(W_ho.size(1) == W_hh.size(1));

        blobs.resize(5);
        blobs[0] = W_xh;
        blobs[1] = b_h;
        blobs[2] = W_hh;
        blobs[3] = W_ho;
        blobs[4] = b_o;
    }

    void allocate(const std::vector<Blob*> &input, std::vector<Blob> &output)
    {
        CV_Assert(input.size() >= 1 && input.size() <= 2);

        Wxh = blobs[0].matRefConst();
        bh  = blobs[1].matRefConst();
        Whh = blobs[2].matRefConst();
        Who = blobs[3].matRefConst();
        bo  = blobs[4].matRefConst();

        numH = Wxh.rows;
        numX = Wxh.cols;
        numO = Who.rows;

        CV_Assert(input[0]->dims() >= 2);
        CV_Assert((int)input[0]->total(2) == numX);
        CV_Assert(input[0]->type() == CV_32F || input[0]->type() == CV_64F);
        dtype = input[0]->type();
        numTimestamps = input[0]->size(0);
        numSamples = input[0]->size(1);
        numSamplesTotal = numTimestamps * numSamples;

        hCurr.create(numSamples, numH, dtype);
        hPrev.create(numSamples, numH, dtype);
        hPrev.setTo(0);

        dummyBiasOnes.create(numSamples, 1, dtype);
        dummyBiasOnes.setTo(1);
        bh = bh.reshape(1, 1); //is 1 x numH Mat
        bo = bo.reshape(1, 1); //is 1 x numO Mat

        reshapeOutput(output);
    }

    void reshapeOutput(std::vector<Blob> &output)
    {
        output.resize((produceH) ? 2 : 1);
        output[0].create(Shape(numTimestamps, numSamples, numO), dtype);
        if (produceH)
            output[1].create(Shape(numTimestamps, numSamples, numH), dtype);
    }

    void forward(std::vector<Blob*> &input, std::vector<Blob> &output)
    {
        Mat xTs = reshaped(input[0]->getRefConst<Mat>(), Shape(numSamplesTotal, numX));
        Mat oTs = reshaped(output[0].getRef<Mat>(), Shape(numSamplesTotal, numO));
        Mat hTs = (produceH) ? reshaped(output[1].getRef<Mat>(), Shape(numSamplesTotal, numH)) : Mat();

        for (int ts = 0; ts < numTimestamps; ts++)
        {
            Range curRowRange = Range(ts * numSamples, (ts + 1) * numSamples);
            Mat xCurr = xTs.rowRange(curRowRange);

            dnn::gemm(hPrev, Whh, 1, hCurr, 0, GEMM_2_T); // W_{hh} * h_{prev}
            dnn::gemm(xCurr, Wxh, 1, hCurr, 1, GEMM_2_T); //+W_{xh} * x_{curr}
            dnn::gemm(dummyBiasOnes, bh, 1, hCurr, 1);    //+bh
            tanh(hCurr, hPrev);

            Mat oCurr = oTs.rowRange(curRowRange);
            dnn::gemm(hPrev, Who, 1, oCurr, 0, GEMM_2_T); // W_{ho} * h_{prev}
            dnn::gemm(dummyBiasOnes, bo, 1, oCurr, 1);    //+b_o
            tanh(oCurr, oCurr);

            if (produceH)
                hPrev.copyTo(hTs.rowRange(curRowRange));
        }
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
