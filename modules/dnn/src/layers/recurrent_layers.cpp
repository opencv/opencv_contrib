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
#include "op_blas.hpp"
#include <iostream>
#include <iterator>
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
    bool allocated;

    std::vector<int> outTailShape;                 //shape of single output sample
    std::vector<int> outTsMatShape, outTsShape;    //shape of N output samples
    std::vector<int> outResShape;                  //shape of T timestamps and N output samples

    bool useTimestampDim;
    bool produceCellOutput;

public:

    LSTMLayerImpl(const LayerParams& params)
    {
        setParamsFrom(params);
        type = "LSTM";
        useTimestampDim = true;
        produceCellOutput = false;
        allocated = false;
        outTailShape.clear();
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

    void setC(const Mat &C)
    {
        CV_Assert(C.type() == CV_32F);
        if (!cInternal.empty())
        {
            CV_Assert(C.total() == cInternal.total() && cInternal.isContinuous());
            Mat cInternal_(C.dims, &C.size.p[0], C.type(), cInternal.ptr());
            C.copyTo(cInternal_);
        }
        else
            C.copyTo(cInternal);
    }

    void setH(const Mat &H)
    {
        CV_Assert(H.type() == CV_32F);
        if (!hInternal.empty())
        {
            CV_Assert(H.total() == hInternal.total() && hInternal.isContinuous());
            Mat hInternal_(H.dims, &H.size.p[0], H.type(), hInternal.ptr());
            H.copyTo(hInternal_);
        }
        else
            H.copyTo(hInternal);
    }

    Mat getC() const
    {
        CV_Assert(shapeTotal(outTsShape) == cInternal.total());
        return Mat((int)outTsShape.size(), &outTsShape[0], cInternal.type(), (char*)cInternal.ptr());
    }

    Mat getH() const
    {
        CV_Assert(shapeTotal(outTsShape) == hInternal.total());
        return Mat((int)outTsShape.size(), &outTsShape[0], hInternal.type(), (char*)hInternal.ptr());
    }

    void setOutShape(const std::vector<int> &outTailShape_)
    {
        CV_Assert(!allocated || shapeTotal(outTailShape) == shapeTotal(outTailShape_));
        outTailShape = outTailShape_;
    }

    void setWeights(const Mat &Wh, const Mat &Wx, const Mat &bias)
    {
        CV_Assert(Wh.dims == 2 && Wx.dims == 2);
        CV_Assert(Wh.rows == Wx.rows);
        CV_Assert(Wh.rows == 4*Wh.cols);
        CV_Assert(Wh.rows == (int)bias.total());
        CV_Assert(Wh.type() == Wx.type() && Wx.type() == bias.type());

        blobs.resize(3);
        blobs[0] = Mat(Wh.clone());
        blobs[1] = Mat(Wx.clone());
        blobs[2] = Mat(bias.clone()).reshape(1, 1);
    }

    void allocate(const std::vector<Mat*> &input, std::vector<Mat> &output)
    {
        CV_Assert(blobs.size() == 3);
        CV_Assert(input.size() == 1);
        const Mat& inp0 = *input[0];

        Mat &Wh = blobs[0], &Wx = blobs[1];
        numOut = Wh.size[1];
        numInp = Wx.size[1];

        if (!outTailShape.empty())
            CV_Assert(shapeTotal(outTailShape) == numOut);
        else
            outTailShape.assign(1, numOut);

        outResShape.clear();
        if (useTimestampDim)
        {
            CV_Assert(inp0.dims >= 2 && (int)inp0.total(2) == numInp);
            numTimeStamps = inp0.size[0];
            numSamples = inp0.size[1];
            outResShape.push_back(numTimeStamps);
        }
        else
        {
            CV_Assert(inp0.dims >= 2 && (int)inp0.total(1) == numInp);
            numTimeStamps = 1;
            numSamples = inp0.size[0];
        }

        outResShape.push_back(numSamples);
        outResShape.insert(outResShape.end(), outTailShape.begin(), outTailShape.end());

        outTsMatShape.clear();
        outTsMatShape.push_back(numSamples);
        outTsMatShape.push_back(numOut);

        outTsShape.clear();
        outTsShape.push_back(numSamples);
        outTsShape.insert(outTsShape.end(), outTailShape.begin(), outTailShape.end());

        const int dtype = CV_32F;
        CV_Assert(inp0.type() == dtype && Wh.type() == dtype);

        size_t i, noutputs = produceCellOutput ? 2 : 1;
        output.resize(noutputs);

        for( i = 0; i < noutputs; i++ )
            output[i].create(outResShape, dtype);

        if (hInternal.empty())
        {
            hInternal.create(outTsMatShape, dtype);
            hInternal.setTo(0.);
        }
        else
        {
            CV_Assert(hInternal.total() == (size_t)numSamples*numOut);
            hInternal = hInternal.reshape(1, outTsMatShape);
        }

        if (cInternal.empty())
        {
            cInternal.create(outTsMatShape, dtype);
            cInternal.setTo(0.);
        }
        else
        {
            CV_Assert(cInternal.total() == (size_t)numSamples*numOut);
            cInternal = cInternal.reshape(1, outTsMatShape);
        }

        gates.create(numSamples, 4*numOut, dtype);

        dummyOnes.create(numSamples, 1, dtype);
        dummyOnes.setTo(1.);

        allocated = true;
    }

    void forward(std::vector<Mat*> &input, std::vector<Mat> &output)
    {
        const Mat &Wh = blobs[0];
        const Mat &Wx = blobs[1];
        const Mat &bias = blobs[2];

        int numSamplesTotal = numTimeStamps*numSamples;
        Mat xTs = input[0]->reshape(1, numSamplesTotal);

        Mat hOutTs = output[0].reshape(1, numSamplesTotal);
        Mat cOutTs = produceCellOutput ? output[1].reshape(1, numSamplesTotal) : Mat();

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
            multiply(gateF, cInternal, gateF);  // f_t (*) c_{t-1}
            multiply(gateI, gateG, gateI);      // i_t (*) g_t
            add(gateF, gateI, cInternal);       // c_t = f_t (*) c_{t-1} + i_t (*) g_t

            //compute h_t
            tanh(cInternal, hInternal);
            multiply(gateO, hInternal, hInternal);

            //save results in output blobs
            hInternal.copyTo(hOutTs.rowRange(curRowRange));
            if (produceCellOutput)
                cInternal.copyTo(cOutTs.rowRange(curRowRange));
        }
    }
};

Ptr<LSTMLayer> LSTMLayer::create(const LayerParams& params)
{
    return Ptr<LSTMLayer>(new LSTMLayerImpl(params));
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

    RNNLayerImpl(const LayerParams& params)
    {
        setParamsFrom(params);
        type = "RNN";
        produceH = false;
    }

    void setProduceHiddenOutput(bool produce = false)
    {
        produceH = produce;
    }

    void setWeights(const Mat &W_xh, const Mat &b_h, const Mat &W_hh, const Mat &W_ho, const Mat &b_o)
    {
        CV_Assert(W_hh.dims == 2 && W_xh.dims == 2);
        CV_Assert(W_hh.size[0] == W_xh.size[0] && W_hh.size[0] == W_hh.size[1] && (int)b_h.total() == W_xh.size[0]);
        CV_Assert(W_ho.size[0] == (int)b_o.total());
        CV_Assert(W_ho.size[1] == W_hh.size[1]);

        blobs.resize(5);
        blobs[0] = Mat(W_xh.clone());
        blobs[1] = Mat(b_h.clone());
        blobs[2] = Mat(W_hh.clone());
        blobs[3] = Mat(W_ho.clone());
        blobs[4] = Mat(b_o.clone());
    }

    void allocate(const std::vector<Mat*> &input, std::vector<Mat> &output)
    {
        CV_Assert(input.size() >= 1 && input.size() <= 2);

        Wxh = blobs[0];
        bh  = blobs[1];
        Whh = blobs[2];
        Who = blobs[3];
        bo  = blobs[4];

        numH = Wxh.rows;
        numX = Wxh.cols;
        numO = Who.rows;

        const Mat& inp0 = *input[0];

        CV_Assert(inp0.dims >= 2);
        CV_Assert(inp0.total(2) == numX);
        dtype = CV_32F;
        CV_Assert(inp0.type() == dtype);
        numTimestamps = inp0.size[0];
        numSamples = inp0.size[1];
        numSamplesTotal = numTimestamps * numSamples;

        hCurr.create(numSamples, numH, dtype);
        hPrev.create(numSamples, numH, dtype);
        hPrev.setTo(0.);

        dummyBiasOnes.create(numSamples, 1, dtype);
        dummyBiasOnes.setTo(1.);
        bh = bh.reshape(1, 1); //is 1 x numH Mat
        bo = bo.reshape(1, 1); //is 1 x numO Mat

        reshapeOutput(output);
    }

    void reshapeOutput(std::vector<Mat> &output)
    {
        output.resize(produceH ? 2 : 1);
        int sz0[] = { numTimestamps, numSamples, numO };
        output[0].create(3, sz0, dtype);
        if (produceH)
        {
            int sz1[] = { numTimestamps, numSamples, numH };
            output[1].create(3, sz1, dtype);
        }
    }

    void forward(std::vector<Mat*> &input, std::vector<Mat> &output)
    {
        Mat xTs = input[0]->reshape(1, numSamplesTotal);
        Mat oTs = output[0].reshape(1, numSamplesTotal);
        Mat hTs = produceH ? output[1].reshape(1, numSamplesTotal) : Mat();

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

CV_EXPORTS_W Ptr<RNNLayer> RNNLayer::create(const LayerParams& params)
{
    return Ptr<RNNLayer>(new RNNLayerImpl(params));
}

}
}
