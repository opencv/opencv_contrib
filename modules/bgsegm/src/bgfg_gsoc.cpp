/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                          License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000, Intel Corporation, all rights reserved.
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

/**
 * @file   bgfg_gsoc.cpp
 * @author Vladislav Samsonov <vvladxx@gmail.com>
 * @brief  Background Subtraction using Local SVD Binary Pattern. See the following paper:
 * http://www.cv-foundation.org/openaccess/content_cvpr_2016_workshops/w24/papers/Guo_Background_Subtraction_Using_CVPR_2016_paper.pdf
 * This file also contains implementation of the different yet better algorithm which is called GSOC, as it was implemented during GSOC and was not originated from any paper.
 *
*/

#include "precomp.hpp"
#include <opencv2/calib3d.hpp>
#include <iostream>
#include "opencv2/core/cvdef.h"

namespace cv
{
namespace bgsegm
{
namespace
{

const float LSBPtau = 0.05f;

#ifdef _MSC_VER
#include <intrin.h>
#pragma intrinsic(__popcnt)
#endif
inline int LSBPDist32(unsigned n) {
#if defined(__GNUC__) || defined(__clang__)
    return __builtin_popcount(n);
#elif defined(_MSC_VER)
    return __popcnt(n);
#else
    // Taken from http://graphics.stanford.edu/~seander/bithacks.html#CountBitsSetParallel
    n = n - ((n >> 1) & 0x55555555);
    n = (n & 0x33333333) + ((n >> 2) & 0x33333333);
    return ((n + ((n >> 4) & 0xF0F0F0F)) * 0x1010101) >> 24;
    // ---
#endif
}

inline float L2sqdist(const Point3f& a) {
    return a.dot(a);
}

inline float L1dist(const Point3f& a) {
    return std::abs(a.x) + std::abs(a.y) + std::abs(a.z);
}

inline float det3x3(float a11, float a12, float a13, float a22, float a23, float a33) {
    return a11 * (a22 * a33 - a23 * a23) + a12 * (2 * a13 * a23 - a33 * a12) - a13 * a13 * a22;
}

inline float localSVD(float a11, float a12, float a13, float a21, float a22, float a23, float a31, float a32, float a33) {
    float b11 = a11 * a11 + a12 * a12 + a13 * a13;
    float b12 = a11 * a21 + a12 * a22 + a13 * a23;
    float b13 = a11 * a31 + a12 * a32 + a13 * a33;
    float b22 = a21 * a21 + a22 * a22 + a23 * a23;
    float b23 = a21 * a31 + a22 * a32 + a23 * a33;
    float b33 = a31 * a31 + a32 * a32 + a33 * a33;
    const float q = (b11 + b22 + b33) / 3;

    b11 -= q;
    b22 -= q;
    b33 -= q;

    float p = std::sqrt((b11 * b11 + b22 * b22 + b33 * b33 + 2 * (b12 * b12 + b13 * b13 + b23 * b23)) / 6);

    if (p == 0)
        return 0;

    const float pi = 1 / p;
    const float r = det3x3(pi * b11, pi * b12, pi * b13, pi * b22, pi * b23, pi * b33) / 2;
    float phi;

    if (r <= -1)
        phi = float(CV_PI / 3);
    else if (r >= 1)
        phi = 0;
    else
        phi = std::acos(r) / 3;

    p *= 2;
    const float e1 = q + p * std::cos(phi);
    float e2, e3;

    if (e1 < 3 * q) {
        e3 = std::max(q + p * std::cos(phi + float(2 * CV_PI / 3)), 0.0f);
        e2 = std::max(3 * q - e1 - e3, 0.0f);
    }
    else {
        e2 = 0;
        e3 = 0;
    }

    return std::sqrt(e2 / e1) + std::sqrt(e3 / e1);
}

void removeNoise(Mat& fgMask, const Mat& compMask, const size_t threshold, const uchar filler) {
    const Size sz = fgMask.size();
    Mat labels;
    const int nComponents = connectedComponents(compMask, labels, 8, CV_32S);
    std::vector<size_t> compArea(nComponents, 0);

    for (int i = 0; i < sz.height; ++i)
        for (int j = 0; j < sz.width; ++j)
            ++compArea[labels.at<int>(i, j)];

    for (int i = 0; i < sz.height; ++i)
        for (int j = 0; j < sz.width; ++j)
            if (compArea[labels.at<int>(i, j)] < threshold)
                fgMask.at<uchar>(i, j) = filler;
}

void FindSparseCorrLK(const Mat& src, const Mat& dst, std::vector<Point2f>& srcPoints, std::vector<Point2f>& dstPoints) {
    Size size = src.size();
    const unsigned blockSize = 16;

    for (int x = blockSize / 2; x < size.width; x += blockSize)
        for (int y = blockSize / 2; y < size.height; y += blockSize) {
            srcPoints.push_back(Point2f(float(x), float(y)));
            dstPoints.push_back(Point2f(float(x), float(y)));
        }

    std::vector<uchar> predictedStatus;
    std::vector<float> predictedError;
    Mat srcGr, dstGr;
    src.copyTo(srcGr);
    dst.copyTo(dstGr);
    srcGr *= 255;
    dstGr *= 255;
    srcGr.convertTo(srcGr, CV_8UC3);
    dstGr.convertTo(dstGr, CV_8UC3);

    calcOpticalFlowPyrLK(srcGr, dstGr, srcPoints, dstPoints, predictedStatus, predictedError);

    size_t j = 0;
    for (size_t i = 0; i < srcPoints.size(); ++i) {
        if (predictedStatus[i]) {
            srcPoints[j] = srcPoints[i];
            dstPoints[j] = dstPoints[i];
            ++j;
        }
    }
    srcPoints.resize(j);
    dstPoints.resize(j);
}

class BackgroundSampleGSOC {
public:
    Point3f color;
    int desc;
    uint64 time;
    uint64 hits;

    BackgroundSampleGSOC(Point3f c = Point3f(), int d = 0, uint64 t = 0, uint64 h = 0) : color(c), desc(d), time(t), hits(h) {}
};

class BackgroundSampleLSBP {
public:
    Point3f color;
    int desc;
    float minDecisionDist;

    BackgroundSampleLSBP(Point3f c = Point3f(), int d = 0, float mdd = 1e9f) : color(c), desc(d), minDecisionDist(mdd) {}
};

template<typename BackgroundSampleType>
class BackgroundModel {
protected:
    std::vector<BackgroundSampleType> samples;
    const Size size;
    const int nSamples;
    const int stride;

public:
    BackgroundModel(Size sz, int S) : size(sz), nSamples(S), stride(sz.width * S) {
        samples.resize(sz.area() * S);
    }

    void swap(BackgroundModel& bm) {
        samples.swap(bm.samples);
    }

    void motionCompensation(const BackgroundModel& bm, const std::vector<Point2f>& points) {
        for (int i = 0; i < size.height; ++i)
                for (int j = 0; j < size.width; ++j) {
                    Point2i p = points[j * size.height + i];
                    if (p.x < 0)
                        p.x = 0;
                    if (p.y < 0)
                        p.y = 0;
                    if (p.x >= size.width)
                        p.x = size.width - 1;
                    if (p.y >= size.height)
                        p.y = size.height - 1;

                    memcpy(&samples[i * stride + j * nSamples], &bm.samples[p.y * stride + p.x * nSamples], sizeof(BackgroundSampleType) * nSamples);
                }
    }

    const BackgroundSampleType& operator()(int k) const {
        return samples[k];
    }

    BackgroundSampleType& operator()(int k) {
        return samples[k];
    }

    const BackgroundSampleType& operator()(int i, int j, int k) const {
        return samples[i * stride + j * nSamples + k];
    }

    BackgroundSampleType& operator()(int i, int j, int k) {
        return samples[i * stride + j * nSamples + k];
    }

    Size getSize() const {
        return size;
    }
};

class BackgroundModelGSOC : public BackgroundModel<BackgroundSampleGSOC> {
public:
    BackgroundModelGSOC(Size sz, int S) : BackgroundModel(sz, S) {};

    float findClosest(int i, int j, const Point3f& color, int& indOut) const {
        const int end = i * stride + (j + 1) * nSamples;
        int minInd = i * stride + j * nSamples;
        float minDist = L2sqdist(color - samples[minInd].color);
        for (int k = minInd + 1; k < end; ++k) {
            const float dist = L2sqdist(color - samples[k].color);
            if (dist < minDist) {
                minInd = k;
                minDist = dist;
            }
        }
        indOut = minInd;
        return minDist;
    }

    void replaceOldest(int i, int j, const BackgroundSampleGSOC& sample) {
        const int end = i * stride + (j + 1) * nSamples;
        int minInd = i * stride + j * nSamples;
        for (int k = minInd + 1; k < end; ++k) {
            if (samples[k].time < samples[minInd].time)
                minInd = k;
        }
        samples[minInd] = sample;
    }

    Point3f getMean(int i, int j, uint64 threshold) const {
        const int end = i * stride + (j + 1) * nSamples;
        Point3f acc(0, 0, 0);
        int cnt = 0;
        for (int k = i * stride + j * nSamples; k < end; ++k) {
            if (samples[k].hits > threshold) {
                acc += samples[k].color;
                ++cnt;
            }
        }
        if (cnt == 0) {
            cnt = nSamples;
            for (int k = i * stride + j * nSamples; k < end; ++k)
                acc += samples[k].color;
        }
        acc.x /= cnt;
        acc.y /= cnt;
        acc.z /= cnt;
        return acc;
    }
};

class BackgroundModelLSBP : public BackgroundModel<BackgroundSampleLSBP> {
public:
    BackgroundModelLSBP(Size sz, int S) : BackgroundModel(sz, S) {};

    int countMatches(int i, int j, const Point3f& color, int desc, float threshold, int descThreshold, float& minDist) const {
        const int end = i * stride + (j + 1) * nSamples;
        int count = 0;
        minDist = 1e9;
        for (int k = i * stride + j * nSamples; k < end; ++k) {
            const float dist = L1dist(color - samples[k].color);
            if (dist < threshold && LSBPDist32(static_cast<unsigned>(desc ^ samples[k].desc)) < descThreshold)
                ++count;
            if (dist < minDist)
                minDist = dist;
        }
        return count;
    }

    Point3f getMean(int i, int j) const {
        const int end = i * stride + (j + 1) * nSamples;
        Point3f acc(0, 0, 0);
        for (int k = i * stride + j * nSamples; k < end; ++k) {
            acc += samples[k].color;
        }
        acc.x /= nSamples;
        acc.y /= nSamples;
        acc.z /= nSamples;
        return acc;
    }

    float getDMean(int i, int j) const {
        const int end = i * stride + (j + 1) * nSamples;
        float d = 0;
        for (int k = i * stride + j * nSamples; k < end; ++k)
            d += samples[k].minDecisionDist;

        return d / nSamples;
    }
};

class ParallelLocalSVDValues : public ParallelLoopBody {
private:
    const Size sz;
    Mat& localSVDValues;
    const Mat& frameGray;

    ParallelLocalSVDValues &operator=(const ParallelLocalSVDValues&);

public:
    ParallelLocalSVDValues(const Size& _sz, Mat& _localSVDValues, const Mat& _frameGray) : sz(_sz), localSVDValues(_localSVDValues), frameGray(_frameGray) {};

    void operator()(const Range &range) const {
        for (int i = range.start; i < range.end; ++i)
            for (int j = 1; j < sz.width - 1; ++j) {
                localSVDValues.at<float>(i, j) = localSVD(
                    frameGray.at<float>(i - 1, j - 1), frameGray.at<float>(i - 1, j), frameGray.at<float>(i - 1, j + 1),
                    frameGray.at<float>(i, j - 1), frameGray.at<float>(i, j), frameGray.at<float>(i, j + 1),
                    frameGray.at<float>(i + 1, j - 1), frameGray.at<float>(i + 1, j), frameGray.at<float>(i + 1, j + 1));
            }
    }
};

class ParallelFromLocalSVDValues : public ParallelLoopBody {
private:
    const Size sz;
    Mat& desc;
    const Mat& localSVDValues;
    const Point2i* LSBPSamplePoints;

    ParallelFromLocalSVDValues &operator=(const ParallelFromLocalSVDValues&);

public:
    ParallelFromLocalSVDValues(const Size& _sz, Mat& _desc, const Mat& _localSVDValues, const Point2i* _LSBPSamplePoints) : sz(_sz), desc(_desc), localSVDValues(_localSVDValues), LSBPSamplePoints(_LSBPSamplePoints) {};

    void operator()(const Range &range) const {
        for (int index = range.start; index < range.end; ++index) {
            const int i = index / sz.width, j = index % sz.width;
            int& descVal = desc.at<int>(i, j);
            descVal = 0;
            const float centerVal = localSVDValues.at<float>(i, j);

            for (int n = 0; n < 32; ++n) {
                const int ri = i + LSBPSamplePoints[n].y;
                const int rj = j + LSBPSamplePoints[n].x;
                if (ri >= 0 && rj >= 0 && ri < sz.height && rj < sz.width && std::abs(localSVDValues.at<float>(ri, rj) - centerVal) > LSBPtau)
                    descVal |= int(1U) << n;
            }
        }
    }
};

} // namespace

void BackgroundSubtractorLSBPDesc::calcLocalSVDValues(OutputArray _localSVDValues, const Mat& frame) {
    Mat frameGray;
    const Size sz = frame.size();
    _localSVDValues.create(sz, CV_32F);
    Mat localSVDValues = _localSVDValues.getMat();
    localSVDValues = 0.0f;

    cvtColor(frame, frameGray, COLOR_BGR2GRAY);

    parallel_for_(Range(1, sz.height - 1), ParallelLocalSVDValues(sz, localSVDValues, frameGray));

    for (int i = 1; i < sz.height - 1; ++i) {
        localSVDValues.at<float>(i, 0) = localSVD(
            frameGray.at<float>(i - 1, 0), frameGray.at<float>(i - 1, 0), frameGray.at<float>(i - 1, 1),
            frameGray.at<float>(i, 0), frameGray.at<float>(i, 0), frameGray.at<float>(i, 1),
            frameGray.at<float>(i + 1, 0), frameGray.at<float>(i + 1, 0), frameGray.at<float>(i + 1, 1));

        localSVDValues.at<float>(i, sz.width - 1) = localSVD(
            frameGray.at<float>(i - 1, sz.width - 2), frameGray.at<float>(i - 1, sz.width - 1), frameGray.at<float>(i - 1, sz.width - 1),
            frameGray.at<float>(i, sz.width - 2), frameGray.at<float>(i, sz.width - 1), frameGray.at<float>(i, sz.width - 1),
            frameGray.at<float>(i + 1, sz.width - 2), frameGray.at<float>(i + 1, sz.width - 1), frameGray.at<float>(i + 1, sz.width - 1));
    }

    for (int j = 1; j < sz.width - 1; ++j) {
        localSVDValues.at<float>(0, j) = localSVD(
            frameGray.at<float>(0, j - 1), frameGray.at<float>(0, j), frameGray.at<float>(0, j + 1),
            frameGray.at<float>(0, j - 1), frameGray.at<float>(0, j), frameGray.at<float>(0, j + 1),
            frameGray.at<float>(1, j - 1), frameGray.at<float>(1, j), frameGray.at<float>(1, j + 1));
        localSVDValues.at<float>(sz.height - 1, j) = localSVD(
            frameGray.at<float>(sz.height - 2, j - 1), frameGray.at<float>(sz.height - 2, j), frameGray.at<float>(sz.height - 2, j + 1),
            frameGray.at<float>(sz.height - 1, j - 1), frameGray.at<float>(sz.height - 1, j), frameGray.at<float>(sz.height - 1, j + 1),
            frameGray.at<float>(sz.height - 1, j - 1), frameGray.at<float>(sz.height - 1, j), frameGray.at<float>(sz.height - 1, j + 1));
    }
}

void BackgroundSubtractorLSBPDesc::computeFromLocalSVDValues(OutputArray _desc, const Mat& localSVDValues, const Point2i* LSBPSamplePoints) {
    const Size sz = localSVDValues.size();
    _desc.create(sz, CV_32S);
    Mat desc = _desc.getMat();

    parallel_for_(Range(0, sz.area()), ParallelFromLocalSVDValues(sz, desc, localSVDValues, LSBPSamplePoints));
}

void BackgroundSubtractorLSBPDesc::compute(OutputArray desc, const Mat& frame, const Point2i* LSBPSamplePoints) {
    Mat localSVDValues;
    calcLocalSVDValues(localSVDValues, frame);
    computeFromLocalSVDValues(desc, localSVDValues, LSBPSamplePoints);
}

class BackgroundSubtractorGSOCImpl : public BackgroundSubtractorGSOC {
private:
    Ptr<BackgroundModelGSOC> backgroundModel;
    Ptr<BackgroundModelGSOC> backgroundModelPrev;
    uint64 currentTime;
    const int motionCompensation;
    const int nSamples;
    const float replaceRate;
    const float propagationRate;
    const uint64 hitsThreshold;
    const float alpha;
    const float beta;
    const float blinkingSupressionDecay;
    const float blinkingSupressionMultiplier;
    const float noiseRemovalThresholdFacBG;
    const float noiseRemovalThresholdFacFG;
    Mat distMovingAvg;
    Mat prevFgMask;
    Mat prevFrame;
    Mat blinkingSupression;
    RNG rng;

    void postprocessing(Mat& fgMask);

public:
    BackgroundSubtractorGSOCImpl(int mc,
                                 int nSamples,
                                 float replaceRate,
                                 float propagationRate,
                                 int hitsThreshold,
                                 float alpha,
                                 float beta,
                                 float blinkingSupressionDecay,
                                 float blinkingSupressionMultiplier,
                                 float noiseRemovalThresholdFacBG,
                                 float noiseRemovalThresholdFacFG);

    CV_WRAP virtual void apply(InputArray image, OutputArray fgmask, double learningRate = -1);

    CV_WRAP virtual void getBackgroundImage(OutputArray backgroundImage) const;

    friend class ParallelGSOC;
};

class BackgroundSubtractorLSBPImpl : public BackgroundSubtractorLSBP {
private:
    Ptr<BackgroundModelLSBP> backgroundModel;
    Ptr<BackgroundModelLSBP> backgroundModelPrev;
    const int motionCompensation;
    const int nSamples;
    const int LSBPRadius;
    const float Tlower;
    const float Tupper;
    const float Tinc;
    const float Tdec;
    const float Rscale;
    const float Rincdec;
    const float noiseRemovalThresholdFacBG;
    const float noiseRemovalThresholdFacFG;
    const int LSBPthreshold;
    const int minCount;
    Mat T;
    Mat R;
    Mat prevFrame;
    RNG rng;
    Point2i LSBPSamplePoints[32];

    void postprocessing(Mat& fgMask);

public:
    BackgroundSubtractorLSBPImpl(int mc,
                                 int nSamples,
                                 int LSBPRadius,
                                 float Tlower,
                                 float Tupper,
                                 float Tinc,
                                 float Tdec,
                                 float Rscale,
                                 float Rincdec,
                                 float noiseRemovalThresholdFacBG,
                                 float noiseRemovalThresholdFacFG,
                                 int LSBPthreshold,
                                 int minCount
                                );

    CV_WRAP virtual void apply(InputArray image, OutputArray fgmask, double learningRate = -1);

    CV_WRAP virtual void getBackgroundImage(OutputArray backgroundImage) const;

    friend class ParallelLSBP;
};

class ParallelGSOC : public ParallelLoopBody {
private:
    const Size sz;
    BackgroundSubtractorGSOCImpl* bgs;
    const Mat& frame;
    const double learningRate;
    Mat& fgMask;

    ParallelGSOC &operator=(const ParallelGSOC&);

public:
    ParallelGSOC(const Size& _sz, BackgroundSubtractorGSOCImpl* _bgs, const Mat& _frame, double _learningRate, Mat& _fgMask)
    : sz(_sz), bgs(_bgs), frame(_frame), learningRate(_learningRate), fgMask(_fgMask) {};

    void operator()(const Range &range) const {
        BackgroundModelGSOC* backgroundModel = bgs->backgroundModel.get();
        Mat& distMovingAvg = bgs->distMovingAvg;

        for (int index = range.start; index < range.end; ++index) {
            const int i = index / sz.width, j = index % sz.width;
            int k;
            const float minDist = backgroundModel->findClosest(i, j, frame.at<Point3f>(i, j), k);

            distMovingAvg.at<float>(i, j) *= 1 - float(learningRate);
            distMovingAvg.at<float>(i, j) += float(learningRate) * minDist;

            const float threshold = bgs->alpha * distMovingAvg.at<float>(i, j) + bgs->beta;
            BackgroundSampleGSOC& sample = (* backgroundModel)(k);

            if (minDist > threshold) {
                fgMask.at<uchar>(i, j) = 255;

                if (bgs->rng.uniform(0.0f, 1.0f) < bgs->replaceRate)
                    backgroundModel->replaceOldest(i, j, BackgroundSampleGSOC(frame.at<Point3f>(i, j), 0, bgs->currentTime));
            }
            else {
                sample.color *= 1 - learningRate;
                sample.color += learningRate * frame.at<Point3f>(i, j);
                sample.time = bgs->currentTime;
                ++sample.hits;

                // Propagation to neighbors
                if (sample.hits > bgs->hitsThreshold && bgs->rng.uniform(0.0f, 1.0f) < bgs->propagationRate) {
                    if (i + 1 < sz.height)
                        backgroundModel->replaceOldest(i + 1, j, sample);
                    if (j + 1 < sz.width)
                        backgroundModel->replaceOldest(i, j + 1, sample);
                    if (i > 0)
                        backgroundModel->replaceOldest(i - 1, j, sample);
                    if (j > 0)
                        backgroundModel->replaceOldest(i, j - 1, sample);
                }

                fgMask.at<uchar>(i, j) = 0;
            }
        }
    }
};

class ParallelLSBP : public ParallelLoopBody {
private:
    const Size sz;
    BackgroundSubtractorLSBPImpl* bgs;
    const Mat& frame;
    const double learningRate;
    const Mat& LSBPDesc;
    Mat& fgMask;

    ParallelLSBP &operator=(const ParallelLSBP&);

public:
    ParallelLSBP(const Size& _sz, BackgroundSubtractorLSBPImpl* _bgs, const Mat& _frame, double _learningRate, const Mat& _LSBPDesc, Mat& _fgMask)
    : sz(_sz), bgs(_bgs), frame(_frame), learningRate(_learningRate), LSBPDesc(_LSBPDesc), fgMask(_fgMask) {};

    void operator()(const Range &range) const {
        BackgroundModelLSBP* backgroundModel = bgs->backgroundModel.get();
        Mat& T = bgs->T;
        Mat& R = bgs->R;

        for (int index = range.start; index < range.end; ++index) {
            const int i = index / sz.width, j = index % sz.width;

            float minDist = 1e9f;
            const float DMean = backgroundModel->getDMean(i, j);

            if (R.at<float>(i, j) > DMean * bgs->Rscale)
                R.at<float>(i, j) *= 1 - bgs->Rincdec;
            else
                R.at<float>(i, j) *= 1 + bgs->Rincdec;

            if (backgroundModel->countMatches(i, j, frame.at<Point3f>(i, j), LSBPDesc.at<int>(i, j), R.at<float>(i, j), bgs->LSBPthreshold, minDist) < bgs->minCount) {
                fgMask.at<uchar>(i, j) = 255;

                T.at<float>(i, j) += bgs->Tinc / DMean;
            }
            else {
                fgMask.at<uchar>(i, j) = 0;

                T.at<float>(i, j) -= bgs->Tdec / DMean;

                if (bgs->rng.uniform(0.0f, 1.0f) < 1 / T.at<float>(i, j))
                    (* backgroundModel)(i, j, bgs->rng.uniform(0, bgs->nSamples)) = BackgroundSampleLSBP(frame.at<Point3f>(i, j), LSBPDesc.at<int>(i, j), minDist);

                if (bgs->rng.uniform(0.0f, 1.0f) < 1 / T.at<float>(i, j)) {
                    const int oi = i + bgs->rng.uniform(-1, 2);
                    const int oj = j + bgs->rng.uniform(-1, 2);

                    if (oi >= 0 && oi < sz.height && oj >= 0 && oj < sz.width)
                        (* backgroundModel)(oi, oj, bgs->rng.uniform(0, bgs->nSamples)) = BackgroundSampleLSBP(frame.at<Point3f>(oi, oj), LSBPDesc.at<int>(oi, oj), minDist);
                }
            }

            T.at<float>(i, j) = std::min(T.at<float>(i, j), bgs->Tupper);
            T.at<float>(i, j) = std::max(T.at<float>(i, j), bgs->Tlower);
        }
    }
};

BackgroundSubtractorGSOCImpl::BackgroundSubtractorGSOCImpl(int _mc,
                                                           int _nSamples,
                                                           float _replaceRate,
                                                           float _propagationRate,
                                                           int _hitsThreshold,
                                                           float _alpha,
                                                           float _beta,
                                                           float _blinkingSupressionDecay,
                                                           float _blinkingSupressionMultiplier,
                                                           float _noiseRemovalThresholdFacBG,
                                                           float _noiseRemovalThresholdFacFG)
: currentTime(0),
  motionCompensation(_mc),
  nSamples(_nSamples),
  replaceRate(_replaceRate),
  propagationRate(_propagationRate),
  hitsThreshold(_hitsThreshold),
  alpha(_alpha),
  beta(_beta),
  blinkingSupressionDecay(_blinkingSupressionDecay),
  blinkingSupressionMultiplier(_blinkingSupressionMultiplier),
  noiseRemovalThresholdFacBG(_noiseRemovalThresholdFacBG),
  noiseRemovalThresholdFacFG(_noiseRemovalThresholdFacFG) {
    CV_Assert(nSamples > 1 && nSamples < 1024);
    CV_Assert(replaceRate >= 0 && replaceRate <= 1);
    CV_Assert(propagationRate >= 0 && propagationRate <= 1);
    CV_Assert(blinkingSupressionDecay > 0 && blinkingSupressionDecay < 1);
    CV_Assert(noiseRemovalThresholdFacBG >= 0 && noiseRemovalThresholdFacBG < 0.5);
    CV_Assert(noiseRemovalThresholdFacFG >= 0 && noiseRemovalThresholdFacFG < 0.5);
    CV_Assert(_hitsThreshold >= 0);
}

void BackgroundSubtractorGSOCImpl::postprocessing(Mat& fgMask) {
    removeNoise(fgMask, fgMask, size_t(noiseRemovalThresholdFacBG * fgMask.size().area()), 0);
    Mat invFgMask = 255 - fgMask;
    removeNoise(fgMask, invFgMask, size_t(noiseRemovalThresholdFacFG * fgMask.size().area()), 255);

    GaussianBlur(fgMask, fgMask, Size(5, 5), 0);
    fgMask = fgMask > 127;
}

void BackgroundSubtractorGSOCImpl::apply(InputArray _image, OutputArray _fgmask, double learningRate) {
    const Size sz = _image.size();
    _fgmask.create(sz, CV_8U);
    Mat fgMask = _fgmask.getMat();

    Mat frame = _image.getMat();

    CV_Assert(frame.depth() == CV_8U || frame.depth() == CV_32F);
    CV_Assert(frame.channels() == 1 || frame.channels() == 3);

    if (frame.channels() != 3)
        cvtColor(frame, frame, COLOR_GRAY2BGR);

    if (frame.depth() != CV_32F) {
        frame.convertTo(frame, CV_32F);
        frame /= 255;
    }

    CV_Assert(frame.channels() == 3);

    if (backgroundModel.empty()) {
        backgroundModel = makePtr<BackgroundModelGSOC>(sz, nSamples);
        backgroundModelPrev = makePtr<BackgroundModelGSOC>(sz, nSamples);
        distMovingAvg = Mat(sz, CV_32F, Scalar::all(0.005f));
        prevFgMask = Mat(sz, CV_8U, Scalar::all(0));
        blinkingSupression = Mat(sz, CV_32F, Scalar::all(0.0f));

        for (int i = 0; i < sz.height; ++i)
            for (int j = 0; j < sz.width; ++j) {
                BackgroundSampleGSOC sample(frame.at<Point3f>(i, j), 0);
                for (int k = 0; k < nSamples; ++k) {
                    (* backgroundModel)(i, j, k) = sample;
                    (* backgroundModelPrev)(i, j, k) = sample;
                }
            }
    }

    CV_Assert(backgroundModel->getSize() == sz);

    if (motionCompensation) {
        std::vector<Point2f> srcPoints;
        std::vector<Point2f> dstPoints;

        if (prevFrame.empty())
            frame.copyTo(prevFrame);

        if (motionCompensation == LSBP_CAMERA_MOTION_COMPENSATION_LK)
            FindSparseCorrLK(frame, prevFrame, srcPoints, dstPoints);

        if (srcPoints.size()) {
            Mat H = findHomography(srcPoints, dstPoints, LMEDS);

            srcPoints.clear();
            for (int x = 0; x < sz.width; ++x)
                for (int y = 0; y < sz.height; ++y)
                    srcPoints.push_back(Point2f(float(x), float(y)));
            dstPoints.resize(srcPoints.size());
            perspectiveTransform(srcPoints, dstPoints, H);

            backgroundModel->swap(* backgroundModelPrev);
            backgroundModel->motionCompensation(* backgroundModelPrev, dstPoints);
        }

        frame.copyTo(prevFrame);
    }

    if (learningRate > 1 || learningRate < 0)
        learningRate = 0.1;

    parallel_for_(Range(0, sz.area()), ParallelGSOC(sz, this, frame, learningRate, fgMask));

    ++currentTime;

    cv::add(blinkingSupression, (fgMask != prevFgMask) / 255, blinkingSupression, cv::noArray(), CV_32F);
    blinkingSupression *= blinkingSupressionDecay;
    fgMask.copyTo(prevFgMask);
    Mat prob = blinkingSupression * (blinkingSupressionMultiplier * (1 - blinkingSupressionDecay) / blinkingSupressionDecay);

    for (int i = 0; i < sz.height; ++i)
        for (int j = 0; j < sz.width; ++j)
            if (rng.uniform(0.0f, 1.0f) < prob.at<float>(i, j))
                backgroundModel->replaceOldest(i, j, BackgroundSampleGSOC(frame.at<Point3f>(i, j), 0, currentTime));

    this->postprocessing(fgMask);
}

void BackgroundSubtractorGSOCImpl::getBackgroundImage(OutputArray _backgroundImage) const {
    CV_Assert(!backgroundModel.empty());
    const Size sz = backgroundModel->getSize();
    _backgroundImage.create(sz, CV_8UC3);
    Mat backgroundImage = _backgroundImage.getMat();
    for (int i = 0; i < sz.height; ++i)
        for (int j = 0; j < sz.width; ++j)
            backgroundImage.at< Point3_<uchar> >(i, j) = backgroundModel->getMean(i, j, hitsThreshold) * 255;
}

BackgroundSubtractorLSBPImpl::BackgroundSubtractorLSBPImpl(int _mc,
                                                           int _nSamples,
                                                           int _LSBPRadius,
                                                           float _Tlower,
                                                           float _Tupper,
                                                           float _Tinc,
                                                           float _Tdec,
                                                           float _Rscale,
                                                           float _Rincdec,
                                                           float _noiseRemovalThresholdFacBG,
                                                           float _noiseRemovalThresholdFacFG,
                                                           int _LSBPthreshold,
                                                           int _minCount
                                                          )
: motionCompensation(_mc),
  nSamples(_nSamples),
  LSBPRadius(_LSBPRadius),
  Tlower(_Tlower),
  Tupper(_Tupper),
  Tinc(_Tinc),
  Tdec(_Tdec),
  Rscale(_Rscale),
  Rincdec(_Rincdec),
  noiseRemovalThresholdFacBG(_noiseRemovalThresholdFacBG),
  noiseRemovalThresholdFacFG(_noiseRemovalThresholdFacFG),
  LSBPthreshold(_LSBPthreshold),
  minCount(_minCount) {
    CV_Assert(nSamples > 1 && nSamples < 1024);
    CV_Assert(LSBPRadius > 0);
    CV_Assert(Tlower < Tupper && Tlower > 0);
    CV_Assert(noiseRemovalThresholdFacBG >= 0 && noiseRemovalThresholdFacBG < 0.5);
    CV_Assert(noiseRemovalThresholdFacFG >= 0 && noiseRemovalThresholdFacFG < 0.5);

    for (int i = 0; i < 32; ++i) {
        const double phi = i * CV_2PI / 32.0;
        LSBPSamplePoints[i] = Point2i(int(LSBPRadius * std::cos(phi)), int(LSBPRadius * std::sin(phi)));
    }
}

void BackgroundSubtractorLSBPImpl::postprocessing(Mat& fgMask) {
    removeNoise(fgMask, fgMask, size_t(noiseRemovalThresholdFacBG * fgMask.size().area()), 0);
    Mat invFgMask = 255 - fgMask;
    removeNoise(fgMask, invFgMask, size_t(noiseRemovalThresholdFacFG * fgMask.size().area()), 255);

    GaussianBlur(fgMask, fgMask, Size(5, 5), 0);
    fgMask = fgMask > 127;
}

void BackgroundSubtractorLSBPImpl::apply(InputArray _image, OutputArray _fgmask, double learningRate) {
    const Size sz = _image.size();
    _fgmask.create(sz, CV_8U);
    Mat fgMask = _fgmask.getMat();

    Mat frame = _image.getMat();

    CV_Assert(frame.depth() == CV_8U || frame.depth() == CV_32F);
    CV_Assert(frame.channels() == 1 || frame.channels() == 3);

    if (frame.channels() != 3)
        cvtColor(frame, frame, COLOR_GRAY2BGR);

    if (frame.depth() != CV_32F) {
        frame.convertTo(frame, CV_32F, 1.0/255);
    }

    CV_Assert(frame.channels() == 3);
    Mat LSBPDesc(sz, CV_32S, Scalar::all(0));

    BackgroundSubtractorLSBPDesc::compute(LSBPDesc, frame, LSBPSamplePoints);

    if (backgroundModel.empty()) {
        backgroundModel = makePtr<BackgroundModelLSBP>(sz, nSamples);
        backgroundModelPrev = makePtr<BackgroundModelLSBP>(sz, nSamples);
        T = Mat(sz, CV_32F);
        T = (Tlower + Tupper) * 0.5f;
        R = Mat(sz, CV_32F);
        R = 0.1f;

        for (int i = 0; i < sz.height; ++i)
            for (int j = 0; j < sz.width; ++j) {
                BackgroundSampleLSBP sample(frame.at<Point3f>(i, j), LSBPDesc.at<int>(i, j));
                for (int k = 0; k < nSamples; ++k) {
                    (* backgroundModel)(i, j, k) = sample;
                    (* backgroundModelPrev)(i, j, k) = sample;
                }
            }
    }

    CV_Assert(backgroundModel->getSize() == sz);

    if (motionCompensation) {
        std::vector<Point2f> srcPoints;
        std::vector<Point2f> dstPoints;

        if (prevFrame.empty())
            frame.copyTo(prevFrame);

        if (motionCompensation == LSBP_CAMERA_MOTION_COMPENSATION_LK)
            FindSparseCorrLK(frame, prevFrame, srcPoints, dstPoints);

        if (srcPoints.size()) {
            Mat H = findHomography(srcPoints, dstPoints, LMEDS);

            srcPoints.clear();
            for (int x = 0; x < sz.width; ++x)
                for (int y = 0; y < sz.height; ++y)
                    srcPoints.push_back(Point2f(float(x), float(y)));
            dstPoints.resize(srcPoints.size());
            perspectiveTransform(srcPoints, dstPoints, H);

            backgroundModel->swap(* backgroundModelPrev);
            backgroundModel->motionCompensation(* backgroundModelPrev, dstPoints);
        }

        frame.copyTo(prevFrame);
    }

    if (learningRate > 1 || learningRate < 0)
        learningRate = 0.1;

    parallel_for_(Range(0, sz.area()), ParallelLSBP(sz, this, frame, learningRate, LSBPDesc, fgMask));

    this->postprocessing(fgMask);
}

void BackgroundSubtractorLSBPImpl::getBackgroundImage(OutputArray _backgroundImage) const {
    CV_Assert(!backgroundModel.empty());
    const Size sz = backgroundModel->getSize();
    _backgroundImage.create(sz, CV_8UC3);
    Mat backgroundImage = _backgroundImage.getMat();
    for (int i = 0; i < sz.height; ++i)
        for (int j = 0; j < sz.width; ++j)
            backgroundImage.at< Point3_<uchar> >(i, j) = backgroundModel->getMean(i, j) * 255;
}

Ptr<BackgroundSubtractorGSOC> createBackgroundSubtractorGSOC(int mc,
                                                             int nSamples,
                                                             float replaceRate,
                                                             float propagationRate,
                                                             int hitsThreshold,
                                                             float alpha,
                                                             float beta,
                                                             float blinkingSupressionDecay,
                                                             float blinkingSupressionMultiplier,
                                                             float noiseRemovalThresholdFacBG,
                                                             float noiseRemovalThresholdFacFG) {
    return makePtr<BackgroundSubtractorGSOCImpl>(mc,
                                                 nSamples,
                                                 replaceRate,
                                                 propagationRate,
                                                 hitsThreshold,
                                                 alpha,
                                                 beta,
                                                 blinkingSupressionDecay,
                                                 blinkingSupressionMultiplier,
                                                 noiseRemovalThresholdFacBG,
                                                 noiseRemovalThresholdFacFG);
}

Ptr<BackgroundSubtractorLSBP> createBackgroundSubtractorLSBP(int mc,
                                                             int nSamples,
                                                             int LSBPRadius,
                                                             float Tlower,
                                                             float Tupper,
                                                             float Tinc,
                                                             float Tdec,
                                                             float Rscale,
                                                             float Rincdec,
                                                             float noiseRemovalThresholdFacBG,
                                                             float noiseRemovalThresholdFacFG,
                                                             int LSBPthreshold,
                                                             int minCount
                                                            ) {
    return Ptr<BackgroundSubtractorLSBPImpl>(
        new BackgroundSubtractorLSBPImpl(
                                            mc,
                                            nSamples,
                                            LSBPRadius,
                                            Tlower,
                                            Tupper,
                                            Tinc,
                                            Tdec,
                                            Rscale,
                                            Rincdec,
                                            noiseRemovalThresholdFacBG,
                                            noiseRemovalThresholdFacFG,
                                            LSBPthreshold,
                                            minCount
                                        ));
}

} // namespace bgsegm
} // namespace cv
