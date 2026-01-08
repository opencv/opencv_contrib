/*
 *  By downloading, copying, installing or using the software you agree to this license.
 *  If you do not agree to this license, do not download, install,
 *  copy or use the software.
 *
 *
 *  License Agreement
 *  For Open Source Computer Vision Library
 *  (3 - clause BSD License)
 *
 *  Redistribution and use in source and binary forms, with or without modification,
 *  are permitted provided that the following conditions are met :
 *
 *  * Redistributions of source code must retain the above copyright notice,
 *  this list of conditions and the following disclaimer.
 *
 *  * Redistributions in binary form must reproduce the above copyright notice,
 *  this list of conditions and the following disclaimer in the documentation
 *  and / or other materials provided with the distribution.
 *
 *  * Neither the names of the copyright holders nor the names of the contributors
 *  may be used to endorse or promote products derived from this software
 *  without specific prior written permission.
 *
 *  This software is provided by the copyright holders and contributors "as is" and
 *  any express or implied warranties, including, but not limited to, the implied
 *  warranties of merchantability and fitness for a particular purpose are disclaimed.
 *  In no event shall copyright holders or contributors be liable for any direct,
 *  indirect, incidental, special, exemplary, or consequential damages
 *  (including, but not limited to, procurement of substitute goods or services;
 *  loss of use, data, or profits; or business interruption) however caused
 *  and on any theory of liability, whether in contract, strict liability,
 *  or tort(including negligence or otherwise) arising in any way out of
 *  the use of this software, even if advised of the possibility of such damage.
 */

#include "precomp.hpp"
#include <climits>
#include <iostream>
using namespace std;

#ifdef _MSC_VER
#   pragma warning(disable: 4512)
#endif

namespace cv
{
namespace ximgproc
{

typedef Vec<float, 1> Vec1f;
typedef Vec<uchar, 1> Vec1b;

#ifndef SQR
#define SQR(a) ((a)*(a))
#endif

void jointBilateralFilter_32f(Mat& joint, Mat& src, Mat& dst, int radius, double sigmaColor, double sigmaSpace, int borderType);

void jointBilateralFilter_8u(Mat& joint, Mat& src, Mat& dst, int radius, double sigmaColor, double sigmaSpace, int borderType);

template<typename JointVec, typename SrcVec>
class JointBilateralFilter_32f : public ParallelLoopBody
{
    Mat &joint, &src;
    Mat &dst;
    int radius, maxk;
    float scaleIndex;
    int *spaceOfs;
    float *spaceWeights, *expLUT;

public:

    JointBilateralFilter_32f(Mat& joint_, Mat& src_, Mat& dst_, int radius_,
        int maxk_, float scaleIndex_, int *spaceOfs_, float *spaceWeights_, float *expLUT_)
        :
        joint(joint_), src(src_), dst(dst_), radius(radius_), maxk(maxk_),
        scaleIndex(scaleIndex_), spaceOfs(spaceOfs_), spaceWeights(spaceWeights_), expLUT(expLUT_)
    {
        CV_DbgAssert(joint.type() == traits::Type<JointVec>::value && src.type() == dst.type() && src.type() == traits::Type<SrcVec>::value);
        CV_DbgAssert(joint.rows == src.rows && src.rows == dst.rows + 2*radius);
        CV_DbgAssert(joint.cols == src.cols && src.cols == dst.cols + 2*radius);
    }

    void operator () (const Range& range) const CV_OVERRIDE
    {
        for (int i = radius + range.start; i < radius + range.end; i++)
        {
            for (int j = radius; j < src.cols - radius; j++)
            {
                JointVec *jointCenterPixPtr = joint.ptr<JointVec>(i) + j;
                SrcVec *srcCenterPixPtr = src.ptr<SrcVec>(i) + j;

                JointVec jointPix0 = *jointCenterPixPtr;
                SrcVec srcSum = SrcVec::all(0.0f);
                float wSum = 0.0f;

                for (int k = 0; k < maxk; k++)
                {
                    float *jointPix = reinterpret_cast<float*>(jointCenterPixPtr + spaceOfs[k]);
                    float alpha = 0.0f;

                    for (int cn = 0; cn < JointVec::channels; cn++)
                        alpha += std::abs(jointPix0[cn] - jointPix[cn]);
                    alpha *= scaleIndex;
                    int idx = (int)(alpha);
                    alpha -= idx;
                    float weight = spaceWeights[k] * (expLUT[idx] + alpha*(expLUT[idx + 1] - expLUT[idx]));

                    float *srcPix = reinterpret_cast<float*>(srcCenterPixPtr + spaceOfs[k]);
                    for (int cn = 0; cn < SrcVec::channels; cn++)
                        srcSum[cn] += weight*srcPix[cn];
                    wSum += weight;
                }

                dst.at<SrcVec>(i - radius, j - radius) = srcSum / wSum;
            }
        }
    }
};

void jointBilateralFilter_32f(Mat& joint, Mat& src, Mat& dst, int radius, double sigmaColor, double sigmaSpace, int borderType)
{
    CV_DbgAssert(joint.depth() == CV_32F && src.depth() == CV_32F);

    int d = 2*radius + 1;
    int jCn = joint.channels();
    const int kExpNumBinsPerChannel = 1 << 12;
    double minValJoint, maxValJoint;

    minMaxLoc(joint, &minValJoint, &maxValJoint);
    if (abs(maxValJoint - minValJoint) < FLT_EPSILON)
    {
        //TODO: make circle pattern instead of square
        GaussianBlur(src, dst, Size(d, d), sigmaSpace, 0, borderType);
        return;
    }
    float colorRange = (float)(maxValJoint - minValJoint) * jCn;
    colorRange = std::max(0.01f, colorRange);

    int kExpNumBins = kExpNumBinsPerChannel * jCn;
    vector<float> expLUTv(kExpNumBins + 2);
    float *expLUT = &expLUTv[0];
    float scaleIndex = kExpNumBins/colorRange;

    double gaussColorCoeff = -0.5 / (sigmaColor*sigmaColor);
    double gaussSpaceCoeff = -0.5 / (sigmaSpace*sigmaSpace);

    for (int i = 0; i < kExpNumBins + 2; i++)
    {
        double val = i / scaleIndex;
        expLUT[i] = (float) std::exp(val * val * gaussColorCoeff);
    }

    Mat jointTemp, srcTemp;
    copyMakeBorder(joint, jointTemp, radius, radius, radius, radius, borderType);
    copyMakeBorder(src, srcTemp, radius, radius, radius, radius, borderType);
    size_t srcElemStep = srcTemp.step / srcTemp.elemSize();
    size_t jElemStep = jointTemp.step / jointTemp.elemSize();
    CV_Assert(srcElemStep == jElemStep);

    vector<float> spaceWeightsv(d*d);
    vector<int> spaceOfsJointv(d*d);
    float *spaceWeights = &spaceWeightsv[0];
    int *spaceOfsJoint = &spaceOfsJointv[0];

    int maxk = 0;
    for (int i = -radius; i <= radius; i++)
    {
        for (int j = -radius; j <= radius; j++)
        {
            double r2 = i*i + j*j;
            if (r2 > SQR(radius))
                continue;

            spaceWeights[maxk] = (float) std::exp(r2 * gaussSpaceCoeff);
            spaceOfsJoint[maxk] = (int) (i*jElemStep + j);
            maxk++;
        }
    }

    Range range(0, joint.rows);
    if (joint.type() == CV_32FC1)
    {
        if (src.type() == CV_32FC1)
        {
            parallel_for_(range, JointBilateralFilter_32f<Vec1f, Vec1f>(jointTemp, srcTemp, dst, radius, maxk, scaleIndex, spaceOfsJoint, spaceWeights, expLUT));
        }
        if (src.type() == CV_32FC3)
        {
            parallel_for_(range, JointBilateralFilter_32f<Vec1f, Vec3f>(jointTemp, srcTemp, dst, radius, maxk, scaleIndex, spaceOfsJoint, spaceWeights, expLUT));
        }
    }

    if (joint.type() == CV_32FC3)
    {
        if (src.type() == CV_32FC1)
        {
            parallel_for_(range, JointBilateralFilter_32f<Vec3f, Vec1f>(jointTemp, srcTemp, dst, radius, maxk, scaleIndex, spaceOfsJoint, spaceWeights, expLUT));
        }
        if (src.type() == CV_32FC3)
        {
            parallel_for_(range, JointBilateralFilter_32f<Vec3f, Vec3f>(jointTemp, srcTemp, dst, radius, maxk, scaleIndex, spaceOfsJoint, spaceWeights, expLUT));
        }
    }
}

template<typename JointVec, typename SrcVec>
class JointBilateralFilter_8u : public ParallelLoopBody
{
    Mat &joint, &src;
    Mat &dst;
    int radius, maxk;
    float scaleIndex;
    int *spaceOfs;
    float *spaceWeights, *expLUT;

public:

    JointBilateralFilter_8u(Mat& joint_, Mat& src_, Mat& dst_, int radius_,
        int maxk_, int *spaceOfs_, float *spaceWeights_, float *expLUT_)
        :
        joint(joint_), src(src_), dst(dst_), radius(radius_), maxk(maxk_),
        spaceOfs(spaceOfs_), spaceWeights(spaceWeights_), expLUT(expLUT_)
    {
        CV_DbgAssert(joint.type() == traits::Type<JointVec>::value && src.type() == dst.type() && src.type() == traits::Type<SrcVec>::value);
        CV_DbgAssert(joint.rows == src.rows && src.rows == dst.rows + 2 * radius);
        CV_DbgAssert(joint.cols == src.cols && src.cols == dst.cols + 2 * radius);
    }

    void operator () (const Range& range) const CV_OVERRIDE
    {
        typedef Vec<int, JointVec::channels> JointVeci;
        typedef Vec<float, SrcVec::channels> SrcVecf;

        for (int i = radius + range.start; i < radius + range.end; i++)
        {
            for (int j = radius; j < src.cols - radius; j++)
            {
                JointVec *jointCenterPixPtr = joint.ptr<JointVec>(i) + j;
                SrcVec *srcCenterPixPtr = src.ptr<SrcVec>(i) + j;

                JointVeci jointPix0 = JointVeci(*jointCenterPixPtr);
                SrcVecf srcSum = SrcVecf::all(0.0f);
                float wSum = 0.0f;

                for (int k = 0; k < maxk; k++)
                {
                    uchar *jointPix = reinterpret_cast<uchar*>(jointCenterPixPtr + spaceOfs[k]);
                    int alpha = 0;
                    for (int cn = 0; cn < JointVec::channels; cn++)
                        alpha += std::abs(jointPix0[cn] - (int)jointPix[cn]);

                    float weight = spaceWeights[k] * expLUT[alpha];

                    uchar *srcPix = reinterpret_cast<uchar*>(srcCenterPixPtr + spaceOfs[k]);
                    for (int cn = 0; cn < SrcVec::channels; cn++)
                        srcSum[cn] += weight*srcPix[cn];
                    wSum += weight;
                }

                dst.at<SrcVec>(i - radius, j - radius) = SrcVec(srcSum / wSum);
            }
        }
    }
};

void jointBilateralFilter_8u(Mat& joint, Mat& src, Mat& dst, int radius, double sigmaColor, double sigmaSpace, int borderType)
{
    CV_DbgAssert(joint.depth() == CV_8U && src.depth() == CV_8U);

    int d = 2 * radius + 1;
    int jCn = joint.channels();

    double gaussColorCoeff = -0.5 / (sigmaColor*sigmaColor);
    double gaussSpaceCoeff = -0.5 / (sigmaSpace*sigmaSpace);

    vector<float> expLUTv(jCn*256);
    float *expLUT = &expLUTv[0];

    for (int i = 0; i < (int)expLUTv.size(); i++)
    {
        expLUT[i] = (float)std::exp(i * i * gaussColorCoeff);
    }

    Mat jointTemp, srcTemp;
    copyMakeBorder(joint, jointTemp, radius, radius, radius, radius, borderType);
    copyMakeBorder(src, srcTemp, radius, radius, radius, radius, borderType);
    size_t srcElemStep = srcTemp.step / srcTemp.elemSize();
    size_t jElemStep = jointTemp.step / jointTemp.elemSize();
    CV_Assert(srcElemStep == jElemStep);

    vector<float> spaceWeightsv(d*d);
    vector<int> spaceOfsJointv(d*d);
    float *spaceWeights = &spaceWeightsv[0];
    int *spaceOfsJoint = &spaceOfsJointv[0];

    int maxk = 0;
    for (int i = -radius; i <= radius; i++)
    {
        for (int j = -radius; j <= radius; j++)
        {
            double r2 = i*i + j*j;
            if (r2 > SQR(radius))
                continue;

            spaceWeights[maxk] = (float)std::exp(r2 * gaussSpaceCoeff);
            spaceOfsJoint[maxk] = (int)(i*jElemStep + j);
            maxk++;
        }
    }

    Range range(0, src.rows);
    if (joint.type() == CV_8UC1)
    {
        if (src.type() == CV_8UC1)
        {
            parallel_for_(range, JointBilateralFilter_8u<Vec1b, Vec1b>(jointTemp, srcTemp, dst, radius, maxk, spaceOfsJoint, spaceWeights, expLUT));
        }
        if (src.type() == CV_8UC3)
        {
            parallel_for_(range, JointBilateralFilter_8u<Vec1b, Vec3b>(jointTemp, srcTemp, dst, radius, maxk, spaceOfsJoint, spaceWeights, expLUT));
        }
    }

    if (joint.type() == CV_8UC3)
    {
        if (src.type() == CV_8UC1)
        {
            parallel_for_(range, JointBilateralFilter_8u<Vec3b, Vec1b>(jointTemp, srcTemp, dst, radius, maxk, spaceOfsJoint, spaceWeights, expLUT));
        }
        if (src.type() == CV_8UC3)
        {
            parallel_for_(range, JointBilateralFilter_8u<Vec3b, Vec3b>(jointTemp, srcTemp, dst, radius, maxk, spaceOfsJoint, spaceWeights, expLUT));
        }
    }
}

void jointBilateralFilter(InputArray joint_, InputArray src_, OutputArray dst_, int d, double sigmaColor, double sigmaSpace, int borderType)
{
    CV_Assert(!src_.empty());

    if (joint_.empty())
    {
        bilateralFilter(src_, dst_, d, sigmaColor, sigmaSpace, borderType);
        return;
    }

    Mat src = src_.getMat();
    Mat joint = joint_.getMat();

    if (src.data == joint.data)
    {
        bilateralFilter(src_, dst_, d, sigmaColor, sigmaSpace, borderType);
        return;
    }

    CV_Assert(src.size() == joint.size());
    CV_Assert(src.depth() == joint.depth() && (src.depth() == CV_8U || src.depth() == CV_32F) );

    if (sigmaColor <= 0)
        sigmaColor = 1;
    if (sigmaSpace <= 0)
        sigmaSpace = 1;

    int radius;
    if (d <= 0)
        radius = cvRound(sigmaSpace*1.5);
    else
        radius = d / 2;
    radius = std::max(radius, 1);

    dst_.create(src.size(), src.type());
    Mat dst = dst_.getMat();

    if (dst.data == joint.data)
        joint = joint.clone();
    if (dst.data == src.data)
        src = src.clone();

    int jointCnNum = joint.channels();
    int srcCnNum = src.channels();

    if ( (srcCnNum == 1 || srcCnNum == 3) && (jointCnNum == 1 || jointCnNum == 3) )
    {
        if (joint.depth() == CV_8U)
        {
            jointBilateralFilter_8u(joint, src, dst, radius, sigmaColor, sigmaSpace, borderType);
        }
        else
        {
            jointBilateralFilter_32f(joint, src, dst, radius, sigmaColor, sigmaSpace, borderType);
        }
    }
    else
    {
        CV_Error(Error::BadNumChannels, "Unsupported number of channels");
    }
}

}
}
