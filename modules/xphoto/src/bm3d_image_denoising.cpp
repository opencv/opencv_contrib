/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                        Intel License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective icvers.
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
//   * The name of Intel Corporation may not be used to endorse or promote products
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

#include "opencv2/xphoto.hpp"
#include "opencv2/core.hpp"

#ifdef OPENCV_ENABLE_NONFREE

#include "bm3d_denoising_invoker_step1.hpp"
#include "bm3d_denoising_invoker_step2.hpp"
#include "bm3d_denoising_transforms.hpp"

#endif

namespace cv
{
namespace xphoto
{

#ifdef OPENCV_ENABLE_NONFREE

template<typename ST, typename D, typename TT>
static void bm3dDenoising_(
    const Mat& src,
    Mat& basic,
    Mat& dst,
    const float& h,
    const int &templateWindowSize,
    const int &searchWindowSize,
    const int &hBMStep1,
    const int &hBMStep2,
    const int &groupSize,
    const int &slidingStep,
    const float &beta,
    const int &step)
{
    double granularity = (double)std::max(1., (double)src.total() / (1 << 16));

    switch (CV_MAT_CN(src.type())) {
    case 1:
        if (step == BM3D_STEP1 || step == BM3D_STEPALL)
        {
            parallel_for_(cv::Range(0, src.rows),
                Bm3dDenoisingInvokerStep1<ST, D, float, TT, HaarTransform<ST, TT> >(
                    src,
                    basic,
                    templateWindowSize,
                    searchWindowSize,
                    h,
                    hBMStep1,
                    groupSize,
                    slidingStep,
                    beta),
                granularity);
        }
        if (step == BM3D_STEP2 || step == BM3D_STEPALL)
        {
            parallel_for_(cv::Range(0, src.rows),
                Bm3dDenoisingInvokerStep2<ST, D, float, TT, HaarTransform<ST, TT> >(
                    src,
                    basic,
                    dst,
                    templateWindowSize,
                    searchWindowSize,
                    h,
                    hBMStep2,
                    groupSize,
                    slidingStep,
                    beta),
                granularity);
        }
        break;
    default:
        CV_Error(Error::StsBadArg,
            "Unsupported number of channels! Only 1 channel is supported at the moment.");
    }
}

void bm3dDenoising(
    InputArray _src,
    InputOutputArray _basic,
    OutputArray _dst,
    float h,
    int templateWindowSize,
    int searchWindowSize,
    int blockMatchingStep1,
    int blockMatchingStep2,
    int groupSize,
    int slidingStep,
    float beta,
    int normType,
    int step,
    int transformType)
{
    int type = _src.type(), depth = CV_MAT_DEPTH(type), cn = CV_MAT_CN(type);
    CV_Assert(1 == cn);
    CV_Assert(HAAR == transformType);
    CV_Assert(searchWindowSize > templateWindowSize);
    CV_Assert(slidingStep > 0 && slidingStep < templateWindowSize);

    Size srcSize = _src.size();

    switch (step)
    {
    case BM3D_STEP1:
        _basic.create(srcSize, type);
        break;
    case BM3D_STEP2:
        CV_Assert(type == _basic.type());
        _dst.create(srcSize, type);
        break;
    case BM3D_STEPALL:
        _dst.create(srcSize, type);
        break;
    default:
        CV_Error(Error::StsBadArg, "Unsupported BM3D step!");
    }

    Mat src = _src.getMat();
    Mat basic = _basic.getMat().empty() ? Mat(srcSize, type) : _basic.getMat();
    Mat dst = _dst.getMat();

    switch (normType) {
    case cv::NORM_L2:
        switch (depth) {
        case CV_8U:
            bm3dDenoising_<uchar, DistSquared, short>(
                src,
                basic,
                dst,
                h,
                templateWindowSize,
                searchWindowSize,
                blockMatchingStep1,
                blockMatchingStep2,
                groupSize,
                slidingStep,
                beta,
                step);
            break;
        default:
            CV_Error(Error::StsBadArg,
                "Unsupported depth! Only CV_8U is supported for NORM_L2");
        }
        break;
    case cv::NORM_L1:
        switch (depth) {
        case CV_8U:
            bm3dDenoising_<uchar, DistAbs, short>(
                src,
                basic,
                dst,
                h,
                templateWindowSize,
                searchWindowSize,
                blockMatchingStep1,
                blockMatchingStep2,
                groupSize,
                slidingStep,
                beta,
                step);
            break;
        case CV_16U:
            bm3dDenoising_<ushort, DistAbs, int>(
                src,
                basic,
                dst,
                h,
                templateWindowSize,
                searchWindowSize,
                blockMatchingStep1,
                blockMatchingStep2,
                groupSize,
                slidingStep,
                beta,
                step);
            break;
        default:
            CV_Error(Error::StsBadArg,
                "Unsupported depth! Only CV_8U and CV_16U are supported for NORM_L1");
        }
        break;
    default:
        CV_Error(Error::StsBadArg,
            "Unsupported norm type! Only NORM_L2 and NORM_L1 are supported");
    }
}

void bm3dDenoising(
    InputArray _src,
    OutputArray _dst,
    float h,
    int templateWindowSize,
    int searchWindowSize,
    int blockMatchingStep1,
    int blockMatchingStep2,
    int groupSize,
    int slidingStep,
    float beta,
    int normType,
    int step,
    int transformType)
{
    if (step == BM3D_STEP2)
        CV_Error(Error::StsBadArg,
            "Unsupported step type! To use BM3D_STEP2 one need to provide basic image.");

    Mat basic;

    bm3dDenoising(
        _src,
        basic,
        _dst,
        h,
        templateWindowSize,
        searchWindowSize,
        blockMatchingStep1,
        blockMatchingStep2,
        groupSize,
        slidingStep,
        beta,
        normType,
        step,
        transformType);

    if (step == BM3D_STEP1)
        _dst.assign(basic);
}

#else

void bm3dDenoising(
    InputArray _src,
    InputOutputArray _basic,
    OutputArray _dst,
    float h,
    int templateWindowSize,
    int searchWindowSize,
    int blockMatchingStep1,
    int blockMatchingStep2,
    int groupSize,
    int slidingStep,
    float beta,
    int normType,
    int step,
    int transformType)
{
    // Empty implementation

    CV_UNUSED(_src);
    CV_UNUSED(_basic);
    CV_UNUSED(_dst);
    CV_UNUSED(h);
    CV_UNUSED(templateWindowSize);
    CV_UNUSED(searchWindowSize);
    CV_UNUSED(blockMatchingStep1);
    CV_UNUSED(blockMatchingStep2);
    CV_UNUSED(groupSize);
    CV_UNUSED(slidingStep);
    CV_UNUSED(beta);
    CV_UNUSED(normType);
    CV_UNUSED(step);
    CV_UNUSED(transformType);

    CV_Error(Error::StsNotImplemented,
        "This algorithm is patented and is excluded in this configuration;"
        "Set OPENCV_ENABLE_NONFREE CMake option and rebuild the library");
}

void bm3dDenoising(
    InputArray _src,
    OutputArray _dst,
    float h,
    int templateWindowSize,
    int searchWindowSize,
    int blockMatchingStep1,
    int blockMatchingStep2,
    int groupSize,
    int slidingStep,
    float beta,
    int normType,
    int step,
    int transformType)
{
    // Empty implementation

    CV_UNUSED(_src);
    CV_UNUSED(_dst);
    CV_UNUSED(h);
    CV_UNUSED(templateWindowSize);
    CV_UNUSED(searchWindowSize);
    CV_UNUSED(blockMatchingStep1);
    CV_UNUSED(blockMatchingStep2);
    CV_UNUSED(groupSize);
    CV_UNUSED(slidingStep);
    CV_UNUSED(beta);
    CV_UNUSED(normType);
    CV_UNUSED(step);
    CV_UNUSED(transformType);

    CV_Error(Error::StsNotImplemented,
        "This algorithm is patented and is excluded in this configuration;"
        "Set OPENCV_ENABLE_NONFREE CMake option and rebuild the library");
}

#endif

}  // namespace xphoto
}  // namespace cv