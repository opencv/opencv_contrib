/*
 * Copyright (c) 2024-2025 Qualcomm Innovation Center, Inc. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
*/

#include "precomp.hpp"

namespace cv {
namespace fastcv {

class FcvWarpPerspectiveLoop_Invoker : public cv::ParallelLoopBody
{
    public:

    FcvWarpPerspectiveLoop_Invoker(const Mat& _src1, const Mat& _src2, Mat& _dst1, Mat& _dst2,
        const float * _M, fcvInterpolationType _interpolation = FASTCV_INTERPOLATION_TYPE_NEAREST_NEIGHBOR,
        fcvBorderType _borderType = fcvBorderType::FASTCV_BORDER_UNDEFINED, const int _borderValue = 0)
        : ParallelLoopBody(), src1(_src1), src2(_src2), dst1(_dst1), dst2(_dst2), M(_M), interpolation(_interpolation),
        borderType(_borderType), borderValue(_borderValue)
    {}

    virtual void operator()(const cv::Range& range) const CV_OVERRIDE
    {
        uchar* dst1_ptr = dst1.data + range.start * dst1.step;
        int rangeHeight = range.end - range.start;

        float rangeMatrix[9];
        rangeMatrix[0] = M[0];
        rangeMatrix[1] = M[1];
        rangeMatrix[2] = M[2]+range.start*M[1];
        rangeMatrix[3] = M[3];
        rangeMatrix[4] = M[4];
        rangeMatrix[5] = M[5]+range.start*M[4];
        rangeMatrix[6] = M[6];
        rangeMatrix[7] = M[7];
        rangeMatrix[8] = M[8]+range.start*M[7];

        if ((src2.empty()) || (dst2.empty()))
        {
            fcvWarpPerspectiveu8_v5(src1.data, src1.cols, src1.rows, src1.step, src1.channels(), dst1_ptr, dst1.cols, rangeHeight,
                dst1.step, rangeMatrix, interpolation, borderType, borderValue);
        }
        else
        {
            uchar* dst2_ptr = dst2.data + range.start * dst2.step;
            fcv2PlaneWarpPerspectiveu8(src1.data, src2.data, src1.cols, src1.rows, src1.step, src2.step, dst1_ptr, dst2_ptr,
                dst1.cols, rangeHeight, dst1.step, dst2.step, rangeMatrix);
        }
    }

    private:

    const Mat&              src1;
    const Mat&              src2;
    Mat&                    dst1;
    Mat&                    dst2;
    const float*            M;
    fcvInterpolationType    interpolation;
    fcvBorderType           borderType;
    int                     borderValue;

    FcvWarpPerspectiveLoop_Invoker(const FcvWarpPerspectiveLoop_Invoker &);  // = delete;
    const FcvWarpPerspectiveLoop_Invoker& operator= (const FcvWarpPerspectiveLoop_Invoker &);  // = delete;
};

void warpPerspective2Plane(InputArray _src1, InputArray _src2, OutputArray _dst1, OutputArray _dst2, InputArray _M0,
        Size dsize)
{
    INITIALIZATION_CHECK;
    CV_Assert(!_src1.empty() && _src1.type() == CV_8UC1);
    CV_Assert(!_src2.empty() && _src2.type() == CV_8UC1);
    CV_Assert(!_M0.empty());

    Mat src1 = _src1.getMat();
    Mat src2 = _src2.getMat();

    _dst1.create(dsize, src1.type());
    _dst2.create(dsize, src2.type());
    Mat dst1 = _dst1.getMat();
    Mat dst2 = _dst2.getMat();

    Mat M0 = _M0.getMat();
    CV_Assert((M0.type() == CV_32F || M0.type() == CV_64F) && M0.rows == 3 && M0.cols == 3);
    float matrix[9];
    Mat M(3, 3, CV_32F, matrix);
    M0.convertTo(M, M.type());

    int nThreads = getNumThreads();
    int nStripes = nThreads > 1 ? 2*nThreads : 1;

    cv::parallel_for_(cv::Range(0, dsize.height),
        FcvWarpPerspectiveLoop_Invoker(src1, src2, dst1, dst2, matrix), nStripes);
}

void warpPerspective(InputArray _src, OutputArray _dst, InputArray _M0, Size dsize, int interpolation, int borderType,
    const Scalar&  borderValue)
{
    Mat src = _src.getMat();

    _dst.create(dsize, src.type());
    Mat dst = _dst.getMat();

    Mat M0 = _M0.getMat();
    CV_Assert((M0.type() == CV_32F || M0.type() == CV_64F) && M0.rows == 3 && M0.cols == 3);
    float matrix[9];
    Mat M(3, 3, CV_32F, matrix);
    M0.convertTo(M, M.type());

    // Do not support inplace case
    CV_Assert(src.data != dst.data);
    // Only support CV_8U
    CV_Assert(src.depth() == CV_8U);

    INITIALIZATION_CHECK;

    fcvBorderType           fcvBorder;
    uint8_t                 fcvBorderValue = 0;
    fcvInterpolationType    fcvInterpolation;

    switch (borderType)
    {
        case BORDER_CONSTANT:
        {
            // Border value should be same
            CV_Assert((borderValue[0] == borderValue[1]) &&
                      (borderValue[0] == borderValue[2]) &&
                      (borderValue[0] == borderValue[3]));

            fcvBorder       = fcvBorderType::FASTCV_BORDER_CONSTANT;
            fcvBorderValue  = static_cast<uint8_t>(borderValue[0]);
            break;
        }
        case BORDER_REPLICATE:
        {
            fcvBorder = fcvBorderType::FASTCV_BORDER_REPLICATE;
            break;
        }
        case BORDER_TRANSPARENT:
        {
            fcvBorder = fcvBorderType::FASTCV_BORDER_UNDEFINED;
            break;
        }
        default:
            CV_Error(cv::Error::StsBadArg, cv::format("Border type:%d is not supported", borderType));
    }

    switch(interpolation)
    {
        case INTER_NEAREST:
        {
            fcvInterpolation = FASTCV_INTERPOLATION_TYPE_NEAREST_NEIGHBOR;
            break;
        }
        case INTER_LINEAR:
        {
            fcvInterpolation = FASTCV_INTERPOLATION_TYPE_BILINEAR;
            break;
        }
        case INTER_AREA:
        {
            fcvInterpolation = FASTCV_INTERPOLATION_TYPE_AREA;
            break;
        }
        default:
            CV_Error(cv::Error::StsBadArg, cv::format("Interpolation type:%d is not supported", interpolation));
    }

    int nThreads = cv::getNumThreads();
    int nStripes = nThreads > 1 ? 2*nThreads : 1;

    // placeholder
    Mat tmp;

    cv::parallel_for_(cv::Range(0, dsize.height),
        FcvWarpPerspectiveLoop_Invoker(src, tmp, dst, tmp, matrix, fcvInterpolation, fcvBorder, fcvBorderValue), nStripes);
}

void warpAffine(InputArray _src, OutputArray _dst, InputArray _M, Size dsize,
                int interpolation, int borderValue)
{
    INITIALIZATION_CHECK;
    CV_Assert(!_src.empty());
    CV_Assert(!_M.empty());

    Mat src = _src.getMat();
    Mat M = _M.getMat();

    CV_CheckEQ(M.rows, 2, "Affine Matrix must have 2 rows");
    CV_Check(M.cols, M.cols == 2 || M.cols == 3, "Affine Matrix must be 2x2 or 2x3");

    if (M.rows == 2 && M.cols == 2)
    {
        CV_CheckTypeEQ(src.type(), CV_8UC1, "2x2 matrix transformation only supports CV_8UC1");

        // Check if src is a ROI
        Size wholeSize;
        Point ofs;
        src.locateROI(wholeSize, ofs);
        bool isROI = (wholeSize.width > src.cols || wholeSize.height > src.rows);

        Mat fullImage;
        Point2f center;

        if (isROI)
        {
            center.x = ofs.x + src.cols / 2.0f;
            center.y = ofs.y + src.rows / 2.0f;

            CV_Check(center.x, center.x >= 0 && center.x < wholeSize.width, "ROI center X is outside full image bounds");
            CV_Check(center.y, center.y >= 0 && center.y < wholeSize.height, "ROI center Y is outside full image bounds");

            size_t offset = ofs.y * src.step + ofs.x * src.elemSize();
            fullImage = Mat(wholeSize, src.type(), src.data - offset);
        }
        else
        {
            // Use src as is, center at image center
            fullImage = src;
            center.x = src.cols / 2.0f;
            center.y = src.rows / 2.0f;

            CV_LOG_WARNING(NULL, "2x2 matrix with non-ROI input. Using image center for patch extraction.");
        }

        float affineMatrix[4] = {
            M.at<float>(0, 0), M.at<float>(0, 1),
            M.at<float>(1, 0), M.at<float>(1, 1)};

        float position[2] = {center.x, center.y};

        _dst.create(dsize, src.type());
        Mat dst = _dst.getMat();
        dst.step = dst.cols * src.elemSize();

        int status = fcvTransformAffineu8_v2(
            (const uint8_t *)fullImage.data,
            fullImage.cols, fullImage.rows, fullImage.step,
            position,
            affineMatrix,
            (uint8_t *)dst.data,
            dst.cols, dst.rows, dst.step);

        if (status != 0)
        {
            CV_Error(Error::StsInternal, "FastCV patch extraction failed");
        }

        return;
    }

    // Validate 2x3 matrix for standard transformation
    CV_CheckEQ(M.cols, 3, "Matrix must be 2x3 for standard affine transformation");
    CV_Check(src.type(), src.type() == CV_8UC1 || src.type() == CV_8UC3, "Standard transformation supports CV_8UC1 or CV_8UC3");

    float32_t affineMatrix[6] = {
        M.at<float>(0, 0), M.at<float>(0, 1), M.at<float>(0, 2),
        M.at<float>(1, 0), M.at<float>(1, 1), M.at<float>(1, 2)};

    _dst.create(dsize, src.type());
    Mat dst = _dst.getMat();

    if (src.channels() == 1)
    {
        fcvStatus status;
        fcvInterpolationType fcvInterpolation;

        switch (interpolation)
        {
        case cv::InterpolationFlags::INTER_NEAREST:
            fcvInterpolation = FASTCV_INTERPOLATION_TYPE_NEAREST_NEIGHBOR;
            break;
        case cv::InterpolationFlags::INTER_LINEAR:
            fcvInterpolation = FASTCV_INTERPOLATION_TYPE_BILINEAR;
            break;
        case cv::InterpolationFlags::INTER_AREA:
            fcvInterpolation = FASTCV_INTERPOLATION_TYPE_AREA;
            break;
        default:
            CV_Error(cv::Error::StsBadArg, "Unsupported interpolation type");
        }

        status = fcvTransformAffineClippedu8_v3(
            (const uint8_t *)src.data, src.cols, src.rows, src.step,
            affineMatrix,
            (uint8_t *)dst.data, dst.cols, dst.rows, dst.step,
            NULL,
            fcvInterpolation,
            FASTCV_BORDER_CONSTANT,
            borderValue);

        if (status != FASTCV_SUCCESS)
        {
            std::string s = fcvStatusStrings.count(status) ? fcvStatusStrings.at(status) : "unknown";
            CV_Error(cv::Error::StsInternal, "FastCV error: " + s);
        }
    }
    else if (src.channels() == 3)
    {
        CV_LOG_INFO(NULL, "warpAffine: 3-channel images use bicubic interpolation internally.");

        std::vector<uint32_t> dstBorder;
        try
        {
            dstBorder.resize(dsize.height * 2);
        }
        catch (const std::bad_alloc &)
        {
            CV_Error(Error::StsNoMem, "Failed to allocate border array");
        }

        fcv3ChannelTransformAffineClippedBCu8(
            (const uint8_t *)src.data, src.cols, src.rows, src.step[0],
            affineMatrix,
            (uint8_t *)dst.data, dst.cols, dst.rows, dst.step[0],
            dstBorder.data());
    }
}

} // fastcv::
} // cv::