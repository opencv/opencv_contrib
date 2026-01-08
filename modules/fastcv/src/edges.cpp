/*
 * Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
*/

#include "precomp.hpp"

namespace cv {
namespace fastcv {

void sobel3x3u8(cv::InputArray _src, cv::OutputArray _dst, cv::OutputArray _dsty, int ddepth, bool normalization)
{
    INITIALIZATION_CHECK;

    CV_Assert(!_src.empty() && _src.type() == CV_8UC1);

    Size size = _src.size();
    _dst.create(size, ddepth);
    Mat src = _src.getMat();
    Mat dst = _dst.getMat();
    if (_dsty.needed())
    {
        _dsty.create(size, ddepth);
        Mat dsty = _dsty.getMat();

        switch(ddepth)
        {
            case CV_8S:
                if (normalization)
                    fcvImageGradientSobelPlanars8_v2(src.data, src.cols, src.rows, src.step, (int8_t*)dst.data,
                        (int8_t*)dsty.data, dst.step);
                else
                    CV_Error(cv::Error::StsBadArg,
                        cv::format("Depth: %d should do normalization, make sure the normalization parameter is true", ddepth));
                break;
            case CV_16S:
                if (normalization)
                    fcvImageGradientSobelPlanars16_v2(src.data, src.cols, src.rows, src.step, (int16_t*)dst.data,
                        (int16_t*)dsty.data, dst.step);
                else
                    fcvImageGradientSobelPlanars16_v3(src.data, src.cols, src.rows, src.step, (int16_t*)dst.data,
                        (int16_t*)dsty.data, dst.step);
                break;
            case CV_32F:
                if (normalization)
                    fcvImageGradientSobelPlanarf32_v2(src.data, src.cols, src.rows, src.step, (float32_t*)dst.data,
                        (float32_t*)dsty.data, dst.step);
                else
                    fcvImageGradientSobelPlanarf32_v3(src.data, src.cols, src.rows, src.step, (float32_t*)dst.data,
                        (float32_t*)dsty.data, dst.step);
                break;
            default:
                CV_Error(cv::Error::StsBadArg, cv::format("depth: %d is not supported", ddepth));
                break;
        }
    }
    else
    {
        fcvFilterSobel3x3u8_v2(src.data, src.cols, src.rows, src.step, dst.data, dst.step);
    }
}

void sobel(cv::InputArray _src, cv::OutputArray _dx, cv::OutputArray _dy, int kernel_size, int borderType, int borderValue)
{
    INITIALIZATION_CHECK;

    CV_Assert(!_src.empty() && _src.type() == CV_8UC1);
    Size size = _src.size();
    _dx.create( size, CV_16SC1);
    _dy.create( size, CV_16SC1);

    Mat src = _src.getMat();
    Mat dx = _dx.getMat();
    Mat dy = _dy.getMat();
    fcvStatus status = FASTCV_SUCCESS;

    fcvBorderType   fcvBorder;

    switch (borderType)
    {
        case cv::BorderTypes::BORDER_CONSTANT:
        {
            fcvBorder = fcvBorderType::FASTCV_BORDER_CONSTANT;
            break;
        }
        case cv::BorderTypes::BORDER_REPLICATE:
        {
            fcvBorder = fcvBorderType::FASTCV_BORDER_REPLICATE;
            break;
        }
        default:
        {
            CV_Error(cv::Error::StsBadArg, cv::format("Border type: %d is not supported", borderType));
           break;
        }
    }

    switch (kernel_size)
    {
        case 3:
            status = fcvFilterSobel3x3u8s16(src.data, src.cols, src.rows, src.step, (int16_t*)dx.data, (int16_t*)dy.data,
                dx.step, fcvBorder, borderValue);
            break;
        case 5:
            status = fcvFilterSobel5x5u8s16(src.data, src.cols, src.rows, src.step, (int16_t*)dx.data, (int16_t*)dy.data,
                dx.step, fcvBorder, borderValue);
            break;
        case 7:
            status = fcvFilterSobel7x7u8s16(src.data, src.cols, src.rows, src.step, (int16_t*)dx.data, (int16_t*)dy.data,
                dx.step, fcvBorder, borderValue);
            break;
        default:
            CV_Error(cv::Error::StsBadArg, cv::format("Kernel size %d is not supported", kernel_size));
            break;
    }

    if (status != FASTCV_SUCCESS)
    {
        std::string s = fcvStatusStrings.count(status) ? fcvStatusStrings.at(status) : "unknown";
        CV_Error( cv::Error::StsInternal, "FastCV error: " + s);
    }
}

} // fastcv::
} // cv::