/*
 * Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
*/

#include "precomp.hpp"

namespace cv {
namespace fastcv {

void resizeDownBy2(cv::InputArray _src, cv::OutputArray _dst)
{
    INITIALIZATION_CHECK;

    CV_Assert(!_src.empty() && _src.type() == CV_8UC1);

    Mat src = _src.getMat();
    CV_Assert((src.cols & 1)==0 && (src.rows & 1)==0);

    int type = _src.type();
    cv::Size dsize(src.cols / 2, src.rows / 2);

    _dst.create(dsize, type);

    Mat dst = _dst.getMat();

    fcvStatus status = (fcvStatus)fcvScaleDownBy2u8_v2((const uint8_t*)src.data, src.cols, src.rows, src.step, (uint8_t*)dst.data,
        src.cols/2);

    if (status != FASTCV_SUCCESS)
    {
        std::string s = fcvStatusStrings.count(status) ? fcvStatusStrings.at(status) : "unknown";
        CV_Error( cv::Error::StsInternal, "FastCV error: " + s);
    }
}

void resizeDownBy4(cv::InputArray _src, cv::OutputArray _dst)
{
    INITIALIZATION_CHECK;

    CV_Assert(!_src.empty() && _src.type() == CV_8UC1);

    Mat src = _src.getMat();
    CV_Assert((src.cols & 3)==0 && (src.rows & 3)==0);

    int type = _src.type();
    cv::Size dsize(src.cols / 4, src.rows / 4);

    _dst.create(dsize, type);

    Mat dst = _dst.getMat();

    fcvStatus status = (fcvStatus)fcvScaleDownBy4u8_v2((const uint8_t*)src.data, src.cols, src.rows, src.step,
        (uint8_t*)dst.data, src.cols/4);

    if (status != FASTCV_SUCCESS)
    {
        std::string s = fcvStatusStrings.count(status) ? fcvStatusStrings.at(status) : "unknown";
        CV_Error( cv::Error::StsInternal, "FastCV error: " + s);
    }
}

} // fastcv::
} // cv::
