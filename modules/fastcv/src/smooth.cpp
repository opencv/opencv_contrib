/*
 * Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
*/

#include "precomp.hpp"

namespace cv {
namespace fastcv {

void bilateralRecursive(cv::InputArray _src, cv::OutputArray _dst, float sigmaColor, float sigmaSpace)
{
    INITIALIZATION_CHECK;

    CV_Assert(!_src.empty() && _src.type() == CV_8UC1);
    CV_Assert(_src.step() % 8 == 0);

    Size size = _src.size();
    int type = _src.type();
    _dst.create(size, type);
    // in case of fixed layout array we cannot fix this on our side, can only fail if false
    CV_Assert(_dst.step() % 8 == 0);

    Mat src = _src.getMat();
    Mat dst = _dst.getMat();

    fcvStatus status  = fcvBilateralFilterRecursiveu8(src.data, src.cols, src.rows, src.step,
                                                      dst.data, dst.step, sigmaColor, sigmaSpace);
    if (status != FASTCV_SUCCESS)
    {
        std::string s = fcvStatusStrings.count(status) ? fcvStatusStrings.at(status) : "unknown";
        CV_Error( cv::Error::StsInternal, "FastCV error: " + s);
    }
}

} // fastcv::
} // cv::
