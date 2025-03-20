/*
 * Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
*/

#include "precomp.hpp"

namespace cv {
namespace fastcv {
namespace dsp {

void sumOfAbsoluteDiffs(cv::InputArray _patch, cv::InputArray _src, cv::OutputArray _dst)
{
    CV_Assert(!_src.empty() && _src.type() == CV_8UC1);
    CV_Assert(_src.step() * _src.rows() > MIN_REMOTE_BUF_SIZE);
    CV_Assert(!_patch.empty() && _patch.type() == CV_8UC1);
    CV_Assert(_patch.size() == Size(8, 8));

    Size size = _src.size();
    _dst.create(size, CV_16UC1);

    Mat patch = _patch.getMat();
    Mat src = _src.getMat();
    CV_Assert(((intptr_t)src.data & 0x7) == 0);

    Mat dst = _dst.getMat();
    CV_Assert(((intptr_t)dst.data & 0x7) == 0);

    fcvSumOfAbsoluteDiffs8x8u8_v2Q((uint8_t*)patch.data, patch.step, (uint8_t*)src.data, src.cols, src.rows, src.step, (uint16_t*)dst.data, dst.step);
}

} // dsp::
} // fastcv::
} // cv::
