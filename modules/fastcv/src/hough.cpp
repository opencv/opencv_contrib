/*
 * Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
*/

#include "precomp.hpp"

namespace cv {
namespace fastcv {

void houghLines(InputArray _src, OutputArray _lines, double threshold)
{
    INITIALIZATION_CHECK;

    CV_Assert(!_src.empty() && _src.type() == CV_8UC1);
    CV_Assert(_src.cols() % 8 == 0);
    CV_Assert(_src.step() % 8 == 0);

    Mat src = _src.getMat();

    const uint32_t maxLines = 16384;

    cv::Mat lines(1, maxLines, CV_32FC4);

    uint32_t nLines = maxLines;

    fcvHoughLineu8(src.data, src.cols, src.rows, src.step,
                   (float)threshold, maxLines, &nLines, (fcvLine*)lines.data);

    _lines.create(1, nLines, CV_32FC4);
    lines(Range::all(), Range(0, nLines)).copyTo(_lines);
}

} // fastcv::
} // cv::
