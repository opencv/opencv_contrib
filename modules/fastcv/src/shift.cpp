/*
 * Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
*/

#include "precomp.hpp"

namespace cv {
namespace fastcv {

int meanShift(InputArray _src, Rect& rect, TermCriteria termCrit)
{
    INITIALIZATION_CHECK;

    CV_Assert(!_src.empty() && (_src.type() == CV_8UC1 || _src.type() == CV_32SC1 || _src.type() == CV_32FC1));
    CV_Assert(_src.cols() % 8 == 0);
    CV_Assert(_src.step() % 8 == 0);

    Mat src = _src.getMat();

    fcvRectangleInt window;
    window.x = rect.x;
    window.y = rect.y;
    window.width  = rect.width;
    window.height = rect.height;

    fcvTermCriteria criteria;
    criteria.epsilon  = (termCrit.type & TermCriteria::EPS) ? termCrit.epsilon : 0;
    criteria.max_iter = (termCrit.type & TermCriteria::COUNT) ? termCrit.maxCount : 1024;
    uint32_t nIterations = 0;
    if (src.depth() == CV_8U)
    {
        nIterations = fcvMeanShiftu8(src.data, src.cols, src.rows, src.step,
                                     &window, criteria);
    }
    else if (src.depth() == CV_32S)
    {
        nIterations = fcvMeanShifts32((const int *)src.data, src.cols, src.rows, src.step,
                                      &window, criteria);
    }
    else if (src.depth() == CV_32F)
    {
        nIterations = fcvMeanShiftf32((const float*)src.data, src.cols, src.rows, src.step,
                                       &window, criteria);
    }

    rect.x = window.x;
    rect.y = window.y;
    rect.width  = window.width;
    rect.height = window.height;

    return nIterations;
}

} // fastcv::
} // cv::
