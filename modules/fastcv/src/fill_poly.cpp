/*
 * Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
*/

#include "precomp.hpp"

namespace cv {
namespace fastcv {

void fillConvexPoly(InputOutputArray _img, InputArray _pts, Scalar color)
{
    INITIALIZATION_CHECK;

    CV_Assert(!_img.empty() && _img.depth() == CV_8U && _img.channels() <= 4);
    CV_Assert(_img.cols() % 8 == 0);
    CV_Assert(_img.step() % 8 == 0);

    Mat img = _img.getMat();

    CV_Assert(!_pts.empty() && (_pts.type() == CV_32SC1 || _pts.type() == CV_32SC2));
    CV_Assert(_pts.isContinuous());
    CV_Assert(_pts.total() * _pts.channels() % 2 == 0);

    Mat pts = _pts.getMat();
    uint32_t nPts = pts.total() * pts.channels() / 2;

    Vec4b coloru8 = color;

    fcvFillConvexPolyu8(nPts, (const uint32_t*)pts.data,
                         img.channels(), coloru8.val,
                         img.data, img.cols, img.rows, img.step);
}

} // fastcv::
} // cv::
