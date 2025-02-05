/*
 * Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
*/

#include "precomp.hpp"

namespace cv {
namespace fastcv {

void DCT(InputArray _src, OutputArray _dst)
{
    INITIALIZATION_CHECK;
    CV_Assert(!_src.empty() && _src.type() == CV_8UC1);
    CV_Assert(_src.cols() % 8 == 0);
    CV_Assert(_src.step() % 8 == 0);

    Mat src = _src.getMat();

    _dst.create(_src.rows(), _src.cols(), CV_16SC1);
    // in case of fixed layout array we cannot fix this on our side, can only fail if false
    CV_Assert(_dst.step() % 8 == 0);

    Mat dst = _dst.getMat();

    fcvDCTu8(src.data, src.cols, src.rows, src.step, (short*)dst.data, dst.step);
}

void IDCT(InputArray _src, OutputArray _dst)
{
    INITIALIZATION_CHECK;
    CV_Assert(!_src.empty() && _src.type() == CV_16SC1);
    CV_Assert(_src.cols() % 8 == 0);
    CV_Assert(_src.step() % 8 == 0);

    Mat src = _src.getMat();

    _dst.create(_src.rows(), _src.cols(), CV_8UC1);
    // in case of fixed layout array we cannot fix this on our side, can only fail if false
    CV_Assert(_dst.step() % 8 == 0);

    Mat dst = _dst.getMat();

    fcvIDCTs16((const short*)src.data, src.cols, src.rows, src.step, dst.data, dst.step);
}

} // fastcv::
} // cv::
