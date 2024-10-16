/*
 * Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
*/

#include "precomp.hpp"

namespace cv {
namespace fastcv {

#ifdef FAST_CV_FOUND

void matmuls8s32(InputArray _src1, _InputArray _src2, OutputArray _dst)
{
    INITIALIZATION_CHECK;
    CV_Assert(!_src1.empty() && _src1.type() == CV_8SC1);
    CV_Assert(_src1.cols() <= 131072);
    CV_Assert(_src1.step() % 8 == 0);
    CV_Assert(_src1.cols() == _src2.rows());
    Mat src1 = _src1.getMat();

    CV_Assert(!_src2.empty() && _src2.type() == CV_8SC1);
    CV_Assert(_src2.step() % 8 == 0);
    Mat src2 = _src2.getMat();

    _dst.create(_src1.rows(), _src2.cols(), CV_32SC1);
    // in case of fixed layout array we cannot fix this on our side, can only fail if false
    CV_Assert(_dst.step() % 8 == 0);
    Mat dst = _dst.getMat();

    fcvMatrixMultiplys8s32((const int8_t*)src1.data, src1.cols, src1.rows, src1.step,
                           (const int8_t*)src2.data, src2.cols, src2.step,
                           (int32_t*)dst.data, dst.step);
}

#else

void matmuls8s32(InputArray _src1, _InputArray _src2, OutputArray _dst)
{
    CV_UNUSED(_src1);
    CV_UNUSED(_src2);
    CV_UNUSED(_dst);
    CV_Error( cv::Error::StsNotImplemented, "OpenCV was build without FastCV support" );
}

#endif

} // fastcv::
} // cv::
