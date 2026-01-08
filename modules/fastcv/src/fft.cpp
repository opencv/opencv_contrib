/*
 * Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
*/

#include "precomp.hpp"

namespace cv {
namespace fastcv {

static bool isPow2(int x)
{
    return x && (!(x & (x - 1)));
}

void FFT(InputArray _src, OutputArray _dst)
{
    INITIALIZATION_CHECK;

    CV_Assert(!_src.empty() && _src.type() == CV_8UC1);
    CV_Assert(isPow2(_src.rows()) || _src.rows() == 1);
    CV_Assert(isPow2(_src.cols()));
    CV_Assert(_src.step() % 8 == 0);

    Mat src = _src.getMat();

    _dst.create(_src.rows(), _src.cols(), CV_32FC2);
    // in case of fixed layout array we cannot fix this on our side, can only fail if false
    CV_Assert(_dst.step() % 8 == 0);

    Mat dst = _dst.getMat();

    fcvStatus status = fcvFFTu8(src.data, src.cols, src.rows, src.step,
                                (float*)dst.data, dst.step);

    if (status != FASTCV_SUCCESS)
    {
        std::string s = fcvStatusStrings.count(status) ? fcvStatusStrings.at(status) : "unknown";
        CV_Error( cv::Error::StsInternal, "FastCV error: " + s);
    }
}

void IFFT(InputArray _src, OutputArray _dst)
{
    INITIALIZATION_CHECK;

    CV_Assert(!_src.empty() && _src.type() == CV_32FC2);
    CV_Assert(isPow2(_src.rows()) || _src.rows() == 1);
    CV_Assert(isPow2(_src.cols()));
    // in case of fixed layout array we cannot fix this on our side, can only fail if false
    CV_Assert(_src.step() % 8 == 0);

    Mat src = _src.getMat();

    _dst.create(_src.rows(), _src.cols(), CV_8UC1);
    // in case of fixed layout array we cannot fix this on our side, can only fail if false
    CV_Assert(_dst.step() % 8 == 0);

    Mat dst = _dst.getMat();

    fcvStatus status = fcvIFFTf32((const float*)src.data, src.cols * 2, src.rows, src.step,
                                  dst.data, dst.step);

    if (status != FASTCV_SUCCESS)
    {
        std::string s = fcvStatusStrings.count(status) ? fcvStatusStrings.at(status) : "unknown";
        CV_Error( cv::Error::StsInternal, "FastCV error: " + s);
    }
}

} // fastcv::
} // cv::
