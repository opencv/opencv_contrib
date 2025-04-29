/*
 * Copyright (c) 2025 Qualcomm Innovation Center, Inc. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
*/

#include "precomp.hpp"

namespace cv {
namespace fastcv {
namespace dsp {

static bool isPow2(int x)
{
    return x && (!(x & (x - 1)));
}

void FFT(InputArray _src, OutputArray _dst)
{
    CV_Assert(
        !_src.empty() && 
        _src.type() == CV_8UC1 && 
        IS_FASTCV_ALLOCATED(_src.getMat())
    );

    CV_Assert(isPow2(_src.rows()) || _src.rows() == 1);
    CV_Assert(isPow2(_src.cols()));
    CV_Assert(_src.step() % 8 == 0);
    CV_Assert(static_cast<unsigned long>(_src.rows() * _src.cols()) > MIN_REMOTE_BUF_SIZE);

    Mat src = _src.getMat();
    CV_Assert(reinterpret_cast<uintptr_t>(src.data) % 8 == 0);

    _dst.create(_src.rows(), _src.cols(), CV_32FC2);
    CV_Assert(_dst.step() % 8 == 0);
    Mat dst = _dst.getMat();

    // Check if dst is allocated by the QcAllocator
    CV_Assert(IS_FASTCV_ALLOCATED(dst));
    CV_Assert(reinterpret_cast<uintptr_t>(dst.data) % 8 == 0);
    
    // Check DSP initialization status and initialize if needed
    FASTCV_CHECK_DSP_INIT();

    fcvStatus status = fcvFFTu8Q(src.data, src.cols, src.rows, src.step,
        (float*)dst.data, dst.step);

    if (status != FASTCV_SUCCESS)
    {
        std::string s = fcvStatusStrings.count(status) ? fcvStatusStrings.at(status) : "unknown";
        CV_Error(cv::Error::StsInternal, "FastCV error: " + s);
    }
}

void IFFT(InputArray _src, OutputArray _dst)
{
    CV_Assert(
        !_src.empty() && 
        _src.type() == CV_32FC2 &&
        IS_FASTCV_ALLOCATED(_src.getMat())
    );

    CV_Assert(isPow2(_src.rows()) || _src.rows() == 1);
    CV_Assert(isPow2(_src.cols()));

    CV_Assert(_src.step() % 8 == 0);
    CV_Assert(static_cast<unsigned long>(_src.rows() * _src.cols() * sizeof(float32_t)) > MIN_REMOTE_BUF_SIZE);

    Mat src = _src.getMat();

    CV_Assert(reinterpret_cast<uintptr_t>(src.data) % 8 == 0);

    _dst.create(_src.rows(), _src.cols(), CV_8UC1);

    CV_Assert(_dst.step() % 8 == 0);

    Mat dst = _dst.getMat();
    // Check if dst is allocated by the QcAllocator
    CV_Assert(IS_FASTCV_ALLOCATED(dst));
    CV_Assert(reinterpret_cast<uintptr_t>(dst.data) % 8 == 0);

    // Check DSP initialization status and initialize if needed
    FASTCV_CHECK_DSP_INIT();

    fcvStatus status = fcvIFFTf32Q((const float*)src.data, src.cols * 2, src.rows, src.step,
        dst.data, dst.step);

    if (status != FASTCV_SUCCESS)
    {
        std::string s = fcvStatusStrings.count(status) ? fcvStatusStrings.at(status) : "unknown";
        CV_Error(cv::Error::StsInternal, "FastCV error: " + s);
    }
}

} // dsp::
} // fastcv::
} // cv::