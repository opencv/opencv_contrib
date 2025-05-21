/*
 * Copyright (c) 2025 Qualcomm Innovation Center, Inc. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
*/

#include "precomp.hpp"

namespace cv {
namespace fastcv {
namespace dsp {

void Canny(InputArray _src, OutputArray _dst, int lowThreshold, int highThreshold, int apertureSize, bool L2gradient)
{
    CV_Assert(
        !_src.empty() && 
        lowThreshold <= highThreshold &&
        IS_FASTCV_ALLOCATED(_src.getMat())
    );

    int type = _src.type();
    CV_Assert(type == CV_8UC1);
    CV_Assert(_src.step() % 8 == 0);

    Size size = _src.size();
    _dst.create(size, type);
    Mat src = _src.getMat();
    CV_Assert(src.step >= (size_t)src.cols);
    CV_Assert(reinterpret_cast<uintptr_t>(src.data) % 8 == 0);

    Mat dst = _dst.getMat();

    // Check if dst is allocated by the QcAllocator
    CV_Assert(IS_FASTCV_ALLOCATED(dst));
    CV_Assert(reinterpret_cast<uintptr_t>(dst.data) % 8 == 0);
    CV_Assert(dst.step >= (size_t)src.cols);

    // Check DSP initialization status and initialize if needed
    FASTCV_CHECK_DSP_INIT();

    fcvNormType norm;

    if (L2gradient)
        norm = FASTCV_NORM_L2;
    else
        norm = FASTCV_NORM_L1;

    int16_t* gx = (int16_t*)fcvHwMemAlloc(src.cols * src.rows * sizeof(int16_t), 16);
    int16_t* gy = (int16_t*)fcvHwMemAlloc(src.cols * src.rows * sizeof(int16_t), 16);
    uint32_t gstride = 2 * src.cols;
    fcvStatus status = fcvFilterCannyu8Q((uint8_t*)src.data, src.cols, src.rows, src.step, apertureSize, lowThreshold, highThreshold, norm, (uint8_t*)dst.data, dst.step, gx, gy, gstride);
    fcvHwMemFree(gx);
    fcvHwMemFree(gy);

    if (status != FASTCV_SUCCESS)
    {
        std::string s = fcvStatusStrings.count(status) ? fcvStatusStrings.at(status) : "unknown";
        CV_Error(cv::Error::StsInternal, "FastCV error: " + s);
    }
}

} // dsp::
} // fastcv::
} // cv::