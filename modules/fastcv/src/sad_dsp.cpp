/*
 * Copyright (c) 2025 Qualcomm Innovation Center, Inc. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
*/

#include "precomp.hpp"

namespace cv {
namespace fastcv {
namespace dsp {

void sumOfAbsoluteDiffs(cv::InputArray _patch, cv::InputArray _src, cv::OutputArray _dst) 
{
    cv::Mat patch = _patch.getMat();
    cv::Mat src = _src.getMat();
    
    // Check if matrices are allocated by the QcAllocator
    CV_Assert(IS_FASTCV_ALLOCATED(patch));
    CV_Assert(IS_FASTCV_ALLOCATED(src));
    
    CV_Assert(!_src.empty() && "src is empty");
    CV_Assert(_src.type() == CV_8UC1 && "src type is not CV_8UC1");
    CV_Assert(_src.step() * _src.rows() > MIN_REMOTE_BUF_SIZE && "src buffer size is too small");
    CV_Assert(!_patch.empty() && "patch is empty");
    CV_Assert(_patch.type() == CV_8UC1 && "patch type is not CV_8UC1");
    CV_Assert(_patch.size() == cv::Size(8, 8) && "patch size is not 8x8");

    cv::Size size = _src.size();
    _dst.create(size, CV_16UC1);
    cv::Mat dst = _dst.getMat();

    CV_Assert(((intptr_t)src.data & 0x7) == 0 && "src data is not 8-byte aligned");
    CV_Assert(((intptr_t)dst.data & 0x7) == 0 && "dst data is not 8-byte aligned");
    
    // Check if dst is allocated by the QcAllocator
    CV_Assert(IS_FASTCV_ALLOCATED(dst));

    // Check DSP initialization status and initialize if needed
    FASTCV_CHECK_DSP_INIT();
    
    fcvSumOfAbsoluteDiffs8x8u8_v2Q((uint8_t*)patch.data, patch.step, (uint8_t*)src.data, src.cols, src.rows, src.step, (uint16_t*)dst.data, dst.step);
}

} // dsp::
} // fastcv::
} // cv::
