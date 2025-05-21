/*
 * Copyright (c) 2025 Qualcomm Innovation Center, Inc. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
*/

#include "precomp.hpp"

namespace cv {
namespace fastcv {
namespace dsp {

    void thresholdOtsu(InputArray _src, OutputArray _dst, bool type)
    {
        CV_Assert(
            !_src.empty() && 
            _src.type() == CV_8UC1 && 
            IS_FASTCV_ALLOCATED(_src.getMat())
        );

        CV_Assert((_src.step() * _src.rows()) > MIN_REMOTE_BUF_SIZE);
        CV_Assert(_src.cols() % 8 == 0);
        CV_Assert(_src.step() % 8 == 0);

        Mat src = _src.getMat();
        CV_Assert(((uintptr_t)src.data & 0x7) == 0);

        _dst.create(_src.size(), CV_8UC1);
        CV_Assert(_dst.step() % 8 == 0);
        CV_Assert(_dst.cols() % 8 == 0);
        Mat dst = _dst.getMat();

        // Check if dst is allocated by the QcAllocator
        CV_Assert(IS_FASTCV_ALLOCATED(dst));
        CV_Assert(((uintptr_t)dst.data & 0x7) == 0);
        
        if (src.data == dst.data) {
            CV_Assert(src.step == dst.step);
        }

        // Check DSP initialization status and initialize if needed
        FASTCV_CHECK_DSP_INIT();

        fcvThreshType threshType;

        if (type)
            threshType = FCV_THRESH_BINARY_INV;
        else
            threshType = FCV_THRESH_BINARY;

        fcvFilterThresholdOtsuu8Q(src.data, src.cols, src.rows, src.step, dst.data, dst.step, threshType);
    }

} // dsp::
} // fastcv::
} // cv::