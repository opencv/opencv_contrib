/*
 * Copyright (c) 2025 Qualcomm Innovation Center, Inc. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
*/

#include "precomp.hpp"

namespace cv {
namespace fastcv {
namespace dsp {

void filter2D(InputArray _src, OutputArray _dst, int ddepth, InputArray _kernel)
{
    CV_Assert(
        !_src.empty() && 
        _src.type() == CV_8UC1 && 
        IS_FASTCV_ALLOCATED(_src.getMat()) && 
        IS_FASTCV_ALLOCATED(_kernel.getMat())
    );

    Mat kernel = _kernel.getMat();

    Size ksize = kernel.size();
    CV_Assert(ksize.width == ksize.height);
    CV_Assert(ksize.width % 2 == 1);

    _dst.create(_src.size(), ddepth);
    Mat src = _src.getMat();
    Mat dst = _dst.getMat();

    // Check if dst is allocated by the QcAllocator
    CV_Assert(IS_FASTCV_ALLOCATED(dst));

    // Check DSP initialization status and initialize if needed
    FASTCV_CHECK_DSP_INIT();

    switch (ddepth)
    {
        case CV_8U:
        {
            if(ksize.width == 3)
                fcvFilterCorr3x3s8_v2Q((int8_t*)kernel.data, src.data, src.cols, src.rows, src.step, dst.data, dst.step);
            else
                fcvFilterCorrNxNu8Q((int8_t*)kernel.data, ksize.width, 0, src.data, src.cols, src.rows, src.step, dst.data, dst.step);
            
            break;
        }
        case CV_16S:
        {
            fcvFilterCorrNxNu8s16Q((int8_t*)kernel.data, ksize.width, 0, src.data, src.cols, src.rows, src.step, (int16_t*)dst.data, dst.step);
            break;
        }
        case CV_32F:
        {
            fcvFilterCorrNxNu8f32Q((float32_t*)kernel.data, ksize.width, src.data, src.cols, src.rows, src.step, (float32_t*)dst.data, dst.step);
            break;
        }
        default:
        {
            CV_Error(cv::Error::StsBadArg, cv::format("Kernel Size:%d, Dst type:%s is not supported", ksize.width,
                depthToString(ddepth)));
        }
    }
}

} // dsp::
} // fastcv::
} // cv::