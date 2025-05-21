/*
 * Copyright (c) 2025 Qualcomm Innovation Center, Inc. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
*/

#include "precomp.hpp"

namespace cv {
namespace fastcv {

void resizeDown(cv::InputArray _src, cv::OutputArray _dst, Size dsize, double inv_scale_x, double inv_scale_y)
{
    fcvStatus status = FASTCV_SUCCESS;
    Size ssize = _src.size();

    CV_Assert(!_src.empty() );
    CV_Assert( _src.type() == CV_8UC1 || _src.type() == CV_8UC2 );

    if( dsize.empty() )
    {
        CV_Assert(inv_scale_x > 0);
        CV_Assert(inv_scale_y > 0);
        dsize = Size(saturate_cast<int>(ssize.width*inv_scale_x),
                     saturate_cast<int>(ssize.height*inv_scale_y));
        CV_Assert( !dsize.empty() );
    }
    else
    {
        inv_scale_x = static_cast<double>(dsize.width) / ssize.width;
        inv_scale_y = static_cast<double>(dsize.height) / ssize.height;
        CV_Assert(inv_scale_x > 0);
        CV_Assert(inv_scale_y > 0);
    }

    CV_Assert(dsize.width <= ssize.width && dsize.height <= ssize.height);

    CV_Assert(dsize.width * 20 > ssize.width);
    CV_Assert(dsize.height * 20 > ssize.height);

    INITIALIZATION_CHECK;

    Mat src = _src.getMat();
    _dst.create(dsize, src.type());
    Mat dst = _dst.getMat();

    // Alignment checks
    CV_Assert(reinterpret_cast<uintptr_t>(src.data) % 16 == 0);
    CV_Assert(reinterpret_cast<uintptr_t>(dst.data) % 16 == 0);

    if(src.type() == CV_8UC2)
    {
        fcvScaleDownMNInterleaveu8((const uint8_t*)src.data, src.cols, src.rows, src.step, (uint8_t*)dst.data, dst.cols, dst.rows, dst.step);
    }
    else if (src.cols/dst.cols == 4 && src.rows/dst.rows == 4 && src.cols % dst.cols == 0 && src.rows % dst.rows == 0)
    {
        CV_Assert(src.rows % 4 == 0);
        status = (fcvStatus)fcvScaleDownBy4u8_v2((const uint8_t*)src.data, src.cols, src.rows, src.step, (uint8_t*)dst.data, dst.step);
    }
    else if (src.cols/dst.cols == 2 && src.rows/dst.rows == 2 && src.cols % dst.cols == 0 && src.rows % dst.rows == 0)
    {
        CV_Assert(src.rows % 2 == 0);
        status = (fcvStatus)fcvScaleDownBy2u8_v2((const uint8_t*)src.data, src.cols, src.rows, src.step, (uint8_t*)dst.data, dst.step);
    }
    else
    {
        fcvScaleDownMNu8((const uint8_t*)src.data, src.cols, src.rows, src.step, (uint8_t*)dst.data, dst.cols, dst.rows, dst.step);
    }

    if (status != FASTCV_SUCCESS)
    {
        std::string s = fcvStatusStrings.count(status) ? fcvStatusStrings.at(status) : "unknown";
        CV_Error(cv::Error::StsInternal, "FastCV error: " + s);
    }
}

} // fastcv::
} // cv::
