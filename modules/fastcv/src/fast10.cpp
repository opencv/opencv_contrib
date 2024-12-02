/*
 * Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
*/

#include "precomp.hpp"

namespace cv {
namespace fastcv {

void FAST10(InputArray _src, InputArray _mask, OutputArray _coords, OutputArray _scores, int barrier, int border, bool nmsEnabled)
{
    INITIALIZATION_CHECK;

    CV_Assert(!_src.empty() && _src.type() == CV_8UC1);
    CV_Assert(_src.cols() % 8 == 0);
    CV_Assert(_src.cols() <= 2048);
    CV_Assert(_src.step() % 8 == 0);

    // segfaults at border <= 3, fixing it
    border = std::max(4, border);

    CV_Assert(_src.cols() > 2*border);
    CV_Assert(_src.rows() > 2*border);

    Mat src = _src.getMat();

    Mat mask;
    if (!_mask.empty())
    {
        CV_Assert(_mask.type() == CV_8UC1);
        float kw = (float)src.cols  / (float)_mask.cols();
        float kh = (float)src.rows  / (float)_mask.rows();
        float eps = std::numeric_limits<float>::epsilon();
        if (std::abs(kw - kh) > eps)
        {
            CV_Error(cv::Error::StsBadArg, "Mask proportions do not correspond to image proportions");
        }
        bool sizeFits = false;
        for (int k = -3; k <= 3; k++)
        {
            if (std::abs(kw - std::pow(2.f, (float)k)) < eps)
            {
                sizeFits = true;
                break;
            }
        }
        if (!sizeFits)
        {
            CV_Error(cv::Error::StsBadArg, "Mask size do not correspond to image size divided by k from -3 to 3");
        }

        mask = _mask.getMat();
    }

    CV_Assert(_coords.needed());

    const int maxCorners = 32768;

    Mat coords(1, maxCorners * 2, CV_32SC1);

    AutoBuffer<uint32_t> tempBuf;
    Mat scores;
    if  (_scores.needed())
    {
        scores.create(1, maxCorners, CV_32SC1);

        tempBuf.allocate(maxCorners * 3 + src.rows + 1);
    }

    uint32_t nCorners = maxCorners;

    if (!mask.empty())
    {
        if (!scores.empty())
        {
            fcvCornerFast10InMaskScoreu8(src.data, src.cols, src.rows, src.step,
                                         barrier, border,
                                         (uint32_t*)coords.data, (uint32_t*)scores.data, maxCorners, &nCorners,
                                         mask.data, mask.cols, mask.rows,
                                         nmsEnabled,
                                         tempBuf.data());
        }
        else
        {
            fcvCornerFast10InMasku8(src.data, src.cols, src.rows, src.step,
                                    barrier, border,
                                    (uint32_t*)coords.data, maxCorners, &nCorners,
                                    mask.data, mask.cols, mask.rows);
        }
    }
    else
    {
        if (!scores.empty())
        {
            fcvCornerFast10Scoreu8(src.data, src.cols, src.rows, src.step,
                                   barrier, border,
                                   (uint32_t*)coords.data, (uint32_t*)scores.data, maxCorners, &nCorners,
                                   nmsEnabled,
                                   tempBuf.data());
        }
        else
        {
            fcvCornerFast10u8(src.data, src.cols, src.rows, src.step,
                                    barrier, border,
                                    (uint32_t*)coords.data, maxCorners, &nCorners);
        }
    }

    _coords.create(1, nCorners*2, CV_32SC1);
    coords(Range::all(), Range(0, nCorners*2)).copyTo(_coords);

    if (_scores.needed())
    {
        scores(Range::all(), Range(0, nCorners)).copyTo(_scores);
    }
}

} // fastcv::
} // cv::
