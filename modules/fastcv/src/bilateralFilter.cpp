/*
 * Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
*/

#include "precomp.hpp"

namespace cv {
namespace fastcv {

class FcvFilterLoop_Invoker : public cv::ParallelLoopBody
{
public:

    FcvFilterLoop_Invoker(cv::Mat src_, size_t src_step_, cv::Mat dst_, size_t dst_step_, int width_, int height_,
                          int bdr_, int knl_, float32_t sigma_color_, float32_t sigma_space_) :
        cv::ParallelLoopBody(), src_step(src_step_), dst_step(dst_step_), width(width_), height(height_),
        bdr(bdr_), knl(knl_), sigma_color(sigma_color_), sigma_space(sigma_space_), src(src_), dst(dst_)
    { }

    virtual void operator()(const cv::Range& range) const CV_OVERRIDE
    {
        int height_ = range.end - range.start;
        int width_  = width;
        cv::Mat src_;
        int n = knl/2;

        src_ = cv::Mat(height_ + 2 * n, width_ + 2 * n, CV_8U);
        if (range.start == 0 && range.end == height)
        {
            cv::copyMakeBorder(src(cv::Rect(0, 0, width, height)), src_, n, n, n, n, bdr);
        }
        else if (range.start == 0)
        {
            cv::copyMakeBorder(src(cv::Rect(0, 0, width_, height_ + n)), src_, n, 0, n, n, bdr);
        }
        else if (range.end == (height))
        {
            cv::copyMakeBorder(src(cv::Rect(0, range.start - n, width_, height_ + n)), src_, 0, n, n, n, bdr);
        }
        else
        {
            cv::copyMakeBorder(src(cv::Rect(0, range.start - n, width_, height_ + 2 * n)), src_, 0, 0, n, n, bdr);
        }

        cv::Mat dst_padded = cv::Mat(height_ + 2*n, width_ + 2*n, CV_8U);

        auto func = (knl == 5) ? fcvBilateralFilter5x5u8_v3 :
                    (knl == 7) ? fcvBilateralFilter7x7u8_v3 :
                    (knl == 9) ? fcvBilateralFilter9x9u8_v3 :
                    nullptr;
        func(src_.data, width_ + 2 * n, height_ + 2 * n, width_ + 2 * n,
             dst_padded.data, width_ + 2 * n, sigma_color, sigma_space, 0);

        cv::Mat dst_temp1 = dst_padded(cv::Rect(n, n, width_, height_));
        cv::Mat dst_temp2 = dst(cv::Rect(0, range.start, width_, height_));
        dst_temp1.copyTo(dst_temp2);
    }

private:
    const size_t src_step;
    const size_t dst_step;
    const int width;
    const int height;
    const int bdr;
    const int knl;
    float32_t sigma_color;
    float32_t sigma_space;
    int ret;
    cv::Mat src;
    cv::Mat dst;

    FcvFilterLoop_Invoker(const FcvFilterLoop_Invoker &);  // = delete;
    const FcvFilterLoop_Invoker& operator= (const FcvFilterLoop_Invoker &);  // = delete;
};

void bilateralFilter( InputArray _src, OutputArray _dst, int d,
                      float sigmaColor, float sigmaSpace,
                      int borderType )
{
    INITIALIZATION_CHECK;

    CV_Assert(!_src.empty());
    int type = _src.type();
    CV_Assert(type == CV_8UC1);
    CV_Assert(d == 5 || d == 7 || d == 9);

    Size size = _src.size();
    _dst.create( size, type );
    Mat src = _src.getMat();
    Mat dst = _dst.getMat();

    CV_Assert(src.data != dst.data);

    if( sigmaColor <= 0 )
        sigmaColor = 1;
    if( sigmaSpace <= 0 )
        sigmaSpace = 1;

    int nStripes = (src.rows / 20 == 0) ? 1 : (src.rows / 20);
    cv::parallel_for_(cv::Range(0, src.rows),
              FcvFilterLoop_Invoker(src, src.step, dst, dst.step, src.cols, src.rows, borderType, d, sigmaColor, sigmaSpace), nStripes);
}

}
}
