/*
 * Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
*/

#include "precomp.hpp"

namespace cv {
namespace fastcv {

class FcvHistogramLoop_Invoker : public cv::ParallelLoopBody
{
public:

    FcvHistogramLoop_Invoker(const uchar * src_data_, size_t src_step_, int width_, int height_, int32_t* gl_hist_, int stripeHeight_, cv::Mutex* histogramLock, int nStripes_):
        cv::ParallelLoopBody(), src_data(src_data_), src_step(src_step_), width(width_), height(height_), gl_hist(gl_hist_), stripeHeight(stripeHeight_), histogramLock_(histogramLock), nStripes(nStripes_)
    {
    }

    virtual void operator()(const cv::Range& range) const CV_OVERRIDE
    {
        int height_ = stripeHeight;
        if(range.end == nStripes)
           height_ += (height % nStripes);
        const uchar* yS = src_data;
        int32_t l_hist[256] = {0};
        fcvImageIntensityHistogram(yS, src_step, 0, range.start, width, height_, l_hist);
        cv::AutoLock lock(*histogramLock_);

        for( int i = 0; i < 256; i++ )
            gl_hist[i] += l_hist[i];
    }

private:
    const uchar * src_data;
    const size_t src_step;
    const int width;
    const int height;
    int32_t *gl_hist;
    int ret;
    int stripeHeight;
    cv::Mutex* histogramLock_;
    int nStripes;

    FcvHistogramLoop_Invoker(const FcvHistogramLoop_Invoker &);  // = delete;
    const FcvHistogramLoop_Invoker& operator= (const FcvHistogramLoop_Invoker &);  // = delete;
};

void calcHist( InputArray _src, OutputArray _hist )
{
    INITIALIZATION_CHECK;

    CV_Assert(!_src.empty());
    int type = _src.type();
    CV_Assert(type == CV_8UC1);

    _hist.create( cv::Size(256, 1), CV_32SC1 );
    Mat src = _src.getMat();
    Mat hist = _hist.getMat();

    for( int i = 0; i < 256; i++ )
       hist.ptr<int>()[i] = 0;

    cv::Mutex histogramLockInstance;

    int nStripes = cv::getNumThreads();
    int stripeHeight = src.rows / nStripes;

    cv::parallel_for_(cv::Range(0, nStripes),
              FcvHistogramLoop_Invoker(src.data, src.step[0], src.cols, src.rows, hist.ptr<int>(), stripeHeight, &histogramLockInstance, nStripes), nStripes);
}

} // fastcv::
} // cv::
