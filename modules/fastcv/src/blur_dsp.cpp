/*
 * Copyright (c) 2025 Qualcomm Innovation Center, Inc. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
*/

#include "precomp.hpp"

namespace cv {
namespace fastcv {
namespace dsp {

class FcvFilter2DLoop_Invoker : public ParallelLoopBody
{
    public:

    FcvFilter2DLoop_Invoker(const Mat& _src, Mat& _dst, const Mat& _kernel) :
        ParallelLoopBody(), src(_src), dst(_dst), kernel(_kernel)
    {
        width     = src.cols;
        height    = src.rows;
        ksize     = kernel.size().width;
        halfKsize = ksize/2;
    }

    virtual void operator()(const Range& range) const CV_OVERRIDE
    {
        int topLines     = 0;
        int rangeHeight  = range.end-range.start;
        int paddedHeight = rangeHeight;

        if(range.start >= halfKsize)
        {
            topLines    += halfKsize;
            paddedHeight += halfKsize;
        }

        if(range.end <= height-halfKsize)
        {
            paddedHeight += halfKsize;
        }

        Mat srcPadded, dstPadded;
        srcPadded.allocator = cv::fastcv::getQcAllocator();
        dstPadded.allocator = cv::fastcv::getQcAllocator();
        
        srcPadded = src(Rect(0, range.start - topLines, width, paddedHeight)); 
        dstPadded.create(paddedHeight, width, dst.depth());
        
        CV_Assert(IS_FASTCV_ALLOCATED(srcPadded));
        CV_Assert(IS_FASTCV_ALLOCATED(dstPadded));

        if (dst.depth() == CV_8U)
            fcvFilterCorrNxNu8Q((int8_t*)kernel.data, ksize, 0, srcPadded.data, width, paddedHeight, srcPadded.step,
                dstPadded.data, dstPadded.step);
        else if (dst.depth() == CV_16S)
            fcvFilterCorrNxNu8s16Q((int8_t*)kernel.data, ksize, 0, srcPadded.data, width, paddedHeight, srcPadded.step,
                (int16_t*)dstPadded.data, dstPadded.step);
        else if (dst.depth() == CV_32F)
            fcvFilterCorrNxNu8f32Q((float32_t*)kernel.data, ksize, srcPadded.data, width, paddedHeight, srcPadded.step,
                (float32_t*)dstPadded.data, dstPadded.step);

        // Only copy center part back to output image and ignore the padded lines
        Mat temp1 = dstPadded(Rect(0, topLines, width, rangeHeight));
        Mat temp2 = dst(Rect(0, range.start, width, rangeHeight));
        temp1.copyTo(temp2);
    }

    private:
    const Mat&  src;
    Mat&        dst;
    const Mat&  kernel;
    int         width;
    int         height;
    int         ksize;
    int         halfKsize;

    FcvFilter2DLoop_Invoker(const FcvFilter2DLoop_Invoker &);  // = delete;
    const FcvFilter2DLoop_Invoker& operator= (const FcvFilter2DLoop_Invoker &);  // = delete;
};

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

    int nThreads = getNumThreads();
    int nStripes = (nThreads > 1) ? ((src.rows > 60) ? 3 * nThreads : 1) : 1;

    if (ddepth == CV_8U && ksize.width == 3)
        fcvFilterCorr3x3s8_v2Q((int8_t*)kernel.data, src.data, src.cols, src.rows, src.step, dst.data, dst.step);
    
    switch (ddepth)
    {
        case CV_8U:
        case CV_16S:
        {
            CV_Assert(CV_MAT_DEPTH(kernel.type()) == CV_8S);
            parallel_for_(Range(0, src.rows), FcvFilter2DLoop_Invoker(src, dst, kernel), nStripes);
            break;
        }
        case CV_32F:
        {
            CV_Assert(CV_MAT_DEPTH(kernel.type()) == CV_32F);
            parallel_for_(Range(0, src.rows), FcvFilter2DLoop_Invoker(src, dst, kernel), nStripes);
            break;
        }
        default:
        {
            CV_Error(cv::Error::StsBadArg, cv::format("Kernel Size:%d, Dst type:%s is not supported", ksize.width,
                depthToString(ddepth)));
            break;
        }
    }
}

} // dsp::
} // fastcv::
} // cv::