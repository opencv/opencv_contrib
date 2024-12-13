/*
 * Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
*/

#include "precomp.hpp"

namespace cv {
namespace fastcv {

class FcvWarpPerspectiveLoop_Invoker : public cv::ParallelLoopBody
{
    public:

    FcvWarpPerspectiveLoop_Invoker(InputArray _src1, InputArray _src2, OutputArray _dst1, OutputArray _dst2, InputArray _M0,
        Size _dsize) : cv::ParallelLoopBody()
    {
        src1 = _src1.getMat();
        src2 = _src2.getMat();
        dsize = _dsize;

        _dst1.create(dsize, src1.type());
        _dst2.create(dsize, src2.type());
        dst1 = _dst1.getMat();
        dst2 = _dst2.getMat();

        M = _M0.getMat();
    }

    virtual void operator()(const cv::Range& range) const CV_OVERRIDE
    {
        uchar* dst1_ptr = dst1.data + range.start*dst1.step;
        uchar* dst2_ptr = dst2.data + range.start*dst2.step;
        int rangeHeight = range.end - range.start;

        float rangeMatrix[9];
        rangeMatrix[0] = M.at<float>(0,0);
        rangeMatrix[1] = M.at<float>(0,1);
        rangeMatrix[2] = M.at<float>(0,2)+range.start*M.at<float>(0,1);
        rangeMatrix[3] = M.at<float>(1,0);
        rangeMatrix[4] = M.at<float>(1,1);
        rangeMatrix[5] = M.at<float>(1,2)+range.start*M.at<float>(1,1);
        rangeMatrix[6] = M.at<float>(2,0);
        rangeMatrix[7] = M.at<float>(2,1);
        rangeMatrix[8] = M.at<float>(2,2)+range.start*M.at<float>(2,1);

        fcv2PlaneWarpPerspectiveu8(src1.data, src2.data, src1.cols, src1.rows, src1.step, src2.step, dst1_ptr, dst2_ptr,
            dsize.width, rangeHeight, dst1.step, dst2.step, rangeMatrix);
    }

    private:
    Mat         src1;
    Mat         src2;
    Mat         dst1;
    Mat         dst2;
    Mat         M;
    Size        dsize;

    FcvWarpPerspectiveLoop_Invoker(const FcvWarpPerspectiveLoop_Invoker &);  // = delete;
    const FcvWarpPerspectiveLoop_Invoker& operator= (const FcvWarpPerspectiveLoop_Invoker &);  // = delete;
};

void warpPerspective2Plane(InputArray _src1, InputArray _src2, OutputArray _dst1, OutputArray _dst2, InputArray _M0,
        Size dsize)
{
    INITIALIZATION_CHECK;
    CV_Assert(!_src1.empty() && _src1.type() == CV_8UC1);
    CV_Assert(!_src2.empty() && _src2.type() == CV_8UC1);
    CV_Assert(!_M0.empty());

    cv::parallel_for_(cv::Range(0, dsize.height),
        FcvWarpPerspectiveLoop_Invoker(_src1, _src2, _dst1, _dst2, _M0, dsize), 1);
}

} // fastcv::
} // cv::