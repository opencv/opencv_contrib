/*
 * Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
*/

#include "precomp.hpp"

namespace cv {
namespace fastcv {

cv::Moments moments(InputArray _src, bool binary)
{
    INITIALIZATION_CHECK;

    CV_Assert(!_src.empty());
    int type = _src.type();
    CV_Assert(type == CV_8UC1 || type == CV_32SC1 || type == CV_32FC1);

    Size size = _src.size();
    Mat src = _src.getMat();

    cv::Moments m;
    fcvMoments mFCV;
    fcvStatus status = FASTCV_SUCCESS;
    if(binary)
    {
        cv::Mat src_binary(size, CV_8UC1);
        cv::compare( src, 0, src_binary, cv::CMP_NE );
        fcvImageMomentsu8(src_binary.data, src_binary.cols,
                          src_binary.rows, src_binary.step[0], &mFCV, binary);
    }
    else
    {
        switch(type)
        {
            case CV_8UC1:
                fcvImageMomentsu8(src.data, src.cols, src.rows, src.step[0], &mFCV, binary);
                break;
            case CV_32SC1:
                fcvImageMomentss32(src.ptr<int>(), src.cols, src.rows, src.step[0], &mFCV, binary);
                break;
            case CV_32FC1:
                fcvImageMomentsf32(src.ptr<float>(), src.cols, src.rows, src.step[0], &mFCV, binary);
                break;
        }
    }

    if (status != FASTCV_SUCCESS)
    {
        CV_Error( cv::Error::StsError, cv::format("Error occurred!") );
        return m;
    }

    m.m00  = mFCV.m00;  m.m10  = mFCV.m10;  m.m01  = mFCV.m01;
    m.m20  = mFCV.m20;  m.m11  = mFCV.m11;  m.m02  = mFCV.m02;
    m.m30  = mFCV.m30;  m.m21  = mFCV.m21;  m.m12  = mFCV.m12;
    m.m03  = mFCV.m03;  m.mu02 = mFCV.mu02; m.m03  = mFCV.mu03;
    m.mu11 = mFCV.mu11; m.mu12 = mFCV.mu12; m.mu20 = mFCV.mu20;
    m.mu21 = mFCV.mu21; m.mu30 = mFCV.mu30;

    float32_t inv_m00 = 1.0/mFCV.m00;
    float32_t inv_sqrt_m00 = mFCV.inv_sqrt_m00;
    float32_t s2 = inv_m00 * inv_m00, s3 = s2 * inv_sqrt_m00;

    m.nu20 = mFCV.mu20 * s2; m.nu11 = mFCV.mu11 * s2;
    m.nu02 = mFCV.mu02 * s2; m.nu30 = mFCV.mu30 * s3;
    m.nu21 = mFCV.mu21 * s3; m.nu12 = mFCV.mu12 * s3;
    m.nu03 = mFCV.mu03 * s3;

    return m;
}

} // fastcv::
} // cv::
