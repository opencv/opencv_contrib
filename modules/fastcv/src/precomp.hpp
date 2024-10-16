/*
 * Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
*/

#ifndef OPENCV_FASTCV_PRECOMP_HPP
#define OPENCV_FASTCV_PRECOMP_HPP

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include "opencv2/core/private.hpp"
#include "opencv2/core/utils/logger.hpp"

#include <opencv2/arithm.hpp>
#include <opencv2/cluster.hpp>
#include <opencv2/draw.hpp>
#include <opencv2/fast10.hpp>
#include <opencv2/fft.hpp>
#include <opencv2/hough.hpp>
#include <opencv2/moments.hpp>
#include <opencv2/mser.hpp>
#include <opencv2/remap.hpp>
#include <opencv2/scale.hpp>
#include <opencv2/shift.hpp>
#include <opencv2/smooth.hpp>
#include <opencv2/thresh.hpp>
#include <opencv2/bilateralFilter.hpp>

#include <map>

#include "fastcv.h"

namespace cv {
namespace fastcv {

extern bool isInitialized;

#define INITIALIZATION_CHECK                                                \
{                                                                           \
    if(!isInitialized)                                                      \
    {                                                                       \
        if (fcvSetOperationMode(FASTCV_OP_CPU_PERFORMANCE) != 0)            \
            CV_Error(cv::Error::StsBadArg, cv::format("Set mode failed!")); \
        else                                                                \
            isInitialized = true;                                           \
    }                                                                       \
    CV_INSTRUMENT_REGION();                                                 \
}

const std::map<fcvStatus, std::string> fcvStatusStrings =
{
    { FASTCV_SUCCESS,       "Success"},
    { FASTCV_EFAIL,         "General failure"},
    { FASTCV_EUNALIGNPARAM, "Unaligned pointer parameter"},
    { FASTCV_EBADPARAM,     "Bad parameters"},
    { FASTCV_EINVALSTATE,   "Called at invalid state"},
    { FASTCV_ENORES,        "Insufficient resources, memory, thread"},
    { FASTCV_EUNSUPPORTED,  "Unsupported feature"},
    { FASTCV_EHWQDSP,       "Hardware QDSP failed to respond"},
    { FASTCV_EHWGPU,        "Hardware GPU failed to respond"},
};

} // namespace fastcv
} // namespace cv

#endif // OPENCV_FASTCV_PRECOMP_HPP
