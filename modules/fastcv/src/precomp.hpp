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

#include <opencv2/fastcv.hpp>
#include <map>

#include "fastcv.h"

namespace cv {
namespace fastcv {

#define INITIALIZATION_CHECK                                                \
{                                                                           \
    if (!FastCvContext::getContext().isInitialized)                         \
    {                                                                       \
        CV_Error(cv::Error::StsBadArg, cv::format("Set mode failed!"));     \
    }                                                                       \
    CV_INSTRUMENT_REGION();                                                 \
}

#define FCV_KernelSize_SHIFT 3
#define FCV_MAKETYPE(ksize,depth) ((ksize<<FCV_KernelSize_SHIFT) + depth)

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

struct FastCvContext
{
public:
    // initialize at first call
    // Defines a static local variable context. Variable is created only once.
    static FastCvContext& getContext()
    {
        static FastCvContext context;
        return context;
    }

    FastCvContext()
    {
        if (fcvSetOperationMode(FASTCV_OP_CPU_PERFORMANCE) != 0)
        {
            CV_LOG_WARNING(NULL, "Failed to switch FastCV operation mode");
            isInitialized = false;
        }
        else
        {
            CV_LOG_INFO(NULL, "FastCV Operation Mode Switched");
            isInitialized = true;
        }
    }

    bool isInitialized;
};

} // namespace fastcv
} // namespace cv

#endif // OPENCV_FASTCV_PRECOMP_HPP
