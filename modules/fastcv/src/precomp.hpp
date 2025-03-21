/*
 * Copyright (c) 2025 Qualcomm Innovation Center, Inc. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
*/

#ifndef OPENCV_FASTCV_PRECOMP_HPP
#define OPENCV_FASTCV_PRECOMP_HPP

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include "opencv2/core/private.hpp"
#include "opencv2/core/utils/logger.hpp"
#include <opencv2/core/core_c.h>
#include <opencv2/fastcv.hpp>
#include <map>

#include "fastcv.h"
#include "fastcvDsp.h"

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
#define MIN_REMOTE_BUF_SIZE 176*144*sizeof(uint8_t)

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

namespace dsp {

    #define IS_FASTCV_ALLOCATED(mat) \
    ((mat.u && mat.u->userdata && \
      *static_cast<std::string*>(mat.u->userdata) == "QCOM") ? true : \
     (std::cerr << "Allocation check failed for " #mat \
                << ". Please ensure that cv::fastcv::dsp::fastcvq6init() has been called." \
                << std::endl, false))
    
    struct FastCvDspContext
    {
    public:
        // Initialize at first call
        // Defines a static local variable context.
        static FastCvDspContext& getContext()
        {
            //Instance is created only once.
            static FastCvDspContext context;
            return context;
        }

        //Constructor is called when the FastCvDspContext instance is created
        FastCvDspContext()
        {
            if (fcvQ6Init() != 0)
            {
                CV_LOG_WARNING(NULL, "Failed to switch FastCV DSP operation mode");
                isDspInitialized = false;
            }
            else
            {
                CV_LOG_INFO(NULL, "FastCV DSP Operation Mode Switched");
                isDspInitialized = true;
            }
        }

        bool isDspInitialized;
    };

} // namespace dsp
} // namespace fastcv
} // namespace cv

#endif // OPENCV_FASTCV_PRECOMP_HPP
