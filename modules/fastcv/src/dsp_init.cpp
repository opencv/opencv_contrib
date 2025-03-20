/*
 * Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
*/

#include "precomp.hpp"

namespace cv {
namespace fastcv {
namespace dsp {

int fastcvq6init()
{
    // Get the DSP context
    FastCvDspContext& context = FastCvDspContext::getContext();

    // Check if DSP is initialized
    if (!context.isDspInitialized)
    {
        CV_Error(cv::Error::StsBadArg, cv::format("DSP initialization failed!"));
        return 1;
    }

    // Get custom allocator
    cv::MatAllocator* allocator = cv::fastcv::getDefaultFastCVAllocator();

    // Ensure the allocator is not the standard one
    CV_Assert(allocator != cv::Mat::getStdAllocator());

    // Check if the current allocator is already set to the custom allocator
    if (cv::Mat::getDefaultAllocator() != allocator)
    {
        // Set the custom allocator
        cv::Mat::setDefaultAllocator(allocator);
    }
    
    return 0;
}

void fastcvq6deinit()
{
    // Deinitialize the DSP environment
    fcvQ6DeInit();
}


} // namespace dsp
} // namespace fastcv
} // namespace cv