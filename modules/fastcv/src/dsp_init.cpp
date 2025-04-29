/*
 * Copyright (c) 2025 Qualcomm Innovation Center, Inc. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
*/

#include "precomp.hpp"

namespace cv {
namespace fastcv {
namespace dsp {
//CHANGE FASTCV Q6 INIT
int fcvdspinit()
{
    FastCvDspContext& context = FastCvDspContext::getContext();
    
    if (context.isInitialized()) {
        CV_LOG_INFO(NULL, "FastCV DSP already initialized, skipping initialization");
        return 0;
    }
    if (!context.initialize()) {
        CV_LOG_ERROR(NULL, "Failed to initialize FastCV DSP");
        return -1;
    }
    CV_LOG_INFO(NULL, "FastCV DSP initialized successfully");
    return 0;
}

void fcvdspdeinit()
{
    // Deinitialize the DSP environment
    FastCvDspContext& context = FastCvDspContext::getContext();
    
    if (!context.isInitialized()) {
        CV_LOG_INFO(NULL, "FastCV DSP already deinitialized, skipping deinitialization");
        return;
    }
    if (!context.deinitialize()) {
        CV_LOG_ERROR(NULL, "Failed to deinitialize FastCV DSP");
    }
    CV_LOG_INFO(NULL, "FastCV DSP deinitialized successfully");
}


} // namespace dsp
} // namespace fastcv
} // namespace cv