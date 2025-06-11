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
#include <atomic>

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
    struct FastCvDspContext;

    #define IS_FASTCV_ALLOCATED(mat) \
    ((mat.allocator == cv::fastcv::getQcAllocator()) ? true : \
        (CV_Error(cv::Error::StsBadArg, cv::format("Matrix '%s' not allocated with FastCV allocator. " \
                                    "Please ensure that the matrix is created using " \
                                    "cv::fastcv::getQcAllocator().", #mat)), false))
    
    #define FASTCV_CHECK_DSP_INIT() \
    if (!FastCvDspContext::getContext().isInitialized() && \
        fcvdspinit() != 0) \
    { \
        CV_Error(cv::Error::StsError, "Failed to initialize DSP"); \
    }
                                
    struct FastCvDspContext
    {
    private:
        mutable cv::Mutex initMutex;
        std::atomic<bool> isDspInitialized{false};
        std::atomic<uint64_t> initializationCount{0};
        std::atomic<uint64_t> deInitializationCount{0};

        static FastCvDspContext& getInstanceImpl() {
            static FastCvDspContext context;
            return context;
        }
    public:
        static FastCvDspContext& getContext() {
            return getInstanceImpl();
        }

        FastCvDspContext(const FastCvDspContext&) = delete;
        FastCvDspContext& operator=(const FastCvDspContext&) = delete;

        bool initialize() {
            cv::AutoLock lock(initMutex);
            
            if (isDspInitialized.load(std::memory_order_acquire)) {
                CV_LOG_INFO(NULL, "FastCV DSP already initialized, skipping initialization");
                return true;
            }

            CV_LOG_INFO(NULL, "Initializing FastCV DSP");

            if (fcvQ6Init() == 0) {
                isDspInitialized.store(true, std::memory_order_release);
                initializationCount++;
                CV_LOG_DEBUG(NULL, cv::format("FastCV DSP initialized (init count: %lu, deinit count: %lu)", 
                initializationCount.load(), deInitializationCount.load()));

                return true;
            }
    
            CV_LOG_ERROR(NULL, "FastCV DSP initialization failed");
            return false;
        }

        bool deinitialize() {
            cv::AutoLock lock(initMutex);
            
            if (!isDspInitialized.load(std::memory_order_acquire)) {
                CV_LOG_DEBUG(NULL, "FastCV DSP already deinitialized, skipping deinitialization");
                return true;
            }

            CV_LOG_INFO(NULL, "Deinitializing FastCV DSP");
            
            try {
                fcvQ6DeInit();
                isDspInitialized.store(false, std::memory_order_release);
                deInitializationCount++;
                CV_LOG_DEBUG(NULL, cv::format("FastCV DSP deinitialized (init count: %lu, deinit count: %lu)", 
                    initializationCount.load(), deInitializationCount.load()));
         
                return true;
            }
            catch (...) {
                CV_LOG_ERROR(NULL, "Exception occurred during FastCV DSP deinitialization");
                return false;
            }
        }

        bool isInitialized() const {
            return isDspInitialized.load(std::memory_order_acquire);
        }

        uint64_t getDspInitCount() const {
            return initializationCount.load(std::memory_order_acquire);
        }

        uint64_t getDspDeInitCount() const {
            return deInitializationCount.load(std::memory_order_acquire);
        }

        const cv::Mutex& getInitMutex() const {
            return initMutex;
        }
    
    private:
        FastCvDspContext() = default;
};

} // namespace dsp
} // namespace fastcv
} // namespace cv

#endif // OPENCV_FASTCV_PRECOMP_HPP
