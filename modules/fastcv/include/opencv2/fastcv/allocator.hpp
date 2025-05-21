/*
 * Copyright (c) 2025 Qualcomm Innovation Center, Inc. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
*/

#ifndef OPENCV_FASTCV_ALLOCATOR_HPP
#define OPENCV_FASTCV_ALLOCATOR_HPP

#include <opencv2/core.hpp>
#include <set>
#include <mutex>

namespace cv {
namespace fastcv {

//! @addtogroup fastcv
//! @{

/**
 * @brief Resource manager for FastCV allocations.
 * This class manages active allocations.
 */
class QcResourceManager {
public:
    static QcResourceManager& getInstance();

    void addAllocation(void* ptr);
    void removeAllocation(void* ptr);

private:
    QcResourceManager() = default;
    std::set<void*> activeAllocations;
    std::mutex resourceMutex;
};

/**
 * @brief Qualcomm's custom allocator.
 * This allocator uses Qualcomm's memory management functions.
 */
class QcAllocator : public cv::MatAllocator {
    public:
        QcAllocator();
        ~QcAllocator();
    
        cv::UMatData* allocate(int dims, const int* sizes, int type, void* data0, size_t* step, cv::AccessFlag flags, cv::UMatUsageFlags usageFlags) const CV_OVERRIDE;
        bool allocate(cv::UMatData* u, cv::AccessFlag accessFlags, cv::UMatUsageFlags usageFlags) const CV_OVERRIDE;
        void deallocate(cv::UMatData* u) const CV_OVERRIDE;
};

/**
 * @brief Gets the default Qualcomm's allocator.
 * This function returns a pointer to the default Qualcomm's allocator, which is optimized
 * for use with DSP.
 *
 * @return Pointer to the default FastCV allocator.
 */
CV_EXPORTS cv::MatAllocator* getQcAllocator();

//! @}

} // namespace fastcv
} // namespace cv

#endif // OPENCV_FASTCV_ALLOCATOR_HPP
