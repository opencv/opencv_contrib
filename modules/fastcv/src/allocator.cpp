/*
 * Copyright (c) 2025 Qualcomm Innovation Center, Inc. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
*/

#include "precomp.hpp"

namespace cv {
namespace fastcv {

QcResourceManager& QcResourceManager::getInstance() {
    static QcResourceManager instance;
    return instance;
}

void QcResourceManager::addAllocation(void* ptr) {
    std::lock_guard<std::mutex> lock(resourceMutex);
    activeAllocations.insert(ptr);
    CV_LOG_DEBUG(NULL, cv::format("Active Allocations: %zu", activeAllocations.size()));
}

void QcResourceManager::removeAllocation(void* ptr) {
    std::lock_guard<std::mutex> lock(resourceMutex);
    activeAllocations.erase(ptr);
    CV_LOG_DEBUG(NULL, cv::format("Active Allocations: %zu", activeAllocations.size()));
}

QcAllocator::QcAllocator()
{
}

QcAllocator::~QcAllocator()
{
}

cv::UMatData* QcAllocator::allocate(int dims, const int* sizes, int type,
                    void* data0, size_t* step, cv::AccessFlag flags,
                    cv::UMatUsageFlags usageFlags) const
{
    CV_UNUSED(flags);
    CV_UNUSED(usageFlags);

    size_t total = CV_ELEM_SIZE(type);
    for( int i = dims-1; i >= 0; i-- )
    {
        if( step )
        {
            if( data0 && step[i] != CV_AUTOSTEP )
            {
                CV_Assert(total <= step[i]);
                total = step[i];
            }
            else
                step[i] = total;
        }
        total *= sizes[i];
    }
    uchar* data = data0 ? (uchar*)data0 : (uchar*)fcvHwMemAlloc(total, 16);
    cv::UMatData* u = new cv::UMatData(this);
    u->data = u->origdata = data;
    u->size = total;
    if(data0)
        u->flags |= cv::UMatData::USER_ALLOCATED;

    // Add to active allocations
    cv::fastcv::QcResourceManager::getInstance().addAllocation(data);

    return u;
}

bool QcAllocator::allocate(cv::UMatData* u, cv::AccessFlag accessFlags, cv::UMatUsageFlags usageFlags) const
{
    CV_UNUSED(accessFlags);
    CV_UNUSED(usageFlags);

    return u != nullptr;
}

void QcAllocator::deallocate(cv::UMatData* u) const
{
    if(!u)
        return;

    CV_Assert(u->urefcount == 0);
    CV_Assert(u->refcount == 0);
    if( !(u->flags & cv::UMatData::USER_ALLOCATED) )
    {
        fcvHwMemFree(u->origdata);

        // Remove from active allocations
        cv::fastcv::QcResourceManager::getInstance().removeAllocation(u->origdata);
        u->origdata = 0;
    }

    delete u;
}

cv::MatAllocator* getQcAllocator()
{
    static cv::MatAllocator* allocator = new QcAllocator;
    return allocator;
}

}
}
