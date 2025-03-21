/*
 * Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
*/

#include "precomp.hpp"

namespace cv {
namespace fastcv {

struct FastCVAllocator: public cv::MatAllocator
{
public:
    FastCVAllocator();
    ~FastCVAllocator();

    cv::UMatData* allocate(int dims, const int* sizes, int type, void* data, size_t* step, cv::AccessFlag flags, cv::UMatUsageFlags usageFlags) const CV_OVERRIDE;
    bool allocate(cv::UMatData* u, cv::AccessFlag accessFlags, cv::UMatUsageFlags usageFlags) const CV_OVERRIDE;
    void deallocate(cv::UMatData* u) const CV_OVERRIDE;
};

FastCVAllocator::FastCVAllocator()
{
}

FastCVAllocator::~FastCVAllocator()
{
}

cv::UMatData* FastCVAllocator::allocate(int dims, const int* sizes, int type,
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
    
    u->userdata = new std::string("QCOM");
    return u;
}

bool FastCVAllocator::allocate(cv::UMatData* u, cv::AccessFlag accessFlags, cv::UMatUsageFlags usageFlags) const
{
    CV_UNUSED(accessFlags);
    CV_UNUSED(usageFlags);

    return u != nullptr;
}

void FastCVAllocator::deallocate(cv::UMatData* u) const
{
    if(!u)
        return;

    CV_Assert(u->urefcount == 0);
    CV_Assert(u->refcount == 0);
    if( !(u->flags & cv::UMatData::USER_ALLOCATED) )
    {
        fcvHwMemFree(u->origdata);
        u->origdata = 0;
    }

    if (u->userdata)
    {
        delete static_cast<std::string*>(u->userdata);
        u->userdata = nullptr;
    }

    delete u;
}

cv::MatAllocator* getDefaultFastCVAllocator()
{
    static cv::MatAllocator* allocator = new FastCVAllocator;
    return allocator;
}

}
}
