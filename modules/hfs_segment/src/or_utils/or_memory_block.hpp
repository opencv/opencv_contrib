/*M///////////////////////////////////////////////////////////////////////////////////////
// By downloading, copying, installing or using the software you agree to this license.
// If you do not agree to this license, do not download, install,
// copy or use the software.
//
//
//                              License Agreement
//                    For Open Source Computer Vision Library
//                           (3 - clause BSD License)
//
// Copyright(C) 2000 - 2016, Intel Corporation, all rights reserved.
// Copyright(C) 2009 - 2011, Willow Garage Inc., all rights reserved.
// Copyright(C) 2009 - 2016, NVIDIA Corporation, all rights reserved.
// Copyright(C) 2010 - 2013, Advanced Micro Devices, Inc., all rights reserved.
// Copyright(C) 2015 - 2016, OpenCV Foundation, all rights reserved.
// Copyright(C) 2015 - 2016, Itseez Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met :
//
//      * Redistributions of source code must retain the above copyright notice,
//        this list of conditions and the following disclaimer.
//
//      * Redistributions in binary form must reproduce the above copyright notice,
//        this list of conditions and the following disclaimer in the documentation
//        and / or other materials provided with the distribution.
//
//      * Neither the names of the copyright holders nor the names of the contributors
//        may be used to endorse or promote products derived from this software
//        without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall copyright holders or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort(including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//M*/

#ifndef _OPENCV_OR_MEMORY_BLOCK_HPP_
#define _OPENCV_OR_MEMORY_BLOCK_HPP_
#ifdef __cplusplus

#include "opencv2/core/cuda/common.hpp"

namespace cv { namespace hfs { namespace orutils {

template <typename T>
class MemoryBlock
{
protected:
    bool isAllocated_CPU, isAllocated_CUDA;
    T* data_cpu;

    T* data_cuda;

public:
    enum MemoryCopyDirection 
    { CPU_TO_CPU, CPU_TO_CUDA, CUDA_TO_CPU, CUDA_TO_CUDA };

    size_t dataSize;

    inline T* getGpuData()
    {
        return data_cuda;
    }
    inline T* getCpuData()
    {
        return data_cpu;
    }

    inline const T* getGpuData() const
    {
        return data_cuda;
    }

    inline const T* getCpuData() const
    {
        return data_cpu;
    }

    MemoryBlock(size_t dataSize, bool allocate_CPU, bool allocate_CUDA)
    {
        this->isAllocated_CPU = false;
        this->isAllocated_CUDA = false;

        Allocate(dataSize, allocate_CPU, allocate_CUDA);
        clear();
    }

    void clear(unsigned char defaultValue = 0)
    {
        if (isAllocated_CPU) 
            memset(data_cpu, defaultValue, dataSize * sizeof(T));
        if (isAllocated_CUDA) 
            cudaSafeCall(cudaMemset(data_cuda, 
                defaultValue, dataSize * sizeof(T)));
    }

    void updateDeviceFromHost()
    {
        if (isAllocated_CUDA && isAllocated_CPU)
            cudaSafeCall(cudaMemcpy(data_cuda, 
                data_cpu, dataSize * sizeof(T), cudaMemcpyHostToDevice));
    }
    void updateHostFromDevice()
    {
        if (isAllocated_CUDA && isAllocated_CPU)
            cudaSafeCall(cudaMemcpy(data_cpu, 
                data_cuda, dataSize * sizeof(T), cudaMemcpyDeviceToHost));
    }

    void setFrom(const MemoryBlock<T> *source, 
        MemoryCopyDirection memoryCopyDirection)
    {
        switch (memoryCopyDirection)
        {
        case CPU_TO_CPU:
            memcpy(this->data_cpu, source->data_cpu, 
                source->dataSize * sizeof(T));
            break;
        case CPU_TO_CUDA:
            cudaSafeCall(cudaMemcpyAsync(this->data_cuda, source->data_cpu, 
                source->dataSize * sizeof(T), cudaMemcpyHostToDevice));
            break;
        case CUDA_TO_CPU:
            cudaSafeCall(cudaMemcpy(this->data_cpu, source->data_cuda, 
                source->dataSize * sizeof(T), cudaMemcpyDeviceToHost));
            break;
        case CUDA_TO_CUDA:
            cudaSafeCall(cudaMemcpyAsync(this->data_cuda, source->data_cuda, 
                source->dataSize * sizeof(T), cudaMemcpyDeviceToDevice));
            break;
        default: break;
        }
    }

    virtual ~MemoryBlock() { this->Free(); }

    void Allocate(size_t dataSize, bool allocate_CPU, bool allocate_CUDA)
    {
        Free();

        this->dataSize = dataSize;

        if (allocate_CPU)
        {
            int allocType = 0;

            if (allocate_CUDA) allocType = 1;
            switch (allocType)
            {
            case 0:
                data_cpu = new T[dataSize];
                break;
            case 1:
                cudaSafeCall(cudaMallocHost((void**)&data_cpu, 
                    dataSize * sizeof(T)));
                break;
            }
            this->isAllocated_CPU = allocate_CPU;
        }

        if (allocate_CUDA)
        {
            cudaSafeCall(cudaMalloc((void**)&data_cuda, dataSize * sizeof(T)));
            this->isAllocated_CUDA = allocate_CUDA;
        }
    }

    void Free()
    {
        if (isAllocated_CPU)
        {
            int allocType = 0;
            if (isAllocated_CUDA) allocType = 1;
            switch (allocType)
            {
            case 0:
                delete[] data_cpu;
                break;
            case 1:
                cudaSafeCall(cudaFreeHost(data_cpu));
                break;
            }
            isAllocated_CPU = false;
        }

        if (isAllocated_CUDA)
        {
            cudaSafeCall(cudaFree(data_cuda));
            isAllocated_CUDA = false;
        }
    }

    MemoryBlock(const MemoryBlock&);
    MemoryBlock& operator=(const MemoryBlock&);
};


}}} 

#endif
#endif
