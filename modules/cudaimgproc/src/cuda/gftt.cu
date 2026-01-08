/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#if !defined CUDA_DISABLER

#include <thrust/device_ptr.h>
#include <thrust/sort.h>

#include "opencv2/core/cuda/utility.hpp"
#include <opencv2/cudev/ptr2d/texture.hpp>
#include <thrust/execution_policy.h>
namespace cv { namespace cuda { namespace device
{
    namespace gfft
    {
        template <class Mask> __global__ void findCorners(cv::cudev::TexturePtr<float> tex, float threshold, const Mask mask, float2* corners, int max_count, int rows, int cols, int *g_counter)
        {
            const int j = blockIdx.x * blockDim.x + threadIdx.x;
            const int i = blockIdx.y * blockDim.y + threadIdx.y;

            if (i > 0 && i < rows - 1 && j > 0 && j < cols - 1 && mask(i, j))
            {
                float val = tex(i, j);

                if (val > threshold)
                {
                    float maxVal = val;

                    maxVal = ::fmax(tex(i - 1, j - 1), maxVal);
                    maxVal = ::fmax(tex(i - 1, j), maxVal);
                    maxVal = ::fmax(tex(i - 1, j + 1), maxVal);

                    maxVal = ::fmax(tex(i, j - 1), maxVal);
                    maxVal = ::fmax(tex(i, j + 1), maxVal);

                    maxVal = ::fmax(tex(i + 1, j - 1), maxVal);
                    maxVal = ::fmax(tex(i + 1, j), maxVal);
                    maxVal = ::fmax(tex(i + 1, j + 1), maxVal);

                    if (val == maxVal)
                    {
                        const int ind = ::atomicAdd(g_counter, 1);

                        if (ind < max_count)
                            corners[ind] = make_float2(j, i);
                    }
                }
            }
        }

        int findCorners_gpu(const PtrStepSzf eig, float threshold, PtrStepSzb mask, float2* corners, int max_count, int* counterPtr, cudaStream_t stream)
        {
            cudaSafeCall( cudaMemsetAsync(counterPtr, 0, sizeof(int), stream) );
            cv::cudev::Texture<float> tex(eig);

            dim3 block(16, 16);
            dim3 grid(divUp(eig.cols, block.x), divUp(eig.rows, block.y));

            if (mask.data)
                findCorners<<<grid, block, 0, stream>>>(tex, threshold, SingleMask(mask), corners, max_count, eig.rows, eig.cols, counterPtr);
            else
                findCorners<<<grid, block, 0, stream>>>(tex, threshold, WithOutMask(), corners, max_count, eig.rows, eig.cols, counterPtr);

            cudaSafeCall( cudaGetLastError() );

            int count;
            cudaSafeCall( cudaMemcpyAsync(&count, counterPtr, sizeof(int), cudaMemcpyDeviceToHost, stream) );
            if (stream)
                cudaSafeCall(cudaStreamSynchronize(stream));
            else
                cudaSafeCall( cudaDeviceSynchronize() );
            return std::min(count, max_count);
        }

        class EigGreater
        {
        public:
            EigGreater(cv::cudev::TexturePtr<float> tex_) : tex(tex_) {}
            __device__ __forceinline__ bool operator()(float2 a, float2 b) const{
                return tex(a.y, a.x) > tex(b.y, b.x);
            }
            cv::cudev::TexturePtr<float> tex;
        };

        void sortCorners_gpu(const PtrStepSzf eig, float2* corners, int count, cudaStream_t stream)
        {
            cv::cudev::Texture<float> tex(eig);
            thrust::device_ptr<float2> ptr(corners);
#if THRUST_VERSION >= 100802
            if (stream)
                thrust::sort(thrust::cuda::par(ThrustAllocator::getAllocator()).on(stream), ptr, ptr + count, EigGreater(tex));
            else
                thrust::sort(thrust::cuda::par(ThrustAllocator::getAllocator()), ptr, ptr + count, EigGreater(tex));
#else
            thrust::sort(ptr, ptr + count, EigGreater(tex));
#endif
        }
    } // namespace optical_flow
}}}


#endif /* CUDA_DISABLER */
