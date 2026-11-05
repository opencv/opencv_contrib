/*M///////////////////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not
// use this file except in compliance with the License. You may obtain a copy of
// the License at http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
// WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
// License for the specific language governing permissions and limitations under
// the License.
//
//M*/

// Direct HIP kernels replacing the NPP histogram entry points cudaimgproc's
// histogram.cpp relies on (NPP has no ROCm analogue): histEven for 16U/16S and
// the 4-channel overload, plus histRange (single- and 4-channel, all depths).
//
// NPP semantics reproduced exactly. A histogram over nLevels boundaries
// pLevels[0..nLevels-1] has nLevels-1 bins; a sample v is counted in bin i when
// pLevels[i] <= v < pLevels[i+1] (the last boundary is exclusive). histEven is
// the special case where the boundaries are nppiEvenLevelsHost-spaced, which
// histogram.cpp materialises via the same cvRound layout before calling the
// range kernel here, so even and range share one bin-edge definition.
//
// \author Jeff Daily <jeff.daily@amd.com>

#if defined(__HIP_PLATFORM_AMD__) && !defined(CUDA_DISABLER)

#include <hip/hip_runtime.h>
#include "opencv2/core/cuda/common.hpp"

namespace cv { namespace cuda { namespace device { namespace hist_hip
{
    // Largest bin to update by atomicAdd from shared memory.
    static const int MAX_SMEM_BINS = 1024;

    // Binary search in the level type L: index i with levels[i] <= v < levels[i+1],
    // else -1. The pixel is promoted to L so integer levels that exceed the pixel
    // type's range (e.g. 256 for CV_8U) compare correctly.
    template <typename L>
    __device__ __forceinline__ int findBin(const L* levels, int nLevels, L v)
    {
        if (v < levels[0] || v >= levels[nLevels - 1])
            return -1;
        int lo = 0;
        int hi = nLevels - 1; // number of bins
        while (hi - lo > 1)
        {
            const int mid = (lo + hi) >> 1;
            if (v < levels[mid])
                hi = mid;
            else
                lo = mid;
        }
        return lo;
    }

    // One channel of a CN-channel image (CN==1 for the single-channel case).
    template <typename T, typename L, int CN>
    __global__ void histRangeKernel(const T* src, size_t step, int rows, int cols,
                                    const L* levels, int nLevels, int channel, int* hist)
    {
        extern __shared__ int shist[];
        const int bins = nLevels - 1;
        for (int i = threadIdx.x + threadIdx.y * blockDim.x; i < bins; i += blockDim.x * blockDim.y)
            shist[i] = 0;
        __syncthreads();

        const int x = blockIdx.x * blockDim.x + threadIdx.x;
        const int y = blockIdx.y * blockDim.y + threadIdx.y;
        if (x < cols && y < rows)
        {
            const T* row = (const T*)((const char*)src + (size_t)y * step);
            const T v = row[x * CN + channel];
            const int bin = findBin<L>(levels, nLevels, (L)v);
            if (bin >= 0)
                ::atomicAdd(shist + bin, 1);
        }
        __syncthreads();

        for (int i = threadIdx.x + threadIdx.y * blockDim.x; i < bins; i += blockDim.x * blockDim.y)
        {
            const int hv = shist[i];
            if (hv > 0)
                ::atomicAdd(hist + i, hv);
        }
    }

    template <typename T, typename L, int CN>
    void launch(const PtrStepSzb& src, const L* levels, int nLevels, int channel, int* hist, hipStream_t stream)
    {
        const int bins = nLevels - 1;
        cudaSafeCall(hipMemsetAsync(hist, 0, bins * sizeof(int), stream));

        const dim3 block(32, 8);
        const dim3 grid(divUp(src.cols, block.x), divUp(src.rows, block.y));

        if (bins <= MAX_SMEM_BINS)
        {
            const size_t smem = bins * sizeof(int);
            histRangeKernel<T, L, CN><<<grid, block, smem, stream>>>(
                (const T*)src.data, src.step, src.rows, src.cols, levels, nLevels, channel, hist);
        }
        else
        {
            // Wide histograms: skip the shared-memory stage, hit global directly.
            histRangeKernel<T, L, CN><<<grid, block, 0, stream>>>(
                (const T*)src.data, src.step, src.rows, src.cols, levels, nLevels, channel, hist);
        }
        cudaSafeCall(hipGetLastError());
        if (stream == 0)
            cudaSafeCall(hipDeviceSynchronize());
    }

    // Single-channel range histogram. depth: CV_8U=0, CV_16U=2, CV_16S=3, CV_32F=5.
    // levels point to device memory of the matching level type (32S for integer
    // depths, 32F for CV_32F).
    void histRange_c1(PtrStepSzb src, int depth, const void* levels, int nLevels, int* hist, hipStream_t stream)
    {
        switch (depth)
        {
            case 0: launch<uchar,  int,   1>(src, (const int*)levels,   nLevels, 0, hist, stream); break;
            case 2: launch<ushort, int,   1>(src, (const int*)levels,   nLevels, 0, hist, stream); break;
            case 3: launch<short,  int,   1>(src, (const int*)levels,   nLevels, 0, hist, stream); break;
            default:launch<float,  float, 1>(src, (const float*)levels, nLevels, 0, hist, stream); break;
        }
    }

    // 4-channel range histogram: one independent histogram per channel.
    void histRange_c4(PtrStepSzb src, int depth, const void* const levels[4], const int nLevels[4], int* const hist[4], hipStream_t stream)
    {
        for (int c = 0; c < 4; ++c)
        {
            switch (depth)
            {
                case 0: launch<uchar,  int,   4>(src, (const int*)levels[c],   nLevels[c], c, hist[c], stream); break;
                case 2: launch<ushort, int,   4>(src, (const int*)levels[c],   nLevels[c], c, hist[c], stream); break;
                case 3: launch<short,  int,   4>(src, (const int*)levels[c],   nLevels[c], c, hist[c], stream); break;
                default:launch<float,  float, 4>(src, (const float*)levels[c], nLevels[c], c, hist[c], stream); break;
            }
        }
    }
}}}}

#endif // __HIP_PLATFORM_AMD__ && !CUDA_DISABLER
