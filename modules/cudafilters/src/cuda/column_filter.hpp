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

#include "opencv2/core/cuda.hpp"
#include "opencv2/core/cuda/common.hpp"
#include "opencv2/core/cuda/saturate_cast.hpp"
#include "opencv2/core/cuda/vec_math.hpp"
#include "opencv2/core/cuda/border_interpolate.hpp"

using namespace cv::cuda;
using namespace cv::cuda::device;

namespace column_filter
{
    #define MAX_KERNEL_SIZE 32

    template <int KSIZE, typename T, typename D, typename B>
    __global__ void linearColumnFilter(const PtrStepSz<T> src, PtrStep<D> dst, const float* kernel, const int anchor, const B brd)
    {
        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 200)
            const int BLOCK_DIM_X = 16;
            const int BLOCK_DIM_Y = 16;
            const int PATCH_PER_BLOCK = 4;
            const int HALO_SIZE = KSIZE <= 16 ? 1 : 2;
        #else
            const int BLOCK_DIM_X = 16;
            const int BLOCK_DIM_Y = 8;
            const int PATCH_PER_BLOCK = 2;
            const int HALO_SIZE = 2;
        #endif

        typedef typename TypeVec<float, VecTraits<T>::cn>::vec_type sum_t;

        __shared__ sum_t smem[(PATCH_PER_BLOCK + 2 * HALO_SIZE) * BLOCK_DIM_Y][BLOCK_DIM_X];

        // Use Y grid dimension to index the columns of the image
        const int x = blockIdx.y * BLOCK_DIM_X + threadIdx.x;

        if (x >= src.cols)
            return;

        const T* src_col = src.ptr() + x;

        // Use X grid dimension to index the rows of the image
        const int yStart = blockIdx.x * (BLOCK_DIM_Y * PATCH_PER_BLOCK) + threadIdx.y;

        if (blockIdx.x > 0)
        {
            //Upper halo
            #pragma unroll
            for (int j = 0; j < HALO_SIZE; ++j)
                smem[threadIdx.y + j * BLOCK_DIM_Y][threadIdx.x] = saturate_cast<sum_t>(src(yStart - (HALO_SIZE - j) * BLOCK_DIM_Y, x));
        }
        else
        {
            //Upper halo
            #pragma unroll
            for (int j = 0; j < HALO_SIZE; ++j)
                smem[threadIdx.y + j * BLOCK_DIM_Y][threadIdx.x] = saturate_cast<sum_t>(brd.at_low(yStart - (HALO_SIZE - j) * BLOCK_DIM_Y, src_col, src.step));
        }

        if (blockIdx.x + 2 < gridDim.x)
        {
            //Main data
            #pragma unroll
            for (int j = 0; j < PATCH_PER_BLOCK; ++j)
                smem[threadIdx.y + HALO_SIZE * BLOCK_DIM_Y + j * BLOCK_DIM_Y][threadIdx.x] = saturate_cast<sum_t>(src(yStart + j * BLOCK_DIM_Y, x));

            //Lower halo
            #pragma unroll
            for (int j = 0; j < HALO_SIZE; ++j)
                smem[threadIdx.y + (PATCH_PER_BLOCK + HALO_SIZE) * BLOCK_DIM_Y + j * BLOCK_DIM_Y][threadIdx.x] = saturate_cast<sum_t>(src(yStart + (PATCH_PER_BLOCK + j) * BLOCK_DIM_Y, x));
        }
        else
        {
            //Main data
            #pragma unroll
            for (int j = 0; j < PATCH_PER_BLOCK; ++j)
                smem[threadIdx.y + HALO_SIZE * BLOCK_DIM_Y + j * BLOCK_DIM_Y][threadIdx.x] = saturate_cast<sum_t>(brd.at_high(yStart + j * BLOCK_DIM_Y, src_col, src.step));

            //Lower halo
            #pragma unroll
            for (int j = 0; j < HALO_SIZE; ++j)
                smem[threadIdx.y + (PATCH_PER_BLOCK + HALO_SIZE) * BLOCK_DIM_Y + j * BLOCK_DIM_Y][threadIdx.x] = saturate_cast<sum_t>(brd.at_high(yStart + (PATCH_PER_BLOCK + j) * BLOCK_DIM_Y, src_col, src.step));
        }

        __syncthreads();

        #pragma unroll
        for (int j = 0; j < PATCH_PER_BLOCK; ++j)
        {
            const int y = yStart + j * BLOCK_DIM_Y;

            if (y < src.rows)
            {
                sum_t sum = VecTraits<sum_t>::all(0);

                #pragma unroll
                for (int k = 0; k < KSIZE; ++k)
                    sum = sum + smem[threadIdx.y + HALO_SIZE * BLOCK_DIM_Y + j * BLOCK_DIM_Y - anchor + k][threadIdx.x] * kernel[k];

                dst(y, x) = saturate_cast<D>(sum);
            }
        }
    }

    template <int KSIZE, typename T, typename D, template<typename> class B>
    void caller(PtrStepSz<T> src, PtrStepSz<D> dst, const float* kernel, int anchor, int cc, cudaStream_t stream)
    {
        int BLOCK_DIM_X;
        int BLOCK_DIM_Y;
        int PATCH_PER_BLOCK;

        if (cc >= 20)
        {
            BLOCK_DIM_X = 16;
            BLOCK_DIM_Y = 16;
            PATCH_PER_BLOCK = 4;
        }
        else
        {
            BLOCK_DIM_X = 16;
            BLOCK_DIM_Y = 8;
            PATCH_PER_BLOCK = 2;
        }

        // Process the image in chunks of rows such that the number of blocks in y-direction does not exceed the maximum grid size
        DeviceInfo devInfo;
        const int maxGridY = devInfo.maxGridSize()[1];
        const int nBlocksX = divUp(src.rows, BLOCK_DIM_Y * PATCH_PER_BLOCK);
        const int nBlocksY = divUp(src.cols, BLOCK_DIM_X);
        const int nGrids = divUp(nBlocksY, maxGridY); // Number of launches required to process the whole image

        const dim3 block(BLOCK_DIM_X, BLOCK_DIM_Y);

        B<T> brd(src.rows);

        for(int i = 0; i < nGrids; ++i )
        {
            const int startColumn = i * maxGridY * BLOCK_DIM_X;
            const int endColumn =
                std::min(startColumn + maxGridY * BLOCK_DIM_X, src.cols);

            const int chunkCols = endColumn - startColumn;

            const dim3 grid(
                nBlocksX,
                divUp(chunkCols, BLOCK_DIM_X)
            );

            // Get the chunks as views of the original matrices
            PtrStepSz<T> srcView = src;
            srcView.data = src.ptr() + startColumn;
            srcView.cols = chunkCols;

            PtrStepSz<D> dstView = dst;
            dstView.data = dst.ptr() + startColumn;
            dstView.cols = chunkCols;

            linearColumnFilter<KSIZE, T, D><<<grid, block, 0, stream>>>(srcView, dstView, kernel, anchor, brd);
            cudaSafeCall( cudaGetLastError() );
        }

        if (stream == 0)
            cudaSafeCall( cudaDeviceSynchronize() );
    }

    template <int BRD_TYPE>
    __device__ __forceinline__ int borderIdx(int p, int len)
    {
        if (BRD_TYPE == cv::BORDER_REPLICATE)
            return p < 0 ? 0 : (p >= len ? len - 1 : p);

        if (BRD_TYPE == cv::BORDER_WRAP)
        {
            p %= len;
            if (p < 0)
                p += len;
            return p;
        }

        if (len == 1)
            return 0;

        if (BRD_TYPE == cv::BORDER_REFLECT)
        {
            const int period = 2 * len;
            p %= period;
            if (p < 0)
                p += period;
            if (p >= len)
                p = period - 1 - p;
            return p;
        }

        // cv::BORDER_REFLECT101
        const int period = 2 * (len - 1);
        p %= period;
        if (p < 0)
            p += period;
        if (p >= len)
            p = period - p;
        return p;
    }

    template <typename T, typename D, int BRD_TYPE>
    __global__ void linearColumnFilterArbitrary(const PtrStepSz<T> src, PtrStep<D> dst, const float* kernel, const int ksize, const int anchor)
    {
        const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
        if (x >= (unsigned int)src.cols)
            return;

        typedef typename TypeVec<float, VecTraits<T>::cn>::vec_type sum_t;

        const unsigned int stride_y = blockDim.y * gridDim.y;
        for (unsigned int y = blockIdx.y * blockDim.y + threadIdx.y; y < (unsigned int)src.rows; y += stride_y)
        {
            const int y0 = (int)y - anchor;
            const int k_start = ::max(0, ::min(ksize, -y0));
            const int k_end = ::max(k_start, ::min(ksize, src.rows - y0));

            sum_t sum = VecTraits<sum_t>::all(0);

            if (BRD_TYPE != cv::BORDER_CONSTANT)
            {
                for (int k = 0; k < k_start; ++k)
                    sum = sum + saturate_cast<sum_t>(src(borderIdx<BRD_TYPE>(y0 + k, src.rows), (int)x)) * kernel[k];
            }

            for (int k = k_start; k < k_end; ++k)
                sum = sum + saturate_cast<sum_t>(src(y0 + k, (int)x)) * kernel[k];

            if (BRD_TYPE != cv::BORDER_CONSTANT)
            {
                for (int k = k_end; k < ksize; ++k)
                    sum = sum + saturate_cast<sum_t>(src(borderIdx<BRD_TYPE>(y0 + k, src.rows), (int)x)) * kernel[k];
            }

            dst((int)y, (int)x) = saturate_cast<D>(sum);
        }
    }

    template <typename T, typename D, int BRD_TYPE>
    void callerArbitrary(PtrStepSz<T> src, PtrStepSz<D> dst, const float* kernel, int ksize, int anchor, int cc, cudaStream_t stream)
    {
        if (src.rows <= 0 || src.cols <= 0)
            return;

        const int BLOCK_DIM_X = 32;
        const int BLOCK_DIM_Y = (cc >= 20) ? 8 : 4;

        DeviceInfo devInfo;
        const int maxGridY = std::min(devInfo.maxGridSize()[1], 65535);
        const dim3 block(BLOCK_DIM_X, BLOCK_DIM_Y);
        const dim3 grid(divUp(src.cols, BLOCK_DIM_X), std::min(divUp(src.rows, BLOCK_DIM_Y), maxGridY));

        linearColumnFilterArbitrary<T, D, BRD_TYPE><<<grid, block, 0, stream>>>(src, dst, kernel, ksize, anchor);
        cudaSafeCall( cudaGetLastError() );

        if (stream == 0)
            cudaSafeCall( cudaDeviceSynchronize() );
    }
}

namespace filter
{
    template <typename T, typename D>
    void linearColumn(PtrStepSzb src, PtrStepSzb dst, const float* kernel, int ksize, int anchor, int brd_type, int cc, cudaStream_t stream)
    {
        if (ksize > MAX_KERNEL_SIZE)
        {
            typedef void (*arbitrary_caller_t)(PtrStepSz<T> src, PtrStepSz<D> dst, const float* kernel, int ksize, int anchor, int cc, cudaStream_t stream);

            static const arbitrary_caller_t arbitrary_callers[5] =
            {
                column_filter::callerArbitrary<T, D, cv::BORDER_CONSTANT>,
                column_filter::callerArbitrary<T, D, cv::BORDER_REPLICATE>,
                column_filter::callerArbitrary<T, D, cv::BORDER_REFLECT>,
                column_filter::callerArbitrary<T, D, cv::BORDER_WRAP>,
                column_filter::callerArbitrary<T, D, cv::BORDER_REFLECT101>
            };

            arbitrary_callers[brd_type]((PtrStepSz<T>)src, (PtrStepSz<D>)dst, kernel, ksize, anchor, cc, stream);
            return;
        }

        typedef void (*caller_t)(PtrStepSz<T> src, PtrStepSz<D> dst, const float* kernel, int anchor, int cc, cudaStream_t stream);

        static const caller_t callers[5][33] =
        {
            {
                0,
                column_filter::caller< 1, T, D, BrdColConstant>,
                column_filter::caller< 2, T, D, BrdColConstant>,
                column_filter::caller< 3, T, D, BrdColConstant>,
                column_filter::caller< 4, T, D, BrdColConstant>,
                column_filter::caller< 5, T, D, BrdColConstant>,
                column_filter::caller< 6, T, D, BrdColConstant>,
                column_filter::caller< 7, T, D, BrdColConstant>,
                column_filter::caller< 8, T, D, BrdColConstant>,
                column_filter::caller< 9, T, D, BrdColConstant>,
                column_filter::caller<10, T, D, BrdColConstant>,
                column_filter::caller<11, T, D, BrdColConstant>,
                column_filter::caller<12, T, D, BrdColConstant>,
                column_filter::caller<13, T, D, BrdColConstant>,
                column_filter::caller<14, T, D, BrdColConstant>,
                column_filter::caller<15, T, D, BrdColConstant>,
                column_filter::caller<16, T, D, BrdColConstant>,
                column_filter::caller<17, T, D, BrdColConstant>,
                column_filter::caller<18, T, D, BrdColConstant>,
                column_filter::caller<19, T, D, BrdColConstant>,
                column_filter::caller<20, T, D, BrdColConstant>,
                column_filter::caller<21, T, D, BrdColConstant>,
                column_filter::caller<22, T, D, BrdColConstant>,
                column_filter::caller<23, T, D, BrdColConstant>,
                column_filter::caller<24, T, D, BrdColConstant>,
                column_filter::caller<25, T, D, BrdColConstant>,
                column_filter::caller<26, T, D, BrdColConstant>,
                column_filter::caller<27, T, D, BrdColConstant>,
                column_filter::caller<28, T, D, BrdColConstant>,
                column_filter::caller<29, T, D, BrdColConstant>,
                column_filter::caller<30, T, D, BrdColConstant>,
                column_filter::caller<31, T, D, BrdColConstant>,
                column_filter::caller<32, T, D, BrdColConstant>
            },
            {
                0,
                column_filter::caller< 1, T, D, BrdColReplicate>,
                column_filter::caller< 2, T, D, BrdColReplicate>,
                column_filter::caller< 3, T, D, BrdColReplicate>,
                column_filter::caller< 4, T, D, BrdColReplicate>,
                column_filter::caller< 5, T, D, BrdColReplicate>,
                column_filter::caller< 6, T, D, BrdColReplicate>,
                column_filter::caller< 7, T, D, BrdColReplicate>,
                column_filter::caller< 8, T, D, BrdColReplicate>,
                column_filter::caller< 9, T, D, BrdColReplicate>,
                column_filter::caller<10, T, D, BrdColReplicate>,
                column_filter::caller<11, T, D, BrdColReplicate>,
                column_filter::caller<12, T, D, BrdColReplicate>,
                column_filter::caller<13, T, D, BrdColReplicate>,
                column_filter::caller<14, T, D, BrdColReplicate>,
                column_filter::caller<15, T, D, BrdColReplicate>,
                column_filter::caller<16, T, D, BrdColReplicate>,
                column_filter::caller<17, T, D, BrdColReplicate>,
                column_filter::caller<18, T, D, BrdColReplicate>,
                column_filter::caller<19, T, D, BrdColReplicate>,
                column_filter::caller<20, T, D, BrdColReplicate>,
                column_filter::caller<21, T, D, BrdColReplicate>,
                column_filter::caller<22, T, D, BrdColReplicate>,
                column_filter::caller<23, T, D, BrdColReplicate>,
                column_filter::caller<24, T, D, BrdColReplicate>,
                column_filter::caller<25, T, D, BrdColReplicate>,
                column_filter::caller<26, T, D, BrdColReplicate>,
                column_filter::caller<27, T, D, BrdColReplicate>,
                column_filter::caller<28, T, D, BrdColReplicate>,
                column_filter::caller<29, T, D, BrdColReplicate>,
                column_filter::caller<30, T, D, BrdColReplicate>,
                column_filter::caller<31, T, D, BrdColReplicate>,
                column_filter::caller<32, T, D, BrdColReplicate>
            },
            {
                0,
                column_filter::caller< 1, T, D, BrdColReflect>,
                column_filter::caller< 2, T, D, BrdColReflect>,
                column_filter::caller< 3, T, D, BrdColReflect>,
                column_filter::caller< 4, T, D, BrdColReflect>,
                column_filter::caller< 5, T, D, BrdColReflect>,
                column_filter::caller< 6, T, D, BrdColReflect>,
                column_filter::caller< 7, T, D, BrdColReflect>,
                column_filter::caller< 8, T, D, BrdColReflect>,
                column_filter::caller< 9, T, D, BrdColReflect>,
                column_filter::caller<10, T, D, BrdColReflect>,
                column_filter::caller<11, T, D, BrdColReflect>,
                column_filter::caller<12, T, D, BrdColReflect>,
                column_filter::caller<13, T, D, BrdColReflect>,
                column_filter::caller<14, T, D, BrdColReflect>,
                column_filter::caller<15, T, D, BrdColReflect>,
                column_filter::caller<16, T, D, BrdColReflect>,
                column_filter::caller<17, T, D, BrdColReflect>,
                column_filter::caller<18, T, D, BrdColReflect>,
                column_filter::caller<19, T, D, BrdColReflect>,
                column_filter::caller<20, T, D, BrdColReflect>,
                column_filter::caller<21, T, D, BrdColReflect>,
                column_filter::caller<22, T, D, BrdColReflect>,
                column_filter::caller<23, T, D, BrdColReflect>,
                column_filter::caller<24, T, D, BrdColReflect>,
                column_filter::caller<25, T, D, BrdColReflect>,
                column_filter::caller<26, T, D, BrdColReflect>,
                column_filter::caller<27, T, D, BrdColReflect>,
                column_filter::caller<28, T, D, BrdColReflect>,
                column_filter::caller<29, T, D, BrdColReflect>,
                column_filter::caller<30, T, D, BrdColReflect>,
                column_filter::caller<31, T, D, BrdColReflect>,
                column_filter::caller<32, T, D, BrdColReflect>
            },
            {
                0,
                column_filter::caller< 1, T, D, BrdColWrap>,
                column_filter::caller< 2, T, D, BrdColWrap>,
                column_filter::caller< 3, T, D, BrdColWrap>,
                column_filter::caller< 4, T, D, BrdColWrap>,
                column_filter::caller< 5, T, D, BrdColWrap>,
                column_filter::caller< 6, T, D, BrdColWrap>,
                column_filter::caller< 7, T, D, BrdColWrap>,
                column_filter::caller< 8, T, D, BrdColWrap>,
                column_filter::caller< 9, T, D, BrdColWrap>,
                column_filter::caller<10, T, D, BrdColWrap>,
                column_filter::caller<11, T, D, BrdColWrap>,
                column_filter::caller<12, T, D, BrdColWrap>,
                column_filter::caller<13, T, D, BrdColWrap>,
                column_filter::caller<14, T, D, BrdColWrap>,
                column_filter::caller<15, T, D, BrdColWrap>,
                column_filter::caller<16, T, D, BrdColWrap>,
                column_filter::caller<17, T, D, BrdColWrap>,
                column_filter::caller<18, T, D, BrdColWrap>,
                column_filter::caller<19, T, D, BrdColWrap>,
                column_filter::caller<20, T, D, BrdColWrap>,
                column_filter::caller<21, T, D, BrdColWrap>,
                column_filter::caller<22, T, D, BrdColWrap>,
                column_filter::caller<23, T, D, BrdColWrap>,
                column_filter::caller<24, T, D, BrdColWrap>,
                column_filter::caller<25, T, D, BrdColWrap>,
                column_filter::caller<26, T, D, BrdColWrap>,
                column_filter::caller<27, T, D, BrdColWrap>,
                column_filter::caller<28, T, D, BrdColWrap>,
                column_filter::caller<29, T, D, BrdColWrap>,
                column_filter::caller<30, T, D, BrdColWrap>,
                column_filter::caller<31, T, D, BrdColWrap>,
                column_filter::caller<32, T, D, BrdColWrap>
            },
            {
                0,
                column_filter::caller< 1, T, D, BrdColReflect101>,
                column_filter::caller< 2, T, D, BrdColReflect101>,
                column_filter::caller< 3, T, D, BrdColReflect101>,
                column_filter::caller< 4, T, D, BrdColReflect101>,
                column_filter::caller< 5, T, D, BrdColReflect101>,
                column_filter::caller< 6, T, D, BrdColReflect101>,
                column_filter::caller< 7, T, D, BrdColReflect101>,
                column_filter::caller< 8, T, D, BrdColReflect101>,
                column_filter::caller< 9, T, D, BrdColReflect101>,
                column_filter::caller<10, T, D, BrdColReflect101>,
                column_filter::caller<11, T, D, BrdColReflect101>,
                column_filter::caller<12, T, D, BrdColReflect101>,
                column_filter::caller<13, T, D, BrdColReflect101>,
                column_filter::caller<14, T, D, BrdColReflect101>,
                column_filter::caller<15, T, D, BrdColReflect101>,
                column_filter::caller<16, T, D, BrdColReflect101>,
                column_filter::caller<17, T, D, BrdColReflect101>,
                column_filter::caller<18, T, D, BrdColReflect101>,
                column_filter::caller<19, T, D, BrdColReflect101>,
                column_filter::caller<20, T, D, BrdColReflect101>,
                column_filter::caller<21, T, D, BrdColReflect101>,
                column_filter::caller<22, T, D, BrdColReflect101>,
                column_filter::caller<23, T, D, BrdColReflect101>,
                column_filter::caller<24, T, D, BrdColReflect101>,
                column_filter::caller<25, T, D, BrdColReflect101>,
                column_filter::caller<26, T, D, BrdColReflect101>,
                column_filter::caller<27, T, D, BrdColReflect101>,
                column_filter::caller<28, T, D, BrdColReflect101>,
                column_filter::caller<29, T, D, BrdColReflect101>,
                column_filter::caller<30, T, D, BrdColReflect101>,
                column_filter::caller<31, T, D, BrdColReflect101>,
                column_filter::caller<32, T, D, BrdColReflect101>
            }
        };

        callers[brd_type][ksize]((PtrStepSz<T>)src, (PtrStepSz<D>)dst, kernel, anchor, cc, stream);
    }
}
