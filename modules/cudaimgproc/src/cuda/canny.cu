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

#include "opencv2/core/cuda/common.hpp"
#include "opencv2/core/cuda/emulation.hpp"
#include "opencv2/core/cuda/transform.hpp"
#include "opencv2/core/cuda/functional.hpp"
#include "opencv2/core/cuda/utility.hpp"
#include "opencv2/core/cuda.hpp"
#include <opencv2/cudev/ptr2d/texture.hpp>

using namespace cv::cuda;
using namespace cv::cuda::device;

namespace canny
{
    struct L1 : binary_function<int, int, float>
    {
        __device__ __forceinline__ float operator ()(int x, int y) const
        {
            return ::abs(x) + ::abs(y);
        }

        __host__ __device__ __forceinline__ L1() {}
        __host__ __device__ __forceinline__ L1(const L1&) {}
    };
    struct L2 : binary_function<int, int, float>
    {
        __device__ __forceinline__ float operator ()(int x, int y) const
        {
            return ::sqrtf(x * x + y * y);
        }

        __host__ __device__ __forceinline__ L2() {}
        __host__ __device__ __forceinline__ L2(const L2&) {}
    };
}

namespace cv { namespace cuda { namespace device
{
    template <> struct TransformFunctorTraits<canny::L1> : DefaultTransformFunctorTraits<canny::L1>
    {
        enum { smart_shift = 4 };
    };
    template <> struct TransformFunctorTraits<canny::L2> : DefaultTransformFunctorTraits<canny::L2>
    {
        enum { smart_shift = 4 };
    };
}}}

namespace canny
{
    template <class Norm>
    __global__ void calcMagnitudeKernel(cv::cudev::TextureOffPtr<uchar> texSrc, PtrStepi dx, PtrStepi dy, PtrStepSzf mag, const Norm norm)
    {
        const int x = blockIdx.x * blockDim.x + threadIdx.x;
        const int y = blockIdx.y * blockDim.y + threadIdx.y;

        if (y >= mag.rows || x >= mag.cols)
            return;

        int dxVal = (texSrc(y - 1, x + 1) + 2 * texSrc(y, x + 1) + texSrc(y + 1, x + 1)) - (texSrc(y - 1, x - 1) + 2 * texSrc(y, x - 1) + texSrc(y + 1, x - 1));
        int dyVal = (texSrc(y + 1, x - 1) + 2 * texSrc(y + 1, x) + texSrc(y + 1, x + 1)) - (texSrc(y - 1, x - 1) + 2 * texSrc(y - 1, x) + texSrc(y - 1, x + 1));

        dx(y, x) = dxVal;
        dy(y, x) = dyVal;

        mag(y, x) = norm(dxVal, dyVal);
    }

    void calcMagnitude(PtrStepSzb srcWhole, int xoff, int yoff, PtrStepSzi dx, PtrStepSzi dy, PtrStepSzf mag, bool L2Grad, cudaStream_t stream)
    {
        const dim3 block(16, 16);
        const dim3 grid(divUp(mag.cols, block.x), divUp(mag.rows, block.y));
        cv::cudev::TextureOff<uchar> texSrc(srcWhole, yoff, xoff);
        if (L2Grad)
        {
            L2 norm;
            calcMagnitudeKernel<<<grid, block, 0, stream>>>(texSrc, dx, dy, mag, norm);
        }
        else
        {
            L1 norm;
            calcMagnitudeKernel<<<grid, block, 0, stream>>>(texSrc, dx, dy, mag, norm);
        }

        if (stream == NULL)
            cudaSafeCall(cudaDeviceSynchronize());
    }

    void calcMagnitude(PtrStepSzi dx, PtrStepSzi dy, PtrStepSzf mag, bool L2Grad, cudaStream_t stream)
    {
        if (L2Grad)
        {
            L2 norm;
            transform(dx, dy, mag, norm, WithOutMask(), stream);
        }
        else
        {
            L1 norm;
            transform(dx, dy, mag, norm, WithOutMask(), stream);
        }
    }
}

//////////////////////////////////////////////////////////////////////////////////////////

namespace canny
{
    __global__ void calcMapKernel(cv::cudev::TexturePtr<float> texMag, const PtrStepSzi dx, const PtrStepi dy, PtrStepi map, const float low_thresh, const float high_thresh)
    {
        const int CANNY_SHIFT = 15;
        const int TG22 = (int)(0.4142135623730950488016887242097*(1<<CANNY_SHIFT) + 0.5);

        const int x = blockIdx.x * blockDim.x + threadIdx.x;
        const int y = blockIdx.y * blockDim.y + threadIdx.y;

        if (x == 0 || x >= dx.cols - 1 || y == 0 || y >= dx.rows - 1)
            return;

        int dxVal = dx(y, x);
        int dyVal = dy(y, x);

        const int s = (dxVal ^ dyVal) < 0 ? -1 : 1;
        const float m = texMag(y, x);

        dxVal = ::abs(dxVal);
        dyVal = ::abs(dyVal);

        // 0 - the pixel can not belong to an edge
        // 1 - the pixel might belong to an edge
        // 2 - the pixel does belong to an edge
        int edge_type = 0;

        if (m > low_thresh)
        {
            const int tg22x = dxVal * TG22;
            const int tg67x = tg22x + ((dxVal + dxVal) << CANNY_SHIFT);

            dyVal <<= CANNY_SHIFT;

            if (dyVal < tg22x)
            {
                if (m > texMag(y, x - 1) && m >= texMag(y, x + 1))
                    edge_type = 1 + (int)(m > high_thresh);
            }
            else if(dyVal > tg67x)
            {
                if (m > texMag(y - 1, x) && m >= texMag(y + 1, x))
                    edge_type = 1 + (int)(m > high_thresh);
            }
            else
            {
                if (m > texMag(y - 1, x - s) && m >= texMag(y + 1, x + s))
                    edge_type = 1 + (int)(m > high_thresh);
            }
        }

        map(y, x) = edge_type;
    }

    void calcMap(PtrStepSzi dx, PtrStepSzi dy, PtrStepSzf mag, PtrStepSzi map, float low_thresh, float high_thresh, cudaStream_t stream)
    {
        const dim3 block(16, 16);
        const dim3 grid(divUp(dx.cols, block.x), divUp(dx.rows, block.y));
        cv::cudev::Texture<float> texMag(mag);
        calcMapKernel<<<grid, block, 0, stream>>>(texMag, dx, dy, map, low_thresh, high_thresh);
        if (stream == NULL)
            cudaSafeCall( cudaDeviceSynchronize() );
    }
}

//////////////////////////////////////////////////////////////////////////////////////////

namespace canny
{
    __device__ __forceinline__ bool checkIdx(int y, int x, int rows, int cols)
    {
        return (y >= 0) && (y < rows) && (x >= 0) && (x < cols);
    }

    __global__ void edgesHysteresisLocalKernel(PtrStepSzi map, short2* st, int* d_counter)
    {
        __shared__ volatile int smem[18][18];

        const int x = blockIdx.x * blockDim.x + threadIdx.x;
        const int y = blockIdx.y * blockDim.y + threadIdx.y;

        smem[threadIdx.y + 1][threadIdx.x + 1] = checkIdx(y, x, map.rows, map.cols) ? map(y, x) : 0;
        if (threadIdx.y == 0)
            smem[0][threadIdx.x + 1] = checkIdx(y - 1, x, map.rows, map.cols) ? map(y - 1, x) : 0;
        if (threadIdx.y == blockDim.y - 1)
            smem[blockDim.y + 1][threadIdx.x + 1] = checkIdx(y + 1, x, map.rows, map.cols) ? map(y + 1, x) : 0;
        if (threadIdx.x == 0)
            smem[threadIdx.y + 1][0] = checkIdx(y, x - 1, map.rows, map.cols) ? map(y, x - 1) : 0;
        if (threadIdx.x == blockDim.x - 1)
            smem[threadIdx.y + 1][blockDim.x + 1] = checkIdx(y, x + 1, map.rows, map.cols) ? map(y, x + 1) : 0;
        if (threadIdx.x == 0 && threadIdx.y == 0)
            smem[0][0] = checkIdx(y - 1, x - 1, map.rows, map.cols) ? map(y - 1, x - 1) : 0;
        if (threadIdx.x == blockDim.x - 1 && threadIdx.y == 0)
            smem[0][blockDim.x + 1] = checkIdx(y - 1, x + 1, map.rows, map.cols) ? map(y - 1, x + 1) : 0;
        if (threadIdx.x == 0 && threadIdx.y == blockDim.y - 1)
            smem[blockDim.y + 1][0] = checkIdx(y + 1, x - 1, map.rows, map.cols) ? map(y + 1, x - 1) : 0;
        if (threadIdx.x == blockDim.x - 1 && threadIdx.y == blockDim.y - 1)
            smem[blockDim.y + 1][blockDim.x + 1] = checkIdx(y + 1, x + 1, map.rows, map.cols) ? map(y + 1, x + 1) : 0;

        __syncthreads();

        if (x >= map.cols || y >= map.rows)
            return;

        int n;

        #pragma unroll
        for (int k = 0; k < 16; ++k)
        {
            n = 0;

            if (smem[threadIdx.y + 1][threadIdx.x + 1] == 1)
            {
                n += smem[threadIdx.y    ][threadIdx.x    ] == 2;
                n += smem[threadIdx.y    ][threadIdx.x + 1] == 2;
                n += smem[threadIdx.y    ][threadIdx.x + 2] == 2;

                n += smem[threadIdx.y + 1][threadIdx.x    ] == 2;
                n += smem[threadIdx.y + 1][threadIdx.x + 2] == 2;

                n += smem[threadIdx.y + 2][threadIdx.x    ] == 2;
                n += smem[threadIdx.y + 2][threadIdx.x + 1] == 2;
                n += smem[threadIdx.y + 2][threadIdx.x + 2] == 2;
            }

            __syncthreads();

            if (n > 0)
                smem[threadIdx.y + 1][threadIdx.x + 1] = 2;

            __syncthreads();
        }

        const int e = smem[threadIdx.y + 1][threadIdx.x + 1];

        map(y, x) = e;

        n = 0;

        if (e == 2)
        {
            n += smem[threadIdx.y    ][threadIdx.x    ] == 1;
            n += smem[threadIdx.y    ][threadIdx.x + 1] == 1;
            n += smem[threadIdx.y    ][threadIdx.x + 2] == 1;

            n += smem[threadIdx.y + 1][threadIdx.x    ] == 1;
            n += smem[threadIdx.y + 1][threadIdx.x + 2] == 1;

            n += smem[threadIdx.y + 2][threadIdx.x    ] == 1;
            n += smem[threadIdx.y + 2][threadIdx.x + 1] == 1;
            n += smem[threadIdx.y + 2][threadIdx.x + 2] == 1;
        }

        if (n > 0)
        {
            const int ind =  ::atomicAdd(d_counter, 1);
            st[ind] = make_short2(x, y);
        }
    }

    void edgesHysteresisLocal(PtrStepSzi map, short2* st1, int* d_counter, cudaStream_t stream)
    {
        cudaSafeCall( cudaMemsetAsync(d_counter, 0, sizeof(int), stream) );

        const dim3 block(16, 16);
        const dim3 grid(divUp(map.cols, block.x), divUp(map.rows, block.y));

        edgesHysteresisLocalKernel<<<grid, block, 0, stream>>>(map, st1, d_counter);
        cudaSafeCall( cudaGetLastError() );

        if (stream == NULL)
            cudaSafeCall( cudaDeviceSynchronize() );
    }
}

//////////////////////////////////////////////////////////////////////////////////////////

namespace canny
{
    __constant__ int c_dx[8] = {-1,  0,  1, -1, 1, -1, 0, 1};
    __constant__ int c_dy[8] = {-1, -1, -1,  0, 0,  1, 1, 1};

    __global__ void edgesHysteresisGlobalKernel(PtrStepSzi map, short2* st1, short2* st2, int* d_counter, const int count)
    {
        const int stack_size = 512;

        __shared__ int s_counter;
        __shared__ int s_ind;
        __shared__ short2 s_st[stack_size];

        if (threadIdx.x == 0)
            s_counter = 0;

        __syncthreads();

        int ind = blockIdx.y * gridDim.x + blockIdx.x;

        if (ind >= count)
            return;

        short2 pos = st1[ind];

        if (threadIdx.x < 8)
        {
            pos.x += c_dx[threadIdx.x];
            pos.y += c_dy[threadIdx.x];

            if (pos.x > 0 && pos.x < map.cols - 1 && pos.y > 0 && pos.y < map.rows - 1 && map(pos.y, pos.x) == 1)
            {
                map(pos.y, pos.x) = 2;

                ind = Emulation::smem::atomicAdd(&s_counter, 1);

                s_st[ind] = pos;
            }
        }

        __syncthreads();

        while (s_counter > 0 && s_counter <= stack_size - blockDim.x)
        {
            const int subTaskIdx = threadIdx.x >> 3;
            const int portion = ::min(s_counter, blockDim.x >> 3);

            if (subTaskIdx < portion)
                pos = s_st[s_counter - 1 - subTaskIdx];

            __syncthreads();

            if (threadIdx.x == 0)
                s_counter -= portion;

            __syncthreads();

            if (subTaskIdx < portion)
            {
                pos.x += c_dx[threadIdx.x & 7];
                pos.y += c_dy[threadIdx.x & 7];

                if (pos.x > 0 && pos.x < map.cols - 1 && pos.y > 0 && pos.y < map.rows - 1 && map(pos.y, pos.x) == 1)
                {
                    map(pos.y, pos.x) = 2;

                    ind = Emulation::smem::atomicAdd(&s_counter, 1);

                    s_st[ind] = pos;
                }
            }

            __syncthreads();
        }

        if (s_counter > 0)
        {
            if (threadIdx.x == 0)
            {
                s_ind = ::atomicAdd(d_counter, s_counter);

                if (s_ind + s_counter > map.cols * map.rows)
                    s_counter = 0;
            }

            __syncthreads();

            ind = s_ind;

            for (int i = threadIdx.x; i < s_counter; i += blockDim.x)
                st2[ind + i] = s_st[i];
        }
    }

    void edgesHysteresisGlobal(PtrStepSzi map, short2* st1, short2* st2, int* d_counter, cudaStream_t stream)
    {
        int count;
        cudaSafeCall( cudaMemcpyAsync(&count, d_counter, sizeof(int), cudaMemcpyDeviceToHost, stream) );
        cudaSafeCall( cudaStreamSynchronize(stream) );

        while (count > 0)
        {
            cudaSafeCall( cudaMemsetAsync(d_counter, 0, sizeof(int), stream) );

            const dim3 block(128);
            const dim3 grid(std::min(count, 65535), divUp(count, 65535), 1);

            edgesHysteresisGlobalKernel<<<grid, block, 0, stream>>>(map, st1, st2, d_counter, count);
            cudaSafeCall( cudaGetLastError() );

            if (stream == NULL)
                cudaSafeCall( cudaDeviceSynchronize() );

            cudaSafeCall( cudaMemcpyAsync(&count, d_counter, sizeof(int), cudaMemcpyDeviceToHost, stream) );
            cudaSafeCall( cudaStreamSynchronize(stream) );

            count = std::min(count, map.cols * map.rows);

            //std::swap(st1, st2);
            short2* tmp = st1;
            st1 = st2;
            st2 = tmp;
        }
    }
}

//////////////////////////////////////////////////////////////////////////////////////////

namespace canny
{
    struct GetEdges : unary_function<int, uchar>
    {
        __device__ __forceinline__ uchar operator ()(int e) const
        {
            return (uchar)(-(e >> 1));
        }

        __host__ __device__ __forceinline__ GetEdges() {}
        __host__ __device__ __forceinline__ GetEdges(const GetEdges&) {}
    };
}

namespace cv { namespace cuda { namespace device
{
    template <> struct TransformFunctorTraits<canny::GetEdges> : DefaultTransformFunctorTraits<canny::GetEdges>
    {
        enum { smart_shift = 4 };
    };
}}}

namespace canny
{
    void getEdges(PtrStepSzi map, PtrStepSzb dst, cudaStream_t stream)
    {
        transform(map, dst, GetEdges(), WithOutMask(), stream);
    }
}

#endif /* CUDA_DISABLER */
