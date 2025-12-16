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

#include <cfloat>
#include <cmath>
#include "opencv2/core/cuda/common.hpp"
#include "opencv2/core/cuda/border_interpolate.hpp"
#include "opencv2/core/cuda/vec_traits.hpp"
#include "opencv2/core/cuda/vec_math.hpp"
#include "opencv2/core/cuda/saturate_cast.hpp"
#include "opencv2/core/cuda/filters.hpp"
#include <opencv2/cudev/ptr2d/texture.hpp>

namespace cv { namespace cuda { namespace device
{
    __device__ __forceinline__ float lanczos_weight(float x_)
    {
        float x = fabsf(x_);
        if (x == 0.0f)
            return 1.0f;
        if (x >= 4.0f)
            return 0.0f;
        float pi_x = M_PI * x;
        return sinf(pi_x) * sinf(pi_x / 4.0f) / (pi_x * pi_x / 4.0f);
    }

    // kernels
    template <typename T>
    __global__ void resize_lanczos4(const PtrStepSz<T> src, PtrStepSz<T> dst, const float fy, const float fx)
    {
        const int tx = threadIdx.x;
        const int ty = threadIdx.y;
        const int bx = blockIdx.x;
        const int by = blockIdx.y;

        const int x = bx * blockDim.x + tx;
        const int y = by * blockDim.y + ty;

        const int in_height = src.rows;
        const int in_width = src.cols;

        constexpr int R = 4;
        constexpr int BASE_W = 32;
        constexpr int BASE_H = 8;
        constexpr int SHARED_WIDTH_MAX = BASE_W + R + R;
        constexpr int SHARED_HEIGHT_MAX = BASE_H + R + R;

        __shared__ T shared_src[SHARED_HEIGHT_MAX][SHARED_WIDTH_MAX];

        typedef typename VecTraits<T>::elem_type elem_type;
        constexpr int cn = VecTraits<T>::cn;

        const int out_x0 = bx * blockDim.x;
        const int out_x1 = ::min(out_x0 + blockDim.x - 1, dst.cols - 1);
        const int out_y0 = by * blockDim.y;
        const int out_y1 = ::min(out_y0 + blockDim.y - 1, dst.rows - 1);

        const float src_x0_f = (static_cast<float>(out_x0) + 0.5f) * fx - 0.5f;
        const float src_x1_f = (static_cast<float>(out_x1) + 0.5f) * fx - 0.5f;
        const float src_y0_f = (static_cast<float>(out_y0) + 0.5f) * fy - 0.5f;
        const float src_y1_f = (static_cast<float>(out_y1) + 0.5f) * fy - 0.5f;

        int in_x_min = int(floorf(src_x0_f)) - R;
        int in_x_max = int(floorf(src_x1_f)) + R;
        int in_y_min = int(floorf(src_y0_f)) - R;
        int in_y_max = int(floorf(src_y1_f)) + R;

        if (in_x_min < 0) in_x_min = 0;
        if (in_y_min < 0) in_y_min = 0;
        if (in_x_max >= in_width) in_x_max = in_width - 1;
        if (in_y_max >= in_height) in_y_max = in_height - 1;

        const int W_needed = in_x_max - in_x_min + 1;
        const int H_needed = in_y_max - in_y_min + 1;

        // for fx <= 1 and fy <= 1
        const bool use_shared = (W_needed <= SHARED_WIDTH_MAX) && (H_needed <= SHARED_HEIGHT_MAX);

        if (use_shared)
        {
            for (int sy = ty; sy < H_needed; sy += blockDim.y)
            {
                int iy = in_y_min + sy;
                for (int sx = tx; sx < W_needed; sx += blockDim.x)
                {
                    int ix = in_x_min + sx;
                    shared_src[sy][sx] = src(iy, ix);
                }
            }
            __syncthreads();
        }

        if (x >= dst.cols || y >= dst.rows)
        {
            if (use_shared) { __syncthreads(); }
            return;
        }

        const float src_x = (static_cast<float>(x) + 0.5f) * fx - 0.5f;
        const float src_y = (static_cast<float>(y) + 0.5f) * fy - 0.5f;

        const int xmin = int(floorf(src_x)) - 3;
        const int xmax = int(floorf(src_x)) + 4;
        const int ymin = int(floorf(src_y)) - 3;
        const int ymax = int(floorf(src_y)) + 4;

        float results[cn];
        float acc_weights[cn];
        #pragma unroll
        for (int c = 0; c < cn; ++c) { results[c] = 0.0f; acc_weights[c] = 0.0f; }

        for (int cy = ymin; cy <= ymax; ++cy)
        {
            float wy = lanczos_weight(src_y - static_cast<float>(cy));
            if (wy == 0.0f) continue;

            for (int cx = xmin; cx <= xmax; ++cx)
            {
                float wx = lanczos_weight(src_x - static_cast<float>(cx));
                if (wx == 0.0f) continue;

                float w = wy * wx;

                if (use_shared)
                {
                    int sx = cx - in_x_min;
                    int sy = cy - in_y_min;
                    if (sx < 0) sx = 0;
                    else if (sx >= W_needed) sx = W_needed - 1;
                    if (sy < 0) sy = 0;
                    else if (sy >= H_needed) sy = H_needed - 1;

                    T val = shared_src[sy][sx];
                    const elem_type* val_ptr = reinterpret_cast<const elem_type*>(&val);
                    #pragma unroll
                    for (int c = 0; c < cn; ++c)
                    {
                        elem_type elem_val = val_ptr[c];
                        float channel_val = static_cast<float>(elem_val);
                        results[c] += channel_val * w;
                        acc_weights[c] += w;
                    }
                }
                else
                {
                    // fallback
                    int iy_r = cy < 0 ? 0 : (cy >= in_height ? (in_height - 1) : cy);
                    int ix_r = cx < 0 ? 0 : (cx >= in_width ? (in_width - 1) : cx);
                    T val = src(iy_r, ix_r);
                    const elem_type* val_ptr = reinterpret_cast<const elem_type*>(&val);
                    #pragma unroll
                    for (int c = 0; c < cn; ++c)
                    {
                        elem_type elem_val = val_ptr[c];
                        float channel_val = static_cast<float>(elem_val);
                        results[c] += channel_val * w;
                        acc_weights[c] += w;
                    }
                }
            }
        }

        #pragma unroll
        for (int c = 0; c < cn; ++c)
            results[c] = acc_weights[c] > 0.0f ? (results[c] / acc_weights[c]) : 0.0f;

        T result_vec;
        elem_type* result_ptr = reinterpret_cast<elem_type*>(&result_vec);

        #pragma unroll
        for (int c = 0; c < cn; ++c)
            result_ptr[c] = saturate_cast<elem_type>(results[c]);

        dst(y, x) = result_vec;
    }



    template <typename T> __global__ void resize_nearest(const PtrStep<T> src, PtrStepSz<T> dst, const float fy, const float fx)
    {
        const int dst_x = blockDim.x * blockIdx.x + threadIdx.x;
        const int dst_y = blockDim.y * blockIdx.y + threadIdx.y;

        if (dst_x < dst.cols && dst_y < dst.rows)
        {
            const float src_x = dst_x * fx;
            const float src_y = dst_y * fy;

            dst(dst_y, dst_x) = src(__float2int_rz(src_y), __float2int_rz(src_x));
        }
    }

    template <typename T> __global__ void resize_linear(const PtrStepSz<T> src, PtrStepSz<T> dst, const float fy, const float fx)
    {
        typedef typename TypeVec<float, VecTraits<T>::cn>::vec_type work_type;

        const int dst_x = blockDim.x * blockIdx.x + threadIdx.x;
        const int dst_y = blockDim.y * blockIdx.y + threadIdx.y;

        if (dst_x < dst.cols && dst_y < dst.rows)
        {
            const float src_x = dst_x * fx;
            const float src_y = dst_y * fy;

            work_type out = VecTraits<work_type>::all(0);

            const int x1 = __float2int_rd(src_x);
            const int y1 = __float2int_rd(src_y);
            const int x2 = x1 + 1;
            const int y2 = y1 + 1;
            const int x2_read = ::min(x2, src.cols - 1);
            const int y2_read = ::min(y2, src.rows - 1);

            T src_reg = src(y1, x1);
            out = out + src_reg * ((x2 - src_x) * (y2 - src_y));

            src_reg = src(y1, x2_read);
            out = out + src_reg * ((src_x - x1) * (y2 - src_y));

            src_reg = src(y2_read, x1);
            out = out + src_reg * ((x2 - src_x) * (src_y - y1));

            src_reg = src(y2_read, x2_read);
            out = out + src_reg * ((src_x - x1) * (src_y - y1));

            dst(dst_y, dst_x) = saturate_cast<T>(out);
        }
    }

    template <class Ptr2D, typename T> __global__ void resize(Ptr2D src, PtrStepSz<T> dst, const float fy, const float fx)
    {
        const int dst_x = blockDim.x * blockIdx.x + threadIdx.x;
        const int dst_y = blockDim.y * blockIdx.y + threadIdx.y;

        if (dst_x < dst.cols && dst_y < dst.rows)
        {
            const float src_x = dst_x * fx;
            const float src_y = dst_y * fy;

            dst(dst_y, dst_x) = src(src_y, src_x);
        }
    }

    template <typename Ptr2D, typename T> __global__ void resize_area(const Ptr2D src, PtrStepSz<T> dst)
    {
        const int x = blockDim.x * blockIdx.x + threadIdx.x;
        const int y = blockDim.y * blockIdx.y + threadIdx.y;

        if (x < dst.cols && y < dst.rows)
        {
            dst(y, x) = src(y, x);
        }
    }

    // callers for nearest interpolation

    template <typename T>
    void call_resize_nearest_glob(const PtrStepSz<T>& src, const PtrStepSz<T>& dst, float fy, float fx, cudaStream_t stream)
    {
        const dim3 block(32, 8);
        const dim3 grid(divUp(dst.cols, block.x), divUp(dst.rows, block.y));

        resize_nearest<<<grid, block, 0, stream>>>(src, dst, fy, fx);
        cudaSafeCall( cudaGetLastError() );

        if (stream == 0)
            cudaSafeCall( cudaDeviceSynchronize() );
    }

    template <typename T>
    void call_resize_nearest_tex(const PtrStepSz<T>& srcWhole, int yoff, int xoff, const PtrStepSz<T>& dst, float fy, float fx)
    {
        const dim3 block(32, 8);
        const dim3 grid(divUp(dst.cols, block.x), divUp(dst.rows, block.y));
        if (xoff || yoff) {
            cudev::TextureOff<T> texSrcWhole(srcWhole, yoff, xoff);
            resize<cudev::TextureOffPtr<T>><<<grid, block>>>(texSrcWhole, dst, fy, fx);
        }
        else {
            cudev::Texture<T> texSrcWhole(srcWhole);
            resize<cudev::TexturePtr<T>><<<grid, block>>>(texSrcWhole, dst, fy, fx);
        }
        cudaSafeCall( cudaGetLastError() );
        cudaSafeCall( cudaDeviceSynchronize() );
    }

    // callers for linear interpolation

    template <typename T>
    void call_resize_linear_glob(const PtrStepSz<T>& src, const PtrStepSz<T>& dst, float fy, float fx, cudaStream_t stream)
    {
        const dim3 block(32, 8);
        const dim3 grid(divUp(dst.cols, block.x), divUp(dst.rows, block.y));

        resize_linear<<<grid, block, 0, stream>>>(src, dst, fy, fx);
        cudaSafeCall( cudaGetLastError() );

        if (stream == 0)
            cudaSafeCall( cudaDeviceSynchronize() );
    }

    template <typename T>
    void call_resize_linear_tex(const PtrStepSz<T>& src, const PtrStepSz<T>& srcWhole, int yoff, int xoff, const PtrStepSz<T>& dst, float fy, float fx)
    {
        const dim3 block(32, 8);
        const dim3 grid(divUp(dst.cols, block.x), divUp(dst.rows, block.y));
        if (srcWhole.data == src.data)
        {
            cudev::Texture<T> texSrc(src);
            LinearFilter<cudev::TexturePtr<T>> filteredSrc(texSrc);
            resize<<<grid, block>>>(filteredSrc, dst, fy, fx);
        }
        else
        {
            cudev::TextureOff<T> texSrcWhole(srcWhole, yoff, xoff);
            BrdReplicate<T> brd(src.rows, src.cols);
            BorderReader<cudev::TextureOffPtr<T>, BrdReplicate<T>> brdSrc(texSrcWhole, brd);
            LinearFilter<BorderReader<cudev::TextureOffPtr<T>, BrdReplicate<T>>> filteredSrc(brdSrc);
            resize<<<grid, block>>>(filteredSrc, dst, fy, fx);
        }
        cudaSafeCall( cudaGetLastError() );
        cudaSafeCall( cudaDeviceSynchronize() );
    }

    // callers for cubic interpolation

    template <typename T>
    void call_resize_cubic_glob(const PtrStepSz<T>& src, const PtrStepSz<T>& dst, float fy, float fx, cudaStream_t stream)
    {
        const dim3 block(32, 8);
        const dim3 grid(divUp(dst.cols, block.x), divUp(dst.rows, block.y));

        BrdReplicate<T> brd(src.rows, src.cols);
        BorderReader<PtrStep<T>, BrdReplicate<T>> brdSrc(src, brd);
        CubicFilter<BorderReader< PtrStep<T>, BrdReplicate<T>>> filteredSrc(brdSrc);

        resize<<<grid, block, 0, stream>>>(filteredSrc, dst, fy, fx);
        cudaSafeCall( cudaGetLastError() );

        if (stream == 0)
            cudaSafeCall( cudaDeviceSynchronize() );
    }

    template <typename T>
    void call_resize_cubic_tex(const PtrStepSz<T>& src, const PtrStepSz<T>& srcWhole, int yoff, int xoff, const PtrStepSz<T>& dst, float fy, float fx)
    {
        const dim3 block(32, 8);
        const dim3 grid(divUp(dst.cols, block.x), divUp(dst.rows, block.y));
        if (srcWhole.data == src.data)
        {
            cudev::Texture<T> texSrc(src);
            CubicFilter<cudev::TexturePtr<T>> filteredSrc(texSrc);
            resize<<<grid, block>>>(filteredSrc, dst, fy, fx);
        }
        else
        {
            cudev::TextureOff<T> texSrcWhole(srcWhole, yoff, xoff);
            BrdReplicate<T> brd(src.rows, src.cols);
            BorderReader<cudev::TextureOffPtr<T>, BrdReplicate<T>> brdSrc(texSrcWhole, brd);
            CubicFilter<BorderReader<cudev::TextureOffPtr<T>, BrdReplicate<T>>> filteredSrc(brdSrc);
            resize<<<grid, block>>>(filteredSrc, dst, fy, fx);
        }
        cudaSafeCall( cudaGetLastError() );
        cudaSafeCall( cudaDeviceSynchronize() );
    }

    // callers for lanczos interpolation

    template <typename T>
    void call_resize_lanczos4_glob(const PtrStepSz<T>& src, const PtrStepSz<T>& dst, float fy, float fx, cudaStream_t stream)
    {
        const dim3 block(32, 8);
        const dim3 grid(divUp(dst.cols, block.x), divUp(dst.rows, block.y));

        resize_lanczos4<<<grid, block, 0, stream>>>(src, dst, fy, fx);
        cudaSafeCall( cudaGetLastError() );

        if (stream == 0)
            cudaSafeCall( cudaDeviceSynchronize() );
    }

    template <typename Ptr2D, typename T>
    __global__ void resize_lanczos4_tex(Ptr2D src, PtrStepSz<T> dst, const float fy, const float fx)
    {
        const int x = blockDim.x * blockIdx.x + threadIdx.x;
        const int y = blockDim.y * blockIdx.y + threadIdx.y;

        if (x >= dst.cols || y >= dst.rows)
            return;

        const float src_x = (static_cast<float>(x) + 0.5f) * fx - 0.5f;
        const float src_y = (static_cast<float>(y) + 0.5f) * fy - 0.5f;

        typedef typename VecTraits<T>::elem_type elem_type;
        constexpr int cn = VecTraits<T>::cn;
        float results[cn] = {0.0f};

        for (int c = 0; c < cn; ++c)
        {
            float acc_val = 0.0f;
            float acc_weight = 0.0f;

            const int xmin = int(floorf(src_x)) - 3;
            const int xmax = int(floorf(src_x)) + 4;
            const int ymin = int(floorf(src_y)) - 3;
            const int ymax = int(floorf(src_y)) + 4;

            for (int cy = ymin; cy <= ymax; ++cy)
            {
                float wy = lanczos_weight(src_y - static_cast<float>(cy));
                if (wy == 0.0f)
                    continue;

                for (int cx = xmin; cx <= xmax; ++cx)
                {
                    float wx = lanczos_weight(src_x - static_cast<float>(cx));
                    if (wx == 0.0f)
                        continue;

                    float w = wy * wx;

                    // Use texture memory for sampling (handles boundary automatically)
                    T val = src(static_cast<float>(cy), static_cast<float>(cx));

                    const elem_type* val_ptr = reinterpret_cast<const elem_type*>(&val);
                    elem_type elem_val = val_ptr[c];
                    float channel_val = static_cast<float>(elem_val);

                    acc_val += channel_val * w;
                    acc_weight += w;
                }
            }

            float result = acc_weight > 0.0f ? (acc_val / acc_weight) : 0.0f;
            results[c] = result;
        }

        T result_vec;
        elem_type* result_ptr = reinterpret_cast<elem_type*>(&result_vec);
        for (int c = 0; c < cn; ++c)
        {
            result_ptr[c] = saturate_cast<elem_type>(results[c]);
        }
        dst(y, x) = result_vec;
    }

    template <typename T>
    void call_resize_lanczos4_tex(const PtrStepSz<T>& src, const PtrStepSz<T>& srcWhole, int yoff, int xoff, const PtrStepSz<T>& dst, float fy, float fx)
    {
        const dim3 block(32, 8);
        const dim3 grid(divUp(dst.cols, block.x), divUp(dst.rows, block.y));
        if (srcWhole.data == src.data)
        {
            cudev::Texture<T> texSrc(src);
            resize_lanczos4_tex<cudev::TexturePtr<T>><<<grid, block>>>(texSrc, dst, fy, fx);
        }
        else
        {
            cudev::TextureOff<T> texSrcWhole(srcWhole, yoff, xoff);
            BrdReplicate<T> brd(src.rows, src.cols);
            BorderReader<cudev::TextureOffPtr<T>, BrdReplicate<T>> brdSrc(texSrcWhole, brd);
            resize_lanczos4_tex<BorderReader<cudev::TextureOffPtr<T>, BrdReplicate<T>>><<<grid, block>>>(brdSrc, dst, fy, fx);
        }
        cudaSafeCall( cudaGetLastError() );
        cudaSafeCall( cudaDeviceSynchronize() );
    }

    // ResizeNearestDispatcher

    template <typename T> struct ResizeNearestDispatcher
    {
        static void call(const PtrStepSz<T>& src, const PtrStepSz<T>& /*srcWhole*/, int /*yoff*/, int /*xoff*/, const PtrStepSz<T>& dst, float fy, float fx, cudaStream_t stream)
        {
            call_resize_nearest_glob(src, dst, fy, fx, stream);
        }
    };

    template <typename T> struct SelectImplForNearest
    {
        static void call(const PtrStepSz<T>& src, const PtrStepSz<T>& srcWhole, int yoff, int xoff, const PtrStepSz<T>& dst, float fy, float fx, cudaStream_t stream)
        {
            if (stream)
                call_resize_nearest_glob(src, dst, fy, fx, stream);
            else
            {
                if (fx > 1 || fy > 1)
                    call_resize_nearest_glob(src, dst, fy, fx, 0);
                else
                   call_resize_nearest_tex(srcWhole, yoff, xoff, dst, fy, fx);
            }
        }
    };

    template <> struct ResizeNearestDispatcher<uchar> : SelectImplForNearest<uchar> {};
    template <> struct ResizeNearestDispatcher<uchar4> : SelectImplForNearest<uchar4> {};

    template <> struct ResizeNearestDispatcher<ushort> : SelectImplForNearest<ushort> {};
    template <> struct ResizeNearestDispatcher<ushort4> : SelectImplForNearest<ushort4> {};

    template <> struct ResizeNearestDispatcher<short> : SelectImplForNearest<short> {};
    template <> struct ResizeNearestDispatcher<short4> : SelectImplForNearest<short4> {};

    template <> struct ResizeNearestDispatcher<float> : SelectImplForNearest<float> {};
    template <> struct ResizeNearestDispatcher<float4> : SelectImplForNearest<float4> {};

    // ResizeLinearDispatcher

    template <typename T> struct ResizeLinearDispatcher
    {
        static void call(const PtrStepSz<T>& src, const PtrStepSz<T>& /*srcWhole*/, int /*yoff*/, int /*xoff*/, const PtrStepSz<T>& dst, float fy, float fx, cudaStream_t stream)
        {
            call_resize_linear_glob(src, dst, fy, fx, stream);
        }
    };

    template <typename T> struct SelectImplForLinear
    {
        static void call(const PtrStepSz<T>& src, const PtrStepSz<T>& srcWhole, int yoff, int xoff, const PtrStepSz<T>& dst, float fy, float fx, cudaStream_t stream)
        {
            if (stream)
                call_resize_linear_glob(src, dst, fy, fx, stream);
            else
            {
                if (fx > 1 || fy > 1)
                    call_resize_linear_glob(src, dst, fy, fx, 0);
                else
                    call_resize_linear_tex(src, srcWhole, yoff, xoff, dst, fy, fx);
            }
        }
    };

    template <> struct ResizeLinearDispatcher<uchar> : SelectImplForLinear<uchar> {};
    template <> struct ResizeLinearDispatcher<uchar4> : SelectImplForLinear<uchar4> {};

    template <> struct ResizeLinearDispatcher<ushort> : SelectImplForLinear<ushort> {};
    template <> struct ResizeLinearDispatcher<ushort4> : SelectImplForLinear<ushort4> {};

    template <> struct ResizeLinearDispatcher<short> : SelectImplForLinear<short> {};
    template <> struct ResizeLinearDispatcher<short4> : SelectImplForLinear<short4> {};

    template <> struct ResizeLinearDispatcher<float> : SelectImplForLinear<float> {};
    template <> struct ResizeLinearDispatcher<float4> : SelectImplForLinear<float4> {};

    // ResizeCubicDispatcher

    template <typename T> struct ResizeCubicDispatcher
    {
        static void call(const PtrStepSz<T>& src, const PtrStepSz<T>& /*srcWhole*/, int /*yoff*/, int /*xoff*/, const PtrStepSz<T>& dst, float fy, float fx, cudaStream_t stream)
        {
            call_resize_cubic_glob(src, dst, fy, fx, stream);
        }
    };

    template <typename T> struct SelectImplForCubic
    {
        static void call(const PtrStepSz<T>& src, const PtrStepSz<T>& srcWhole, int yoff, int xoff, const PtrStepSz<T>& dst, float fy, float fx, cudaStream_t stream)
        {
            if (stream)
                call_resize_cubic_glob(src, dst, fy, fx, stream);
           else
                call_resize_cubic_tex(src, srcWhole, yoff, xoff, dst, fy, fx);
        }
    };

    template <> struct ResizeCubicDispatcher<uchar> : SelectImplForCubic<uchar> {};
    template <> struct ResizeCubicDispatcher<uchar4> : SelectImplForCubic<uchar4> {};

    template <> struct ResizeCubicDispatcher<ushort> : SelectImplForCubic<ushort> {};
    template <> struct ResizeCubicDispatcher<ushort4> : SelectImplForCubic<ushort4> {};

    template <> struct ResizeCubicDispatcher<short> : SelectImplForCubic<short> {};
    template <> struct ResizeCubicDispatcher<short4> : SelectImplForCubic<short4> {};

    template <> struct ResizeCubicDispatcher<float> : SelectImplForCubic<float> {};
    template <> struct ResizeCubicDispatcher<float4> : SelectImplForCubic<float4> {};

    // ResizeLanczosDispatcher

    template <typename T> struct ResizeLanczosDispatcher
    {
        static void call(const PtrStepSz<T>& src, const PtrStepSz<T>& /*srcWhole*/, int /*yoff*/, int /*xoff*/, const PtrStepSz<T>& dst, float fy, float fx, cudaStream_t stream)
        {
            call_resize_lanczos4_glob(src, dst, fy, fx, stream);
        }
    };

    template <typename T> struct SelectImplForLanczos
    {
        static void call(const PtrStepSz<T>& src, const PtrStepSz<T>& srcWhole, int yoff, int xoff, const PtrStepSz<T>& dst, float fy, float fx, cudaStream_t stream)
        {
            if (stream)
                call_resize_lanczos4_glob(src, dst, fy, fx, stream);
            else
            {
                if (fx > 1 || fy > 1)
                    call_resize_lanczos4_glob(src, dst, fy, fx, 0);
                else
                    call_resize_lanczos4_tex(src, srcWhole, yoff, xoff, dst, fy, fx);
            }
        }
    };

    // Texture memory doesn't support 3-channel types, so use glob for those
    template <> struct ResizeLanczosDispatcher<uchar> : SelectImplForLanczos<uchar> {};
    template <> struct ResizeLanczosDispatcher<uchar3>
    {
        static void call(const PtrStepSz<uchar3>& src, const PtrStepSz<uchar3>& /*srcWhole*/, int /*yoff*/, int /*xoff*/, const PtrStepSz<uchar3>& dst, float fy, float fx, cudaStream_t stream)
        {
            call_resize_lanczos4_glob(src, dst, fy, fx, stream);
        }
    };
    template <> struct ResizeLanczosDispatcher<uchar4> : SelectImplForLanczos<uchar4> {};

    template <> struct ResizeLanczosDispatcher<ushort> : SelectImplForLanczos<ushort> {};
    template <> struct ResizeLanczosDispatcher<ushort3>
    {
        static void call(const PtrStepSz<ushort3>& src, const PtrStepSz<ushort3>& /*srcWhole*/, int /*yoff*/, int /*xoff*/, const PtrStepSz<ushort3>& dst, float fy, float fx, cudaStream_t stream)
        {
            call_resize_lanczos4_glob(src, dst, fy, fx, stream);
        }
    };
    template <> struct ResizeLanczosDispatcher<ushort4> : SelectImplForLanczos<ushort4> {};

    template <> struct ResizeLanczosDispatcher<short> : SelectImplForLanczos<short> {};
    template <> struct ResizeLanczosDispatcher<short3>
    {
        static void call(const PtrStepSz<short3>& src, const PtrStepSz<short3>& /*srcWhole*/, int /*yoff*/, int /*xoff*/, const PtrStepSz<short3>& dst, float fy, float fx, cudaStream_t stream)
        {
            call_resize_lanczos4_glob(src, dst, fy, fx, stream);
        }
    };
    template <> struct ResizeLanczosDispatcher<short4> : SelectImplForLanczos<short4> {};

    template <> struct ResizeLanczosDispatcher<float> : SelectImplForLanczos<float> {};
    template <> struct ResizeLanczosDispatcher<float3>
    {
        static void call(const PtrStepSz<float3>& src, const PtrStepSz<float3>& /*srcWhole*/, int /*yoff*/, int /*xoff*/, const PtrStepSz<float3>& dst, float fy, float fx, cudaStream_t stream)
        {
            call_resize_lanczos4_glob(src, dst, fy, fx, stream);
        }
    };
    template <> struct ResizeLanczosDispatcher<float4> : SelectImplForLanczos<float4> {};

    // ResizeAreaDispatcher

    template <typename T> struct ResizeAreaDispatcher
    {
        static void call(const PtrStepSz<T>& src, const PtrStepSz<T>&, int, int, const PtrStepSz<T>& dst, float fy, float fx, cudaStream_t stream)
        {
            const int iscale_x = (int) round(fx);
            const int iscale_y = (int) round(fy);

            const dim3 block(32, 8);
            const dim3 grid(divUp(dst.cols, block.x), divUp(dst.rows, block.y));

            if (std::abs(fx - iscale_x) < FLT_MIN && std::abs(fy - iscale_y) < FLT_MIN)
            {
                BrdConstant<T> brd(src.rows, src.cols);
                BorderReader<PtrStep<T>, BrdConstant<T>> brdSrc(src, brd);
                IntegerAreaFilter<BorderReader< PtrStep<T>, BrdConstant<T>>> filteredSrc(brdSrc, fx, fy);

                resize_area<<<grid, block, 0, stream>>>(filteredSrc, dst);
            }
            else
            {
                BrdConstant<T> brd(src.rows, src.cols);
                BorderReader<PtrStep<T>, BrdConstant<T>> brdSrc(src, brd);
                AreaFilter<BorderReader< PtrStep<T>, BrdConstant<T>>> filteredSrc(brdSrc, fx, fy);

                resize_area<<<grid, block, 0, stream>>>(filteredSrc, dst);
            }

            cudaSafeCall( cudaGetLastError() );

            if (stream == 0)
                cudaSafeCall( cudaDeviceSynchronize() );
        }
    };

    // resize

    template <typename T> void resize(const PtrStepSzb& src, const PtrStepSzb& srcWhole, int yoff, int xoff, const PtrStepSzb& dst, float fy, float fx, int interpolation, cudaStream_t stream)
    {
        typedef void (*func_t)(const PtrStepSz<T>& src, const PtrStepSz<T>& srcWhole, int yoff, int xoff, const PtrStepSz<T>& dst, float fy, float fx, cudaStream_t stream);
        static const func_t funcs[5] =
        {
            ResizeNearestDispatcher<T>::call,
            ResizeLinearDispatcher<T>::call,
            ResizeCubicDispatcher<T>::call,
            ResizeAreaDispatcher<T>::call,
            ResizeLanczosDispatcher<T>::call
        };

        // change to linear if area interpolation upscaling
        if (interpolation == 3 && (fx <= 1.f || fy <= 1.f))
            interpolation = 1;

        // Bounds check for interpolation mode
        CV_Assert(interpolation >= 0 && interpolation < 5);

        funcs[interpolation](static_cast< PtrStepSz<T> >(src), static_cast< PtrStepSz<T> >(srcWhole), yoff, xoff, static_cast< PtrStepSz<T> >(dst), fy, fx, stream);
    }

    template void resize<uchar >(const PtrStepSzb& src, const PtrStepSzb& srcWhole, int yoff, int xoff, const PtrStepSzb& dst, float fy, float fx, int interpolation, cudaStream_t stream);
    template void resize<uchar3>(const PtrStepSzb& src, const PtrStepSzb& srcWhole, int yoff, int xoff, const PtrStepSzb& dst, float fy, float fx, int interpolation, cudaStream_t stream);
    template void resize<uchar4>(const PtrStepSzb& src, const PtrStepSzb& srcWhole, int yoff, int xoff, const PtrStepSzb& dst, float fy, float fx, int interpolation, cudaStream_t stream);

    template void resize<ushort >(const PtrStepSzb& src, const PtrStepSzb& srcWhole, int yoff, int xoff, const PtrStepSzb& dst, float fy, float fx, int interpolation, cudaStream_t stream);
    template void resize<ushort3>(const PtrStepSzb& src, const PtrStepSzb& srcWhole, int yoff, int xoff, const PtrStepSzb& dst, float fy, float fx, int interpolation, cudaStream_t stream);
    template void resize<ushort4>(const PtrStepSzb& src, const PtrStepSzb& srcWhole, int yoff, int xoff, const PtrStepSzb& dst, float fy, float fx, int interpolation, cudaStream_t stream);

    template void resize<short >(const PtrStepSzb& src, const PtrStepSzb& srcWhole, int yoff, int xoff, const PtrStepSzb& dst, float fy, float fx, int interpolation, cudaStream_t stream);
    template void resize<short3>(const PtrStepSzb& src, const PtrStepSzb& srcWhole, int yoff, int xoff, const PtrStepSzb& dst, float fy, float fx, int interpolation, cudaStream_t stream);
    template void resize<short4>(const PtrStepSzb& src, const PtrStepSzb& srcWhole, int yoff, int xoff, const PtrStepSzb& dst, float fy, float fx, int interpolation, cudaStream_t stream);

    template void resize<float >(const PtrStepSzb& src, const PtrStepSzb& srcWhole, int yoff, int xoff, const PtrStepSzb& dst, float fy, float fx, int interpolation, cudaStream_t stream);
    template void resize<float3>(const PtrStepSzb& src, const PtrStepSzb& srcWhole, int yoff, int xoff, const PtrStepSzb& dst, float fy, float fx, int interpolation, cudaStream_t stream);
    template void resize<float4>(const PtrStepSzb& src, const PtrStepSzb& srcWhole, int yoff, int xoff, const PtrStepSzb& dst, float fy, float fx, int interpolation, cudaStream_t stream);
}}}

#endif /* CUDA_DISABLER */
