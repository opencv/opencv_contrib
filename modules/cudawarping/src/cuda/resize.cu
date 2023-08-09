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
#include "opencv2/core/cuda/common.hpp"
#include "opencv2/core/cuda/border_interpolate.hpp"
#include "opencv2/core/cuda/vec_traits.hpp"
#include "opencv2/core/cuda/vec_math.hpp"
#include "opencv2/core/cuda/saturate_cast.hpp"
#include "opencv2/core/cuda/filters.hpp"
#include <opencv2/cudev/ptr2d/texture.hpp>
#include <opencv2/imgproc.hpp>

namespace cv
{
    static void interpolateCoordinate(int coordinate, int dst, int src, float scale, float& a, float& b)
    {
        if (coordinate == INTER_HALF_PIXEL
            || coordinate == INTER_HALF_PIXEL_SYMMETRIC
            || coordinate == INTER_HALF_PIXEL_PYTORCH)
        {
            a = scale;
            b = 0.5f * scale - 0.5f;
            if (coordinate == INTER_HALF_PIXEL_SYMMETRIC)
                b += 0.5f * (src - dst * scale);
            if (coordinate == INTER_HALF_PIXEL_PYTORCH && dst <= 1)
                a = b = 0.f;
        }
        else if (coordinate == INTER_ALIGN_CORNERS)
        {
            a = (src - 1.f) / (dst - 1.f);
            b = 0.f;
        }
        else if (coordinate == INTER_ASYMMETRIC)
        {
            a = scale;
            b = 0.f;
        }
        else
            CV_Error(Error::StsBadArg, "Unknown coordinate transformation mode");
    }
}

namespace cv { namespace cuda { namespace device
{
    // kernels

    template <typename T> __global__ void resize_nearest(const PtrStepSz<T> src, PtrStepSz<T> dst, const float a_y, const float b_y, const float a_x, const float b_x)
    {
        const int dst_x = blockDim.x * blockIdx.x + threadIdx.x;
        const int dst_y = blockDim.y * blockIdx.y + threadIdx.y;

        if (dst_x < dst.cols && dst_y < dst.rows)
        {
            const float src_x = dst_x * a_x + b_x;
            const float src_y = dst_y * a_y + b_y;
            const float ix = ::min(::max(__float2int_rd(src_x), 0), src.cols - 1);
            const float iy = ::min(::max(__float2int_rd(src_y), 0), src.rows - 1);

            dst(dst_y, dst_x) = src(iy, ix);
        }
    }

    template <typename T> __global__ void resize_linear(const PtrStepSz<T> src, PtrStepSz<T> dst, const float a_y, const float b_y, const float a_x, const float b_x)
    {
        typedef typename TypeVec<float, VecTraits<T>::cn>::vec_type work_type;

        const int dst_x = blockDim.x * blockIdx.x + threadIdx.x;
        const int dst_y = blockDim.y * blockIdx.y + threadIdx.y;

        if (dst_x < dst.cols && dst_y < dst.rows)
        {
            const float src_x = dst_x * a_x + b_x;
            const float src_y = dst_y * a_y + b_y;
            const int ix = __float2int_rd(src_x);
            const int iy = __float2int_rd(src_y);
            const float fx2 = src_x - ix;
            const float fx1 = 1.f - fx2;
            const float fy2 = src_y - iy;
            const float fy1 = 1.f - fy2;
            const int ix1 = ::max(ix, 0);
            const int ix2 = ::min(ix + 1, src.cols - 1);
            const int iy1 = ::max(iy, 0);
            const int iy2 = ::min(iy + 1, src.rows - 1);

            work_type out = VecTraits<work_type>::all(0);
            out = out + src(iy1, ix1) * (fy1 * fx1);
            out = out + src(iy1, ix2) * (fy1 * fx2);
            out = out + src(iy2, ix1) * (fy2 * fx1);
            out = out + src(iy2, ix2) * (fy2 * fx2);
            dst(dst_y, dst_x) = saturate_cast<T>(out);
        }
    }

    template <class Ptr2D, typename T> __global__ void resize(Ptr2D src, PtrStepSz<T> dst, const float a_y, const float b_y, const float a_x, const float b_x)
    {
        const int dst_x = blockDim.x * blockIdx.x + threadIdx.x;
        const int dst_y = blockDim.y * blockIdx.y + threadIdx.y;

        if (dst_x < dst.cols && dst_y < dst.rows)
        {
            const float src_x = dst_x * a_x + b_x;
            const float src_y = dst_y * a_y + b_y;
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

    template <typename T> __global__ void resize_area_larger(const PtrStepSz<T> src, PtrStepSz<T> dst, float fy, float fx)
    {
        typedef typename TypeVec<float, VecTraits<T>::cn>::vec_type work_type;

        const int dst_x = blockDim.x * blockIdx.x + threadIdx.x;
        const int dst_y = blockDim.y * blockIdx.y + threadIdx.y;

        if (dst_x < dst.cols && dst_y < dst.rows)
        {
            // cpu resize:
            //      sx = cvFloor(dx*scale_x);
            //      fx = (float)((dx+1) - (sx+1)*inv_scale_x);
            //      fx = fx <= 0 ? 0.f : fx - cvFloor(fx);
            const int src_x = __float2int_rd(dst_x * fx);
            const int src_y = __float2int_rd(dst_y * fy);
            float fx2 = dst_x + 1 - (src_x + 1) / fx;
            fx2 = fx2 <= 0.f ? 0.f : fx2 - ::floor(fx2);
            const float fx1 = 1.f - fx2;
            float fy2 = dst_y + 1 - (src_y + 1) / fy;
            fy2 = fy2 <= 0.f ? 0.f : fy2 - ::floor(fy2);
            const float fy1 = 1.f - fy2;
            const int ix1 = src_x;
            const int ix2 = ::min(src_x + 1, src.cols - 1);
            const int iy1 = src_y;
            const int iy2 = ::min(src_y + 1, src.rows - 1);

            work_type out = VecTraits<work_type>::all(0);
            out = out + src(iy1, ix1) * (fy1 * fx1);
            out = out + src(iy1, ix2) * (fy1 * fx2);
            out = out + src(iy2, ix1) * (fy2 * fx1);
            out = out + src(iy2, ix2) * (fy2 * fx2);
            dst(dst_y, dst_x) = saturate_cast<T>(out);
        }
    }


    // callers for nearest interpolation

    template <typename T>
    void call_resize_nearest_glob(const PtrStepSz<T>& src, const PtrStepSz<T>& dst, float a_y, float b_y, float a_x, float b_x, cudaStream_t stream)
    {
        const dim3 block(32, 8);
        const dim3 grid(divUp(dst.cols, block.x), divUp(dst.rows, block.y));

        // nearest use floor(x_org + 0.5), so b plus 0.5 here, to use __float2int_rd directly
        resize_nearest<<<grid, block, 0, stream>>>(src, dst, a_y, b_y + 0.5f, a_x, b_x + 0.5f);
        cudaSafeCall( cudaGetLastError() );

        if (stream == 0)
            cudaSafeCall( cudaDeviceSynchronize() );
    }

    template <typename T>
    void call_resize_nearest_tex(const PtrStepSz<T>& srcWhole, int yoff, int xoff, const PtrStepSz<T>& dst, float a_y, float b_y, float a_x, float b_x)
    {
        const dim3 block(32, 8);
        const dim3 grid(divUp(dst.cols, block.x), divUp(dst.rows, block.y));
        // https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#nearest-point-sampling
        // cudaFilterModePoint: tex(x) = T[floor(x)], so with b + 0.5, we can use `resize` directly
        if (xoff || yoff) {
            cudev::TextureOff<T> texSrcWhole(srcWhole, yoff, xoff);
            resize<cudev::TextureOffPtr<T>><<<grid, block>>>(texSrcWhole, dst, a_y, b_y + 0.5f, a_x, b_x + 0.5f);
        }
        else {
            cudev::Texture<T> texSrcWhole(srcWhole);
            resize<cudev::TexturePtr<T>><<<grid, block>>>(texSrcWhole, dst, a_y, b_y + 0.5f, a_x, b_x + 0.5f);
        }
        cudaSafeCall( cudaGetLastError() );
        cudaSafeCall( cudaDeviceSynchronize() );
    }

    // callers for linear interpolation

    template <typename T>
    void call_resize_linear_glob(const PtrStepSz<T>& src, const PtrStepSz<T>& dst, float a_y, float b_y, float a_x, float b_x, cudaStream_t stream)
    {
        const dim3 block(32, 8);
        const dim3 grid(divUp(dst.cols, block.x), divUp(dst.rows, block.y));

        resize_linear<<<grid, block, 0, stream>>>(src, dst, a_y, b_y, a_x, b_x);
        cudaSafeCall( cudaGetLastError() );

        if (stream == 0)
            cudaSafeCall( cudaDeviceSynchronize() );
    }

    template <typename T>
    void call_resize_linear_tex(const PtrStepSz<T>& src, const PtrStepSz<T>& srcWhole, int yoff, int xoff, const PtrStepSz<T>& dst, float a_y, float b_y, float a_x, float b_x)
    {
        const dim3 block(32, 8);
        const dim3 grid(divUp(dst.cols, block.x), divUp(dst.rows, block.y));
        // tex use 0.5-base coordinate, so b plus 0.5
        if (srcWhole.data == src.data)
        {
            cudev::Texture<T> texSrc(src);
            LinearFilter<cudev::TexturePtr<T>> filteredSrc(texSrc);
            resize<<<grid, block>>>(filteredSrc, dst, a_y, b_y + 0.5f, a_x, b_x + 0.5f);
        }
        else
        {
            cudev::TextureOff<T> texSrcWhole(srcWhole, yoff, xoff);
            BrdReplicate<T> brd(src.rows, src.cols);
            BorderReader<cudev::TextureOffPtr<T>, BrdReplicate<T>> brdSrc(texSrcWhole, brd);
            LinearFilter<BorderReader<cudev::TextureOffPtr<T>, BrdReplicate<T>>> filteredSrc(brdSrc);
            resize<<<grid, block>>>(filteredSrc, dst, a_y, b_y + 0.5f, a_x, b_x + 0.5f);
        }
        cudaSafeCall( cudaGetLastError() );
        cudaSafeCall( cudaDeviceSynchronize() );
    }

    // callers for cubic interpolation

    template <typename T>
    void call_resize_cubic_glob(const PtrStepSz<T>& src, const PtrStepSz<T>& dst, float a_y, float b_y, float a_x, float b_x, cudaStream_t stream)
    {
        const dim3 block(32, 8);
        const dim3 grid(divUp(dst.cols, block.x), divUp(dst.rows, block.y));

        BrdReplicate<T> brd(src.rows, src.cols);
        BorderReader<PtrStep<T>, BrdReplicate<T>> brdSrc(src, brd);
        CubicFilter<BorderReader< PtrStep<T>, BrdReplicate<T>>> filteredSrc(brdSrc);

        resize<<<grid, block, 0, stream>>>(filteredSrc, dst, a_y, b_y, a_x, b_x);
        cudaSafeCall( cudaGetLastError() );

        if (stream == 0)
            cudaSafeCall( cudaDeviceSynchronize() );
    }

    template <typename T>
    void call_resize_cubic_tex(const PtrStepSz<T>& src, const PtrStepSz<T>& srcWhole, int yoff, int xoff, const PtrStepSz<T>& dst, float a_y, float b_y, float a_x, float b_x)
    {
        const dim3 block(32, 8);
        const dim3 grid(divUp(dst.cols, block.x), divUp(dst.rows, block.y));
        // tex use 0.5-base coordinate, so b plus 0.5
        if (srcWhole.data == src.data)
        {
            cudev::Texture<T> texSrc(src);
            CubicFilter<cudev::TexturePtr<T>> filteredSrc(texSrc);
            resize<<<grid, block>>>(filteredSrc, dst, a_y, b_y + 0.5f, a_x, b_x + 0.5f);
        }
        else
        {
            cudev::TextureOff<T> texSrcWhole(srcWhole, yoff, xoff);
            BrdReplicate<T> brd(src.rows, src.cols);
            BorderReader<cudev::TextureOffPtr<T>, BrdReplicate<T>> brdSrc(texSrcWhole, brd);
            CubicFilter<BorderReader<cudev::TextureOffPtr<T>, BrdReplicate<T>>> filteredSrc(brdSrc);
            resize<<<grid, block>>>(filteredSrc, dst, a_y, b_y + 0.5f, a_x, b_x + 0.5f);
        }
        cudaSafeCall( cudaGetLastError() );
        cudaSafeCall( cudaDeviceSynchronize() );
    }

    // ResizeNearestDispatcher

    template <typename T> struct ResizeNearestDispatcher
    {
        static void call(const PtrStepSz<T>& src, const PtrStepSz<T>& /*srcWhole*/, int /*yoff*/, int /*xoff*/, const PtrStepSz<T>& dst, float a_y, float b_y, float a_x, float b_x, cudaStream_t stream)
        {
            call_resize_nearest_glob(src, dst, a_y, b_y, a_x, b_x, stream);
        }
    };

    template <typename T> struct SelectImplForNearest
    {
        static void call(const PtrStepSz<T>& src, const PtrStepSz<T>& srcWhole, int yoff, int xoff, const PtrStepSz<T>& dst, float a_y, float b_y, float a_x, float b_x, cudaStream_t stream)
        {
            if (stream)
                call_resize_nearest_glob(src, dst, a_y, b_y, a_x, b_x, stream);
            else
            {
                if (a_x > 1 || a_y > 1)
                    call_resize_nearest_glob(src, dst, a_y, b_y, a_x, b_x, 0);
                else
                   call_resize_nearest_tex(srcWhole, yoff, xoff, dst, a_y, b_y, a_x, b_x);
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
        static void call(const PtrStepSz<T>& src, const PtrStepSz<T>& /*srcWhole*/, int /*yoff*/, int /*xoff*/, const PtrStepSz<T>& dst, float a_y, float b_y, float a_x, float b_x, cudaStream_t stream)
        {
            call_resize_linear_glob(src, dst, a_y, b_y, a_x, b_x, stream);
        }
    };

    template <typename T> struct SelectImplForLinear
    {
        static void call(const PtrStepSz<T>& src, const PtrStepSz<T>& srcWhole, int yoff, int xoff, const PtrStepSz<T>& dst, float a_y, float b_y, float a_x, float b_x, cudaStream_t stream)
        {
            if (stream)
                call_resize_linear_glob(src, dst, a_y, b_y, a_x, b_x, stream);
            else
            {
                if (a_x > 1 || a_y > 1)
                    call_resize_linear_glob(src, dst, a_y, b_y, a_x, b_x, 0);
                else
                    call_resize_linear_tex(src, srcWhole, yoff, xoff, dst, a_y, b_y, a_x, b_x);
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
        static void call(const PtrStepSz<T>& src, const PtrStepSz<T>& /*srcWhole*/, int /*yoff*/, int /*xoff*/, const PtrStepSz<T>& dst, float a_y, float b_y, float a_x, float b_x, cudaStream_t stream)
        {
            call_resize_cubic_glob(src, dst, a_y, b_y, a_x, b_x, stream);
        }
    };

    template <typename T> struct SelectImplForCubic
    {
        static void call(const PtrStepSz<T>& src, const PtrStepSz<T>& srcWhole, int yoff, int xoff, const PtrStepSz<T>& dst, float a_y, float b_y, float a_x, float b_x, cudaStream_t stream)
        {
            if (stream)
                call_resize_cubic_glob(src, dst, a_y, b_y, a_x, b_x, stream);
            else
                call_resize_cubic_tex(src, srcWhole, yoff, xoff, dst, a_y, b_y, a_x, b_x);
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

    // ResizeAreaDispatcher

    template <typename T> struct ResizeAreaDispatcher
    {
        static void call(const PtrStepSz<T>& src, const PtrStepSz<T>&, int, int, const PtrStepSz<T>& dst, float fy, float fx, cudaStream_t stream)
        {
            const int iscale_x = (int) round(fx);
            const int iscale_y = (int) round(fy);

            const dim3 block(32, 8);
            const dim3 grid(divUp(dst.cols, block.x), divUp(dst.rows, block.y));

            if (fy <= 1.f || fx <= 1.f)
            {
                resize_area_larger<<<grid, block, 0, stream>>>(src, dst, fy, fx);
                return;
            }

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

    template <typename T> void resize(const PtrStepSzb& src, const PtrStepSzb& srcWhole, int yoff, int xoff, const PtrStepSzb& dst, float fy, float fx, int interpolation, int coordinate, cudaStream_t stream)
    {
        typedef void (*func_t)(const PtrStepSz<T>& src, const PtrStepSz<T>& srcWhole, int yoff, int xoff, const PtrStepSz<T>& dst, float a_y, float b_y, float a_x, float b_x, cudaStream_t stream);
        static const func_t funcs[3] =
        {
            ResizeNearestDispatcher<T>::call,
            ResizeLinearDispatcher<T>::call,
            ResizeCubicDispatcher<T>::call,
        };

        if (interpolation == INTER_AREA)
        {
            ResizeAreaDispatcher<T>::call(static_cast<PtrStepSz<T>>(src), static_cast<PtrStepSz<T>>(srcWhole), yoff, xoff, static_cast<PtrStepSz<T>>(dst), fy, fx, stream);
            return;
        }

        float a_x, b_x, a_y, b_y;
        interpolateCoordinate(coordinate, dst.cols, src.cols, fx, a_x, b_x);
        interpolateCoordinate(coordinate, dst.rows, src.rows, fy, a_y, b_y);

        funcs[interpolation](static_cast< PtrStepSz<T> >(src), static_cast< PtrStepSz<T> >(srcWhole), yoff, xoff, static_cast< PtrStepSz<T> >(dst), a_y, b_y, a_x, b_x, stream);
    }

    template void resize<uchar >(const PtrStepSzb& src, const PtrStepSzb& srcWhole, int yoff, int xoff, const PtrStepSzb& dst, float fy, float fx, int interpolation, int coordinate, cudaStream_t stream);
    template void resize<uchar3>(const PtrStepSzb& src, const PtrStepSzb& srcWhole, int yoff, int xoff, const PtrStepSzb& dst, float fy, float fx, int interpolation, int coordinate, cudaStream_t stream);
    template void resize<uchar4>(const PtrStepSzb& src, const PtrStepSzb& srcWhole, int yoff, int xoff, const PtrStepSzb& dst, float fy, float fx, int interpolation, int coordinate, cudaStream_t stream);

    template void resize<ushort >(const PtrStepSzb& src, const PtrStepSzb& srcWhole, int yoff, int xoff, const PtrStepSzb& dst, float fy, float fx, int interpolation, int coordinate, cudaStream_t stream);
    template void resize<ushort3>(const PtrStepSzb& src, const PtrStepSzb& srcWhole, int yoff, int xoff, const PtrStepSzb& dst, float fy, float fx, int interpolation, int coordinate, cudaStream_t stream);
    template void resize<ushort4>(const PtrStepSzb& src, const PtrStepSzb& srcWhole, int yoff, int xoff, const PtrStepSzb& dst, float fy, float fx, int interpolation, int coordinate, cudaStream_t stream);

    template void resize<short >(const PtrStepSzb& src, const PtrStepSzb& srcWhole, int yoff, int xoff, const PtrStepSzb& dst, float fy, float fx, int interpolation, int coordinate, cudaStream_t stream);
    template void resize<short3>(const PtrStepSzb& src, const PtrStepSzb& srcWhole, int yoff, int xoff, const PtrStepSzb& dst, float fy, float fx, int interpolation, int coordinate, cudaStream_t stream);
    template void resize<short4>(const PtrStepSzb& src, const PtrStepSzb& srcWhole, int yoff, int xoff, const PtrStepSzb& dst, float fy, float fx, int interpolation, int coordinate, cudaStream_t stream);

    template void resize<float >(const PtrStepSzb& src, const PtrStepSzb& srcWhole, int yoff, int xoff, const PtrStepSzb& dst, float fy, float fx, int interpolation, int coordinate, cudaStream_t stream);
    template void resize<float3>(const PtrStepSzb& src, const PtrStepSzb& srcWhole, int yoff, int xoff, const PtrStepSzb& dst, float fy, float fx, int interpolation, int coordinate, cudaStream_t stream);
    template void resize<float4>(const PtrStepSzb& src, const PtrStepSzb& srcWhole, int yoff, int xoff, const PtrStepSzb& dst, float fy, float fx, int interpolation, int coordinate, cudaStream_t stream);
}}}

#endif /* CUDA_DISABLER */
