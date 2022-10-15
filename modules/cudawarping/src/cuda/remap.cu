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
#include "opencv2/core/cuda/border_interpolate.hpp"
#include "opencv2/core/cuda/vec_traits.hpp"
#include "opencv2/core/cuda/vec_math.hpp"
#include "opencv2/core/cuda/saturate_cast.hpp"
#include "opencv2/core/cuda/filters.hpp"
#include <opencv2/cudev/ptr2d/texture.hpp>

namespace cv { namespace cuda { namespace device
{
    namespace imgproc
    {
        template <typename Ptr2D, typename T> __global__ void remap(const Ptr2D src, const PtrStepf mapx, const PtrStepf mapy, PtrStepSz<T> dst)
        {
            const int x = blockDim.x * blockIdx.x + threadIdx.x;
            const int y = blockDim.y * blockIdx.y + threadIdx.y;

            if (x < dst.cols && y < dst.rows)
            {
                const float xcoo = mapx.ptr(y)[x];
                const float ycoo = mapy.ptr(y)[x];

                dst.ptr(y)[x] = saturate_cast<T>(src(ycoo, xcoo));
            }
        }

        template <template <typename> class Filter, template <typename> class B, typename T> struct RemapDispatcherStream
        {
            static void call(PtrStepSz<T> src, PtrStepSzf mapx, PtrStepSzf mapy, PtrStepSz<T> dst, const float* borderValue, cudaStream_t stream, bool)
            {
                typedef typename TypeVec<float, VecTraits<T>::cn>::vec_type work_type;

                dim3 block(32, 8);
                dim3 grid(divUp(dst.cols, block.x), divUp(dst.rows, block.y));

                B<work_type> brd(src.rows, src.cols, VecTraits<work_type>::make(borderValue));
                BorderReader<PtrStep<T>, B<work_type>> brdSrc(src, brd);
                Filter<BorderReader<PtrStep<T>, B<work_type>>> filter_src(brdSrc);

                remap<<<grid, block, 0, stream>>>(filter_src, mapx, mapy, dst);
                cudaSafeCall( cudaGetLastError() );
            }
        };

        template <template <typename> class Filter, template <typename> class B, typename T> struct RemapDispatcherNonStream
        {
            static void call(PtrStepSz<T> src, PtrStepSz<T> srcWhole, int xoff, int yoff, PtrStepSzf mapx, PtrStepSzf mapy, PtrStepSz<T> dst, const float* borderValue, bool)
            {
                CV_UNUSED(srcWhole);
                CV_UNUSED(xoff);
                CV_UNUSED(yoff);
                typedef typename TypeVec<float, VecTraits<T>::cn>::vec_type work_type;

                dim3 block(32, 8);
                dim3 grid(divUp(dst.cols, block.x), divUp(dst.rows, block.y));

                B<work_type> brd(src.rows, src.cols, VecTraits<work_type>::make(borderValue));
                BorderReader<PtrStep<T>, B<work_type>> brdSrc(src, brd);
                Filter<BorderReader<PtrStep<T>, B<work_type>>> filter_src(brdSrc);

                remap<<<grid, block>>>(filter_src, mapx, mapy, dst);
                cudaSafeCall( cudaGetLastError() );

                cudaSafeCall( cudaDeviceSynchronize() );
            }
        };

        template <template <typename> class Filter, template <typename> class B, typename T> struct RemapDispatcherNonStreamTex
        {
            static void call(PtrStepSz< T > src, PtrStepSz< T > srcWhole, int xoff, int yoff, PtrStepSzf mapx, PtrStepSzf mapy,
                PtrStepSz< T > dst, const float* borderValue, bool cc20)
            {
                typedef typename TypeVec<float, VecTraits< T >::cn>::vec_type work_type;
                dim3 block(32, cc20 ? 8 : 4);
                dim3 grid(divUp(dst.cols, block.x), divUp(dst.rows, block.y));
                if (srcWhole.cols == src.cols && srcWhole.rows == src.rows)
                {
                    cudev::Texture<T> texSrcWhole(srcWhole);
                    B<work_type> brd(src.rows, src.cols, VecTraits<work_type>::make(borderValue));
                    BorderReader<cudev::TexturePtr<T>, B<work_type>> brdSrc(texSrcWhole, brd);
                    Filter<BorderReader<cudev::TexturePtr<T>, B<work_type>>> filter_src(brdSrc);
                    remap<<<grid, block>>>(filter_src, mapx, mapy, dst);

                }
                else {
                    cudev::TextureOff<T> texSrcWhole(srcWhole, yoff, xoff);
                    B<work_type> brd(src.rows, src.cols, VecTraits<work_type>::make(borderValue));
                    BorderReader<cudev::TextureOffPtr<T>, B<work_type>> brdSrc(texSrcWhole, brd);
                    Filter<BorderReader<cudev::TextureOffPtr<T>, B<work_type>>> filter_src(brdSrc);
                    remap<<<grid, block >>>(filter_src, mapx, mapy, dst);
                }

                cudaSafeCall( cudaGetLastError() );
                cudaSafeCall( cudaDeviceSynchronize() );
            }
        };

        template <template <typename> class Filter, typename T> struct RemapDispatcherNonStreamTex<Filter, BrdReplicate, T>
        {
            static void call(PtrStepSz< T > src, PtrStepSz< T > srcWhole, int xoff, int yoff, PtrStepSzf mapx, PtrStepSzf mapy,
                PtrStepSz< T > dst, const float*, bool)
            {
                dim3 block(32, 8);
                dim3 grid(divUp(dst.cols, block.x), divUp(dst.rows, block.y));
                if (srcWhole.cols == src.cols && srcWhole.rows == src.rows)
                {
                    cudev::Texture<T> texSrcWhole(srcWhole);
                    Filter<cudev::TexturePtr<T>> filter_src(texSrcWhole);
                    remap<<<grid, block>>>(filter_src, mapx, mapy, dst);
                }
                else
                {
                    cudev::TextureOff<T> texSrcWhole(srcWhole, yoff, xoff);
                    BrdReplicate<T> brd(src.rows, src.cols);
                    BorderReader<cudev::TextureOffPtr<T>, BrdReplicate<T>> brdSrc(texSrcWhole, brd);
                    Filter<BorderReader<cudev::TextureOffPtr<T>, BrdReplicate<T>>> filter_src(brdSrc);
                    remap<<<grid, block>>>(filter_src, mapx, mapy, dst);
                }
                cudaSafeCall( cudaGetLastError() );
                cudaSafeCall( cudaDeviceSynchronize() );
            }
        };


        template <template <typename> class Filter, template <typename> class B> struct RemapDispatcherNonStream<Filter, B, uchar> :
            RemapDispatcherNonStreamTex<Filter, B, uchar> {};
        template <template <typename> class Filter, template <typename> class B> struct RemapDispatcherNonStream<Filter, B, uchar4> :
            RemapDispatcherNonStreamTex<Filter, B, uchar4> {};
        template <template <typename> class Filter, template <typename> class B> struct RemapDispatcherNonStream<Filter, B, ushort> :
            RemapDispatcherNonStreamTex<Filter, B, ushort> {};
        template <template <typename> class Filter, template <typename> class B> struct RemapDispatcherNonStream<Filter, B, ushort4> :
            RemapDispatcherNonStreamTex<Filter, B, ushort4> {};
        template <template <typename> class Filter, template <typename> class B> struct RemapDispatcherNonStream<Filter, B, short> :
            RemapDispatcherNonStreamTex<Filter, B, short> {};
        template <template <typename> class Filter, template <typename> class B> struct RemapDispatcherNonStream<Filter, B, short4> :
            RemapDispatcherNonStreamTex<Filter, B, short4> {};
        template <template <typename> class Filter, template <typename> class B> struct RemapDispatcherNonStream<Filter, B, float> :
            RemapDispatcherNonStreamTex<Filter, B, float> {};
        template <template <typename> class Filter, template <typename> class B> struct RemapDispatcherNonStream<Filter, B, float4> :
            RemapDispatcherNonStreamTex<Filter, B, float4> {};

        template <template <typename> class Filter> struct RemapDispatcherNonStream<Filter, BrdReplicate, uchar> :
            RemapDispatcherNonStreamTex<Filter, BrdReplicate, uchar> {};
        template <template <typename> class Filter> struct RemapDispatcherNonStream<Filter, BrdReplicate, uchar4> :
            RemapDispatcherNonStreamTex<Filter, BrdReplicate, uchar4> {};
        template <template <typename> class Filter> struct RemapDispatcherNonStream<Filter, BrdReplicate, ushort> :
            RemapDispatcherNonStreamTex<Filter, BrdReplicate, ushort> {};
        template <template <typename> class Filter> struct RemapDispatcherNonStream<Filter, BrdReplicate, ushort4> :
            RemapDispatcherNonStreamTex<Filter, BrdReplicate, ushort4> {};
        template <template <typename> class Filter> struct RemapDispatcherNonStream<Filter, BrdReplicate, short> :
            RemapDispatcherNonStreamTex<Filter, BrdReplicate, short> {};
        template <template <typename> class Filter> struct RemapDispatcherNonStream<Filter, BrdReplicate, short4> :
            RemapDispatcherNonStreamTex<Filter, BrdReplicate, short4> {};
        template <template <typename> class Filter> struct RemapDispatcherNonStream<Filter, BrdReplicate, float> :
            RemapDispatcherNonStreamTex<Filter, BrdReplicate, float> {};
        template <template <typename> class Filter> struct RemapDispatcherNonStream<Filter, BrdReplicate, float4> :
            RemapDispatcherNonStreamTex<Filter, BrdReplicate, float4> {};

        template <template <typename> class Filter, template <typename> class B, typename T> struct RemapDispatcher
        {
            static void call(PtrStepSz<T> src, PtrStepSz<T> srcWhole, int xoff, int yoff, PtrStepSzf mapx, PtrStepSzf mapy,
                PtrStepSz<T> dst, const float* borderValue, cudaStream_t stream, bool cc20)
            {
                if (stream == 0)
                    RemapDispatcherNonStream<Filter, B, T>::call(src, srcWhole, xoff, yoff, mapx, mapy, dst, borderValue, cc20);
                else
                    RemapDispatcherStream<Filter, B, T>::call(src, mapx, mapy, dst, borderValue, stream, cc20);
            }
        };

        template <typename T> void remap_gpu(PtrStepSzb src, PtrStepSzb srcWhole, int xoff, int yoff, PtrStepSzf xmap, PtrStepSzf ymap,
            PtrStepSzb dst, int interpolation, int borderMode, const float* borderValue, cudaStream_t stream, bool cc20)
        {
            typedef void (*caller_t)(PtrStepSz<T> src, PtrStepSz<T> srcWhole, int xoff, int yoff, PtrStepSzf xmap, PtrStepSzf ymap,
                PtrStepSz<T> dst, const float* borderValue, cudaStream_t stream, bool cc20);

            static const caller_t callers[3][5] =
            {
                {
                    RemapDispatcher<PointFilter, BrdConstant, T>::call,
                    RemapDispatcher<PointFilter, BrdReplicate, T>::call,
                    RemapDispatcher<PointFilter, BrdReflect, T>::call,
                    RemapDispatcher<PointFilter, BrdWrap, T>::call,
                    RemapDispatcher<PointFilter, BrdReflect101, T>::call
                },
                {
                    RemapDispatcher<LinearFilter, BrdConstant, T>::call,
                    RemapDispatcher<LinearFilter, BrdReplicate, T>::call,
                    RemapDispatcher<LinearFilter, BrdReflect, T>::call,
                    RemapDispatcher<LinearFilter, BrdWrap, T>::call,
                    RemapDispatcher<LinearFilter, BrdReflect101, T>::call
                },
                {
                    RemapDispatcher<CubicFilter, BrdConstant, T>::call,
                    RemapDispatcher<CubicFilter, BrdReplicate, T>::call,
                    RemapDispatcher<CubicFilter, BrdReflect, T>::call,
                    RemapDispatcher<CubicFilter, BrdWrap, T>::call,
                    RemapDispatcher<CubicFilter, BrdReflect101, T>::call
                }
            };

            callers[interpolation][borderMode](static_cast<PtrStepSz<T>>(src), static_cast<PtrStepSz<T>>(srcWhole), xoff, yoff, xmap, ymap,
                static_cast<PtrStepSz<T>>(dst), borderValue, stream, cc20);
        }

        template void remap_gpu<uchar >(PtrStepSzb src, PtrStepSzb srcWhole, int xoff, int yoff, PtrStepSzf xmap, PtrStepSzf ymap, PtrStepSzb dst, int interpolation, int borderMode, const float* borderValue, cudaStream_t stream, bool cc20);
        template void remap_gpu<uchar3>(PtrStepSzb src, PtrStepSzb srcWhole, int xoff, int yoff, PtrStepSzf xmap, PtrStepSzf ymap, PtrStepSzb dst, int interpolation, int borderMode, const float* borderValue, cudaStream_t stream, bool cc20);
        template void remap_gpu<uchar4>(PtrStepSzb src, PtrStepSzb srcWhole, int xoff, int yoff, PtrStepSzf xmap, PtrStepSzf ymap, PtrStepSzb dst, int interpolation, int borderMode, const float* borderValue, cudaStream_t stream, bool cc20);

        template void remap_gpu<ushort >(PtrStepSzb src, PtrStepSzb srcWhole, int xoff, int yoff, PtrStepSzf xmap, PtrStepSzf ymap, PtrStepSzb dst, int interpolation, int borderMode, const float* borderValue, cudaStream_t stream, bool cc20);
        template void remap_gpu<ushort3>(PtrStepSzb src, PtrStepSzb srcWhole, int xoff, int yoff, PtrStepSzf xmap, PtrStepSzf ymap, PtrStepSzb dst, int interpolation, int borderMode, const float* borderValue, cudaStream_t stream, bool cc20);
        template void remap_gpu<ushort4>(PtrStepSzb src, PtrStepSzb srcWhole, int xoff, int yoff, PtrStepSzf xmap, PtrStepSzf ymap, PtrStepSzb dst, int interpolation, int borderMode, const float* borderValue, cudaStream_t stream, bool cc20);

        template void remap_gpu<short >(PtrStepSzb src, PtrStepSzb srcWhole, int xoff, int yoff, PtrStepSzf xmap, PtrStepSzf ymap, PtrStepSzb dst, int interpolation, int borderMode, const float* borderValue, cudaStream_t stream, bool cc20);
        template void remap_gpu<short3>(PtrStepSzb src, PtrStepSzb srcWhole, int xoff, int yoff, PtrStepSzf xmap, PtrStepSzf ymap, PtrStepSzb dst, int interpolation, int borderMode, const float* borderValue, cudaStream_t stream, bool cc20);
        template void remap_gpu<short4>(PtrStepSzb src, PtrStepSzb srcWhole, int xoff, int yoff, PtrStepSzf xmap, PtrStepSzf ymap, PtrStepSzb dst, int interpolation, int borderMode, const float* borderValue, cudaStream_t stream, bool cc20);

        template void remap_gpu<float >(PtrStepSzb src, PtrStepSzb srcWhole, int xoff, int yoff, PtrStepSzf xmap, PtrStepSzf ymap, PtrStepSzb dst, int interpolation, int borderMode, const float* borderValue, cudaStream_t stream, bool cc20);
        template void remap_gpu<float3>(PtrStepSzb src, PtrStepSzb srcWhole, int xoff, int yoff, PtrStepSzf xmap, PtrStepSzf ymap, PtrStepSzb dst, int interpolation, int borderMode, const float* borderValue, cudaStream_t stream, bool cc20);
        template void remap_gpu<float4>(PtrStepSzb src, PtrStepSzb srcWhole, int xoff, int yoff, PtrStepSzf xmap, PtrStepSzf ymap, PtrStepSzb dst, int interpolation, int borderMode, const float* borderValue, cudaStream_t stream, bool cc20);
    } // namespace imgproc
}}} // namespace cv { namespace cuda { namespace cudev


#endif /* CUDA_DISABLER */
