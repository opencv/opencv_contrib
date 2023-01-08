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
        struct AffineTransform
        {
            static const int rows = 2;
            static __device__ __forceinline__ float2 calcCoord(const float warpMat[AffineTransform::rows * 3], int x, int y)
            {
                const float xcoo = warpMat[0] * x + warpMat[1] * y + warpMat[2];
                const float ycoo = warpMat[3] * x + warpMat[4] * y + warpMat[5];

                return make_float2(xcoo, ycoo);
            }

            struct Coefficients
            {
                Coefficients(const float* c_)
                {
                    for(int i = 0; i < AffineTransform::rows * 3; i++)
                        c[i] = c_[i];
                }
                float c[AffineTransform::rows * 3];
            };
        };

        struct PerspectiveTransform
        {
            static const int rows = 3;
            static __device__ __forceinline__ float2 calcCoord(const float warpMat[PerspectiveTransform::rows * 3], int x, int y)
            {
                const float coeff = 1.0f / (warpMat[6] * x + warpMat[7] * y + warpMat[8]);

                const float xcoo = coeff * (warpMat[0] * x + warpMat[1] * y + warpMat[2]);
                const float ycoo = coeff * (warpMat[3] * x + warpMat[4] * y + warpMat[5]);

                return make_float2(xcoo, ycoo);
            }
            struct Coefficients
            {
                Coefficients(const float* c_)
                {
                    for(int i = 0; i < PerspectiveTransform::rows * 3; i++)
                        c[i] = c_[i];
                }

                float c[PerspectiveTransform::rows * 3];
            };
        };

        ///////////////////////////////////////////////////////////////////
        // Build Maps

        template <class Transform> __global__ void buildWarpMaps(PtrStepSzf xmap, PtrStepf ymap, const typename Transform::Coefficients warpMat)
        {
            const int x = blockDim.x * blockIdx.x + threadIdx.x;
            const int y = blockDim.y * blockIdx.y + threadIdx.y;

            if (x < xmap.cols && y < xmap.rows)
            {
                const float2 coord = Transform::calcCoord(warpMat.c, x, y);

                xmap(y, x) = coord.x;
                ymap(y, x) = coord.y;
            }
        }

        template <class Transform> void buildWarpMaps_caller(PtrStepSzf xmap, PtrStepSzf ymap, const float warpMat[Transform::rows * 3], cudaStream_t stream)
        {
            dim3 block(32, 8);
            dim3 grid(divUp(xmap.cols, block.x), divUp(xmap.rows, block.y));

            buildWarpMaps<Transform><<<grid, block, 0, stream>>>(xmap, ymap, warpMat);
            cudaSafeCall( cudaGetLastError() );

            if (stream == 0)
                cudaSafeCall( cudaDeviceSynchronize() );
        }

        void buildWarpAffineMaps_gpu(float coeffs[2 * 3], PtrStepSzf xmap, PtrStepSzf ymap, cudaStream_t stream)
        {
            buildWarpMaps_caller<AffineTransform>(xmap, ymap, coeffs, stream);
        }

        void buildWarpPerspectiveMaps_gpu(float coeffs[3 * 3], PtrStepSzf xmap, PtrStepSzf ymap, cudaStream_t stream)
        {
            buildWarpMaps_caller<PerspectiveTransform>(xmap, ymap, coeffs, stream);
        }

        ///////////////////////////////////////////////////////////////////
        // Warp

        template <class Transform, class Ptr2D, typename T> __global__ void warp(const Ptr2D src, PtrStepSz<T> dst, const typename Transform::Coefficients warpMat)
        {
            const int x = blockDim.x * blockIdx.x + threadIdx.x;
            const int y = blockDim.y * blockIdx.y + threadIdx.y;

            if (x < dst.cols && y < dst.rows)
            {
                const float2 coord = Transform::calcCoord(warpMat.c, x, y);

                dst.ptr(y)[x] = saturate_cast<T>(src(coord.y, coord.x));
            }
        }

        template <class Transform, template <typename> class Filter, template <typename> class B, typename T> struct WarpDispatcherStream
        {
            static void call(PtrStepSz<T> src, PtrStepSz<T> dst, const float* borderValue, const float warpMat[Transform::rows*3], cudaStream_t stream, bool)
            {
                typedef typename TypeVec<float, VecTraits<T>::cn>::vec_type work_type;

                dim3 block(32, 8);
                dim3 grid(divUp(dst.cols, block.x), divUp(dst.rows, block.y));

                B<work_type> brd(src.rows, src.cols, VecTraits<work_type>::make(borderValue));
                BorderReader<PtrStep<T>, B<work_type>> brdSrc(src, brd);
                Filter<BorderReader<PtrStep<T>, B<work_type>>> filter_src(brdSrc);

                warp<Transform><<<grid, block, 0, stream>>>(filter_src, dst, warpMat);
                cudaSafeCall( cudaGetLastError() );
            }
        };

        template <class Transform, template <typename> class Filter, template <typename> class B, typename T> struct WarpDispatcherNonStream
        {
            static void call(PtrStepSz<T> src, PtrStepSz<T> srcWhole, int xoff, int yoff, PtrStepSz<T> dst, const float* borderValue, const float warpMat[Transform::rows*3], bool)
            {
                CV_UNUSED(xoff);
                CV_UNUSED(yoff);
                CV_UNUSED(srcWhole);

                typedef typename TypeVec<float, VecTraits<T>::cn>::vec_type work_type;

                dim3 block(32, 8);
                dim3 grid(divUp(dst.cols, block.x), divUp(dst.rows, block.y));

                B<work_type> brd(src.rows, src.cols, VecTraits<work_type>::make(borderValue));
                BorderReader<PtrStep<T>, B<work_type>> brdSrc(src, brd);
                Filter<BorderReader<PtrStep<T>, B<work_type>>> filter_src(brdSrc);

                warp<Transform><<<grid, block>>>(filter_src, dst, warpMat);
                cudaSafeCall( cudaGetLastError() );

                cudaSafeCall( cudaDeviceSynchronize() );
            }
        };

        template <class Transform, template <typename> class Filter, template <typename> class B, typename T> struct WarpDispatcherNonStreamTex
        {
            static void call(PtrStepSz<T> src, PtrStepSz<T> srcWhole, int xoff, int yoff, PtrStepSz<T> dst, const float* borderValue, const float warpMat[Transform::rows*3], bool cc20)
            {
                typedef typename TypeVec<float, VecTraits<T>::cn>::vec_type work_type;
                dim3 block(32, cc20 ? 8 : 4);
                dim3 grid(divUp(dst.cols, block.x), divUp(dst.rows, block.y));
                if (xoff || yoff) {
                    cudev::TextureOff<T> texSrcWhole(srcWhole, yoff, xoff);
                    B<work_type> brd(src.rows, src.cols, VecTraits<work_type>::make(borderValue));
                    BorderReader<cudev::TextureOffPtr<T>, B<work_type>> brdSrc(texSrcWhole, brd);
                    Filter<BorderReader<cudev::TextureOffPtr<T>, B<work_type>>> filter_src(brdSrc);
                    warp<Transform><<<grid, block>>> (filter_src, dst, warpMat);
                }
                else {
                    cudev::Texture<T> texSrcWhole(srcWhole);
                    B<work_type> brd(src.rows, src.cols, VecTraits<work_type>::make(borderValue));
                    BorderReader<cudev::TexturePtr<T>, B<work_type>>brdSrc(texSrcWhole, brd);
                    Filter< BorderReader<cudev::TexturePtr<T>, B<work_type>>> filter_src(brdSrc);
                    warp<Transform><<<grid, block>>> (filter_src, dst, warpMat);
                }
                cudaSafeCall( cudaGetLastError() );
                cudaSafeCall( cudaDeviceSynchronize() );
            }
        };

        template <class Transform, template <typename> class Filter, template <typename> class B> struct WarpDispatcherNonStream<Transform, Filter, B, uchar> :
            WarpDispatcherNonStreamTex<Transform, Filter, B, uchar> {};
        template <class Transform, template <typename> class Filter, template <typename> class B> struct WarpDispatcherNonStream<Transform, Filter, B, uchar4> :
            WarpDispatcherNonStreamTex<Transform, Filter, B, uchar4> {};
        template <class Transform, template <typename> class Filter, template <typename> class B> struct WarpDispatcherNonStream<Transform, Filter, B, ushort> :
            WarpDispatcherNonStreamTex<Transform, Filter, B, ushort> {};
        template <class Transform, template <typename> class Filter, template <typename> class B> struct WarpDispatcherNonStream<Transform, Filter, B, ushort4> :
            WarpDispatcherNonStreamTex<Transform, Filter, B, ushort4> {};
        template <class Transform, template <typename> class Filter, template <typename> class B> struct WarpDispatcherNonStream<Transform, Filter, B, short> :
            WarpDispatcherNonStreamTex<Transform, Filter, B, short> {};
        template <class Transform, template <typename> class Filter, template <typename> class B> struct WarpDispatcherNonStream<Transform, Filter, B, short4> :
            WarpDispatcherNonStreamTex<Transform, Filter, B, short4> {};
        template <class Transform, template <typename> class Filter, template <typename> class B> struct WarpDispatcherNonStream<Transform, Filter, B, float> :
            WarpDispatcherNonStreamTex<Transform, Filter, B, float> {};
        template <class Transform, template <typename> class Filter, template <typename> class B> struct WarpDispatcherNonStream<Transform, Filter, B, float4> :
            WarpDispatcherNonStreamTex<Transform, Filter, B, float4> {};

        template <class Transform, template <typename> class Filter, template <typename> class B, typename T> struct WarpDispatcher
        {
            static void call(PtrStepSz<T> src, PtrStepSz<T> srcWhole, int xoff, int yoff, PtrStepSz<T> dst, const float* borderValue, const float warpMat[Transform::rows*3], cudaStream_t stream, bool cc20)
            {
                if (stream == 0)
                    WarpDispatcherNonStream<Transform, Filter, B, T>::call(src, srcWhole, xoff, yoff, dst, borderValue, warpMat, cc20);
                else
                    WarpDispatcherStream<Transform, Filter, B, T>::call(src, dst, borderValue, warpMat, stream, cc20);
            }
        };

        template <class Transform, typename T>
        void warp_caller(PtrStepSzb src, PtrStepSzb srcWhole, int xoff, int yoff, PtrStepSzb dst, int interpolation,
                         int borderMode, const float* borderValue, const float warpMat[Transform::rows*3], cudaStream_t stream, bool cc20)
        {
            typedef void (*func_t)(PtrStepSz<T> src, PtrStepSz<T> srcWhole, int xoff, int yoff, PtrStepSz<T> dst, const float* borderValue, const float warpMat[Transform::rows*3], cudaStream_t stream, bool cc20);

            static const func_t funcs[3][5] =
            {
                {
                    WarpDispatcher<Transform, PointFilter, BrdConstant, T>::call,
                    WarpDispatcher<Transform, PointFilter, BrdReplicate, T>::call,
                    WarpDispatcher<Transform, PointFilter, BrdReflect, T>::call,
                    WarpDispatcher<Transform, PointFilter, BrdWrap, T>::call,
                    WarpDispatcher<Transform, PointFilter, BrdReflect101, T>::call
                },
                {
                    WarpDispatcher<Transform, LinearFilter, BrdConstant, T>::call,
                    WarpDispatcher<Transform, LinearFilter, BrdReplicate, T>::call,
                    WarpDispatcher<Transform, LinearFilter, BrdReflect, T>::call,
                    WarpDispatcher<Transform, LinearFilter, BrdWrap, T>::call,
                    WarpDispatcher<Transform, LinearFilter, BrdReflect101, T>::call
                },
                {
                    WarpDispatcher<Transform, CubicFilter, BrdConstant, T>::call,
                    WarpDispatcher<Transform, CubicFilter, BrdReplicate, T>::call,
                    WarpDispatcher<Transform, CubicFilter, BrdReflect, T>::call,
                    WarpDispatcher<Transform, CubicFilter, BrdWrap, T>::call,
                    WarpDispatcher<Transform, CubicFilter, BrdReflect101, T>::call
                }
            };

            funcs[interpolation][borderMode](static_cast<PtrStepSz<T>>(src), static_cast<PtrStepSz<T>>(srcWhole), xoff, yoff,
                                             static_cast<PtrStepSz<T>>(dst), borderValue, warpMat, stream, cc20);
        }

        template <typename T> void warpAffine_gpu(PtrStepSzb src, PtrStepSzb srcWhole, int xoff, int yoff, float coeffs[2 * 3], PtrStepSzb dst, int interpolation,
                                                  int borderMode, const float* borderValue, cudaStream_t stream, bool cc20)
        {
            warp_caller<AffineTransform, T>(src, srcWhole, xoff, yoff, dst, interpolation, borderMode, borderValue, coeffs, stream, cc20);
        }

        template void warpAffine_gpu<uchar >(PtrStepSzb src, PtrStepSzb srcWhole, int xoff, int yoff, float coeffs[2 * 3], PtrStepSzb dst, int interpolation, int borderMode, const float* borderValue, cudaStream_t stream, bool cc20);
        template void warpAffine_gpu<uchar3>(PtrStepSzb src, PtrStepSzb srcWhole, int xoff, int yoff, float coeffs[2 * 3], PtrStepSzb dst, int interpolation, int borderMode, const float* borderValue, cudaStream_t stream, bool cc20);
        template void warpAffine_gpu<uchar4>(PtrStepSzb src, PtrStepSzb srcWhole, int xoff, int yoff, float coeffs[2 * 3], PtrStepSzb dst, int interpolation, int borderMode, const float* borderValue, cudaStream_t stream, bool cc20);

        template void warpAffine_gpu<ushort >(PtrStepSzb src, PtrStepSzb srcWhole, int xoff, int yoff, float coeffs[2 * 3], PtrStepSzb dst, int interpolation, int borderMode, const float* borderValue, cudaStream_t stream, bool cc20);
        template void warpAffine_gpu<ushort3>(PtrStepSzb src, PtrStepSzb srcWhole, int xoff, int yoff, float coeffs[2 * 3], PtrStepSzb dst, int interpolation, int borderMode, const float* borderValue, cudaStream_t stream, bool cc20);
        template void warpAffine_gpu<ushort4>(PtrStepSzb src, PtrStepSzb srcWhole, int xoff, int yoff, float coeffs[2 * 3], PtrStepSzb dst, int interpolation, int borderMode, const float* borderValue, cudaStream_t stream, bool cc20);

        template void warpAffine_gpu<short >(PtrStepSzb src, PtrStepSzb srcWhole, int xoff, int yoff, float coeffs[2 * 3], PtrStepSzb dst, int interpolation, int borderMode, const float* borderValue, cudaStream_t stream, bool cc20);
        template void warpAffine_gpu<short3>(PtrStepSzb src, PtrStepSzb srcWhole, int xoff, int yoff, float coeffs[2 * 3], PtrStepSzb dst, int interpolation, int borderMode, const float* borderValue, cudaStream_t stream, bool cc20);
        template void warpAffine_gpu<short4>(PtrStepSzb src, PtrStepSzb srcWhole, int xoff, int yoff, float coeffs[2 * 3], PtrStepSzb dst, int interpolation, int borderMode, const float* borderValue, cudaStream_t stream, bool cc20);

        template void warpAffine_gpu<float >(PtrStepSzb src, PtrStepSzb srcWhole, int xoff, int yoff, float coeffs[2 * 3], PtrStepSzb dst, int interpolation, int borderMode, const float* borderValue, cudaStream_t stream, bool cc20);
        template void warpAffine_gpu<float3>(PtrStepSzb src, PtrStepSzb srcWhole, int xoff, int yoff, float coeffs[2 * 3], PtrStepSzb dst, int interpolation, int borderMode, const float* borderValue, cudaStream_t stream, bool cc20);
        template void warpAffine_gpu<float4>(PtrStepSzb src, PtrStepSzb srcWhole, int xoff, int yoff, float coeffs[2 * 3], PtrStepSzb dst, int interpolation, int borderMode, const float* borderValue, cudaStream_t stream, bool cc20);

        template <typename T> void warpPerspective_gpu(PtrStepSzb src, PtrStepSzb srcWhole, int xoff, int yoff, float coeffs[3 * 3], PtrStepSzb dst, int interpolation,
                                                  int borderMode, const float* borderValue, cudaStream_t stream, bool cc20)
        {
            warp_caller<PerspectiveTransform, T>(src, srcWhole, xoff, yoff, dst, interpolation, borderMode, borderValue, coeffs, stream, cc20);
        }

        template void warpPerspective_gpu<uchar >(PtrStepSzb src, PtrStepSzb srcWhole, int xoff, int yoff, float coeffs[3 * 3], PtrStepSzb dst, int interpolation, int borderMode, const float* borderValue, cudaStream_t stream, bool cc20);
        template void warpPerspective_gpu<uchar3>(PtrStepSzb src, PtrStepSzb srcWhole, int xoff, int yoff, float coeffs[3 * 3], PtrStepSzb dst, int interpolation, int borderMode, const float* borderValue, cudaStream_t stream, bool cc20);
        template void warpPerspective_gpu<uchar4>(PtrStepSzb src, PtrStepSzb srcWhole, int xoff, int yoff, float coeffs[3 * 3], PtrStepSzb dst, int interpolation, int borderMode, const float* borderValue, cudaStream_t stream, bool cc20);

        template void warpPerspective_gpu<ushort >(PtrStepSzb src, PtrStepSzb srcWhole, int xoff, int yoff, float coeffs[3 * 3], PtrStepSzb dst, int interpolation, int borderMode, const float* borderValue, cudaStream_t stream, bool cc20);
        template void warpPerspective_gpu<ushort3>(PtrStepSzb src, PtrStepSzb srcWhole, int xoff, int yoff, float coeffs[3 * 3], PtrStepSzb dst, int interpolation, int borderMode, const float* borderValue, cudaStream_t stream, bool cc20);
        template void warpPerspective_gpu<ushort4>(PtrStepSzb src, PtrStepSzb srcWhole, int xoff, int yoff, float coeffs[3 * 3], PtrStepSzb dst, int interpolation, int borderMode, const float* borderValue, cudaStream_t stream, bool cc20);

        template void warpPerspective_gpu<short >(PtrStepSzb src, PtrStepSzb srcWhole, int xoff, int yoff, float coeffs[3 * 3], PtrStepSzb dst, int interpolation, int borderMode, const float* borderValue, cudaStream_t stream, bool cc20);
        template void warpPerspective_gpu<short3>(PtrStepSzb src, PtrStepSzb srcWhole, int xoff, int yoff, float coeffs[3 * 3], PtrStepSzb dst, int interpolation, int borderMode, const float* borderValue, cudaStream_t stream, bool cc20);
        template void warpPerspective_gpu<short4>(PtrStepSzb src, PtrStepSzb srcWhole, int xoff, int yoff, float coeffs[3 * 3], PtrStepSzb dst, int interpolation, int borderMode, const float* borderValue, cudaStream_t stream, bool cc20);

        template void warpPerspective_gpu<float >(PtrStepSzb src, PtrStepSzb srcWhole, int xoff, int yoff, float coeffs[3 * 3], PtrStepSzb dst, int interpolation, int borderMode, const float* borderValue, cudaStream_t stream, bool cc20);
        template void warpPerspective_gpu<float3>(PtrStepSzb src, PtrStepSzb srcWhole, int xoff, int yoff, float coeffs[3 * 3], PtrStepSzb dst, int interpolation, int borderMode, const float* borderValue, cudaStream_t stream, bool cc20);
        template void warpPerspective_gpu<float4>(PtrStepSzb src, PtrStepSzb srcWhole, int xoff, int yoff, float coeffs[3 * 3], PtrStepSzb dst, int interpolation, int borderMode, const float* borderValue, cudaStream_t stream, bool cc20);
    } // namespace imgproc
}}} // namespace cv { namespace cuda { namespace cudev


#endif /* CUDA_DISABLER */
