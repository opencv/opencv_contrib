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

#include "opencv2/opencv_modules.hpp"

#ifndef HAVE_OPENCV_CUDEV

#error "opencv_cudev is required"

#else

#include <algorithm>
#include "opencv2/cudaarithm.hpp"
#include "opencv2/cudev.hpp"
#include "opencv2/core/private.cuda.hpp"

using namespace cv;
using namespace cv::cuda;
using namespace cv::cudev;

////////////////////////////////////////////////////////////////////////
/// merge

namespace
{
    template <int cn, typename T> struct MergeFunc;

    template <typename T> struct MergeFunc<2, T>
    {
        static void call(const GpuMat* src, GpuMat& dst, Stream& stream)
        {
            const std::array<GlobPtrSz<T>, 2> d_src = {globPtr<T>(src[0]), globPtr<T>(src[1])};
            gridMerge(d_src,
                    globPtr<typename MakeVec<T, 2>::type>(dst),
                    stream);
        }
    };

    template <typename T> struct MergeFunc<3, T>
    {
        static void call(const GpuMat* src, GpuMat& dst, Stream& stream)
        {
            const std::array<GlobPtrSz<T>, 3> d_src = {globPtr<T>(src[0]), globPtr<T>(src[1]), globPtr<T>(src[2])};
            gridMerge(d_src,
                    globPtr<typename MakeVec<T, 3>::type>(dst),
                    stream);
        }
    };

    template <typename T> struct MergeFunc<4, T>
    {
        static void call(const GpuMat* src, GpuMat& dst, Stream& stream)
        {
            const std::array<GlobPtrSz<T>, 4 > d_src = {globPtr<T>(src[0]), globPtr<T>(src[1]), globPtr<T>(src[2]), globPtr<T>(src[3])};
            gridMerge(d_src,
                    globPtr<typename MakeVec<T, 4>::type>(dst),
                    stream);
        }
    };

    enum { kMaxMergeSplitBatch = 64 };

    template <typename T>
    struct MergeBatch
    {
        GlobPtr<T> src[kMaxMergeSplitBatch];
        ushort src_cn[kMaxMergeSplitBatch];
        ushort dst_ofs[kMaxMergeSplitBatch];
        int count;
    };

    template <typename T>
    __global__ void mergeChannelsKernel(const MergeBatch<T> batch, GlobPtr<T> dst, const int rows, const int cols, const int dst_cn)
    {
        const int x = blockIdx.x * blockDim.x + threadIdx.x;
        if (x >= cols)
            return;

        const uint stride_y = blockDim.y * gridDim.y;
        for (uint uy = blockIdx.y * blockDim.y + threadIdx.y; uy < static_cast<uint>(rows); uy += stride_y)
        {
            const int y = static_cast<int>(uy);
            T* dst_px = dst.row(y) + static_cast<size_t>(x) * dst_cn;
            for (int b = 0; b < batch.count; ++b)
            {
                const int scn = batch.src_cn[b];
                const T* src_px = batch.src[b].row(y) + static_cast<size_t>(x) * scn;
                T* out = dst_px + batch.dst_ofs[b];
                for (int c = 0; c < scn; ++c)
                    out[c] = src_px[c];
            }
        }
    }

    template <typename T> struct MergeChannelsFunc
    {
        static void call(const GpuMat* src, size_t n, GpuMat& dst, Stream& stream)
        {
            const int rows = dst.rows;
            const int cols = dst.cols;
            if (rows <= 0 || cols <= 0)
                return;

            const int dst_cn = dst.channels();
            const dim3 block(DefaultSplitMergePolicy::block_size_x, DefaultSplitMergePolicy::block_size_y);
            const dim3 grid(divUp(cols, block.x), std::min(divUp(rows, block.y), 65535));
            cudaStream_t cuStream = StreamAccessor::getStream(stream);

            int ch_offset = 0;
            for (size_t base = 0; base < n; base += kMaxMergeSplitBatch)
            {
                MergeBatch<T> batch;
                const int count = static_cast<int>(std::min<size_t>(n - base, kMaxMergeSplitBatch));
                batch.count = count;
                for (int b = 0; b < count; ++b)
                {
                    const GpuMat& s = src[base + b];
                    const int scn = s.channels();
                    batch.src[b] = shrinkPtr(globPtr<T>(s));
                    batch.src_cn[b] = static_cast<ushort>(scn);
                    batch.dst_ofs[b] = static_cast<ushort>(ch_offset);
                    ch_offset += scn;
                }

                mergeChannelsKernel<T><<<grid, block, 0, cuStream>>>(
                    batch, shrinkPtr(globPtr<T>(dst)), rows, cols, dst_cn);
                CV_CUDEV_SAFE_CALL( cudaGetLastError() );
            }

            if (cuStream == 0)
                CV_CUDEV_SAFE_CALL( cudaDeviceSynchronize() );
        }
    };

    void mergeImpl(const GpuMat* src, size_t n, cv::OutputArray _dst, Stream& stream)
    {
        CV_Assert( src != 0 );
        CV_Assert( n > 0 && n <= static_cast<size_t>(CV_CN_MAX) );
        CV_Assert( !src[0].empty() );

        const int depth = src[0].depth();
        const cv::Size size = src[0].size();
        const size_t elem_size = CV_ELEM_SIZE1(depth);
        CV_Assert( elem_size <= 8 );

        int total_channels = 0;
        bool all_single_channel = true;
        for (size_t i = 0; i < n; ++i)
        {
            CV_Assert( src[i].size() == size );
            CV_Assert( src[i].depth() == depth );
            const int cn = src[i].channels();
            CV_Assert( cn > 0 );
            if (cn != 1)
                all_single_channel = false;
            total_channels += cn;
        }

        CV_Assert( total_channels > 0 && total_channels <= CV_CN_MAX );

        if (n == 1)
        {
            GpuMat src0 = src[0];
            GpuMat dst = getOutputMat(_dst, size, src0.type(), stream);
            if (src0.data != dst.data || src0.step != dst.step)
            {
                if (stream)
                    src0.copyTo(dst, stream);
                else
                    src0.copyTo(dst);
            }
            syncOutput(dst, _dst, stream);
            return;
        }

        std::vector<GpuMat> src_copy;
        if (_dst.kind() == _InputArray::CUDA_GPU_MAT)
        {
            const GpuMat* dst_hdr = &_dst.getGpuMatRef();
            bool aliased = (dst_hdr >= src && dst_hdr < src + n);
            if (!aliased && dst_hdr->datastart != 0)
            {
                for (size_t i = 0; i < n; ++i)
                {
                    if (src[i].datastart == dst_hdr->datastart)
                    {
                        aliased = true;
                        break;
                    }
                }
            }
            if (aliased)
            {
                src_copy.assign(src, src + n);
                for (size_t i = 0; i < n; ++i)
                {
                    if (dst_hdr->datastart != 0 && src_copy[i].datastart == dst_hdr->datastart)
                    {
                        GpuMat tmp;
                        if (stream)
                            src_copy[i].copyTo(tmp, stream);
                        else
                            src_copy[i].copyTo(tmp);
                        src_copy[i] = tmp;
                    }
                }
                src = src_copy.data();
            }
        }

        if (all_single_channel && total_channels <= 4)
        {
            typedef void (*func_t)(const GpuMat* src, GpuMat& dst, Stream& stream);
            static const func_t funcs[3][5] =
            {
                {MergeFunc<2, uchar>::call, MergeFunc<2, ushort>::call, MergeFunc<2, int>::call, 0, MergeFunc<2, double>::call},
                {MergeFunc<3, uchar>::call, MergeFunc<3, ushort>::call, MergeFunc<3, int>::call, 0, MergeFunc<3, double>::call},
                {MergeFunc<4, uchar>::call, MergeFunc<4, ushort>::call, MergeFunc<4, int>::call, 0, MergeFunc<4, double>::call}
            };

            const func_t func = funcs[total_channels - 2][elem_size / 2];

            if (func == 0)
                CV_Error(cv::Error::StsUnsupportedFormat, "Unsupported channel count or data type");

            GpuMat dst = getOutputMat(_dst, size, CV_MAKE_TYPE(depth, total_channels), stream);

            func(src, dst, stream);

            syncOutput(dst, _dst, stream);
        }
        else
        {
            typedef void (*func_t)(const GpuMat* src, size_t n, GpuMat& dst, Stream& stream);
            static const func_t funcs[5] =
            {
                MergeChannelsFunc<uchar>::call,
                MergeChannelsFunc<ushort>::call,
                MergeChannelsFunc<int>::call,
                0,
                MergeChannelsFunc<double>::call
            };

            const func_t func = funcs[elem_size / 2];

            if (func == 0)
                CV_Error(cv::Error::StsUnsupportedFormat, "Unsupported channel count or data type");

            GpuMat dst = getOutputMat(_dst, size, CV_MAKE_TYPE(depth, total_channels), stream);

            func(src, n, dst, stream);

            syncOutput(dst, _dst, stream);
        }
    }
}

void cv::cuda::merge(const GpuMat* src, size_t n, OutputArray dst, Stream& stream)
{
    mergeImpl(src, n, dst, stream);
}


void cv::cuda::merge(const std::vector<GpuMat>& src, OutputArray dst, Stream& stream)
{
    mergeImpl(src.empty() ? 0 : &src[0], src.size(), dst, stream);
}

////////////////////////////////////////////////////////////////////////
/// split

namespace
{
    template <int cn, typename T> struct SplitFunc;

    template <typename T> struct SplitFunc<2, T>
    {
        static void call(const GpuMat& src, GpuMat* dst, Stream& stream)
        {
            GlobPtrSz<T> dstarr[2] =
            {
                globPtr<T>(dst[0]), globPtr<T>(dst[1])
            };

            gridSplit(globPtr<typename MakeVec<T, 2>::type>(src), dstarr, stream);
        }
    };

    template <typename T> struct SplitFunc<3, T>
    {
        static void call(const GpuMat& src, GpuMat* dst, Stream& stream)
        {
            GlobPtrSz<T> dstarr[3] =
            {
                globPtr<T>(dst[0]), globPtr<T>(dst[1]), globPtr<T>(dst[2])
            };

            gridSplit(globPtr<typename MakeVec<T, 3>::type>(src), dstarr, stream);
        }
    };

    template <typename T> struct SplitFunc<4, T>
    {
        static void call(const GpuMat& src, GpuMat* dst, Stream& stream)
        {
            GlobPtrSz<T> dstarr[4] =
            {
                globPtr<T>(dst[0]), globPtr<T>(dst[1]), globPtr<T>(dst[2]), globPtr<T>(dst[3])
            };

            gridSplit(globPtr<typename MakeVec<T, 4>::type>(src), dstarr, stream);
        }
    };

    template <typename T>
    struct SplitBatch
    {
        GlobPtr<T> dst[kMaxMergeSplitBatch];
        int ch_offset;
        int count;
    };

    template <typename T>
    __global__ void splitChannelsKernel(const GlobPtr<T> src, SplitBatch<T> batch, const int rows, const int cols, const int src_cn)
    {
        const int x = blockIdx.x * blockDim.x + threadIdx.x;
        if (x >= cols)
            return;

        const uint stride_y = blockDim.y * gridDim.y;
        for (uint uy = blockIdx.y * blockDim.y + threadIdx.y; uy < static_cast<uint>(rows); uy += stride_y)
        {
            const int y = static_cast<int>(uy);
            const T* src_px = src.row(y) + static_cast<size_t>(x) * src_cn + batch.ch_offset;
            for (int b = 0; b < batch.count; ++b)
                batch.dst[b].row(y)[x] = src_px[b];
        }
    }

    template <typename T> struct SplitChannelsFunc
    {
        static void call(const GpuMat& src, GpuMat* dst, Stream& stream)
        {
            const int rows = src.rows;
            const int cols = src.cols;
            if (rows <= 0 || cols <= 0)
                return;

            const int src_cn = src.channels();
            const dim3 block(DefaultSplitMergePolicy::block_size_x, DefaultSplitMergePolicy::block_size_y);
            const dim3 grid(divUp(cols, block.x), std::min(divUp(rows, block.y), 65535));
            cudaStream_t cuStream = StreamAccessor::getStream(stream);

            for (int ch_offset = 0; ch_offset < src_cn; ch_offset += kMaxMergeSplitBatch)
            {
                SplitBatch<T> batch;
                const int count = std::min<int>(src_cn - ch_offset, kMaxMergeSplitBatch);
                batch.ch_offset = ch_offset;
                batch.count = count;
                for (int b = 0; b < count; ++b)
                    batch.dst[b] = shrinkPtr(globPtr<T>(dst[ch_offset + b]));

                splitChannelsKernel<T><<<grid, block, 0, cuStream>>>(
                    shrinkPtr(globPtr<T>(src)), batch, rows, cols, src_cn);
                CV_CUDEV_SAFE_CALL( cudaGetLastError() );
            }

            if (cuStream == 0)
                CV_CUDEV_SAFE_CALL( cudaDeviceSynchronize() );
        }
    };

    void splitImpl(const GpuMat& src, GpuMat* dst, Stream& stream)
    {
        typedef void (*func_t)(const GpuMat& src, GpuMat* dst, Stream& stream);
        static const func_t funcs[3][5] =
        {
            {SplitFunc<2, uchar>::call, SplitFunc<2, ushort>::call, SplitFunc<2, int>::call, 0, SplitFunc<2, double>::call},
            {SplitFunc<3, uchar>::call, SplitFunc<3, ushort>::call, SplitFunc<3, int>::call, 0, SplitFunc<3, double>::call},
            {SplitFunc<4, uchar>::call, SplitFunc<4, ushort>::call, SplitFunc<4, int>::call, 0, SplitFunc<4, double>::call}
        };

        CV_Assert( dst != 0 );

        const int depth = src.depth();
        const int channels = src.channels();
        const size_t elem_size = CV_ELEM_SIZE1(depth);

        CV_Assert( channels > 0 && channels <= CV_CN_MAX );
        CV_Assert( elem_size <= 8 );

        if (src.empty())
        {
            for (int i = 0; i < channels; ++i)
                dst[i].release();
            return;
        }

        if (channels == 1)
        {
            dst[0].create(src.size(), depth);
            if (src.data != dst[0].data || src.step != dst[0].step)
            {
                if (stream)
                    src.copyTo(dst[0], stream);
                else
                    src.copyTo(dst[0]);
            }
            return;
        }

        GpuMat src_copy;
        const GpuMat* src_ptr = &src;
        for (int i = 0; i < channels; ++i)
        {
            if (dst[i].datastart != 0 && dst[i].datastart == src.datastart)
            {
                if (stream)
                    src.copyTo(src_copy, stream);
                else
                    src.copyTo(src_copy);
                src_ptr = &src_copy;
                break;
            }
        }

        if (channels <= 4)
        {
            const func_t func = funcs[channels - 2][elem_size / 2];

            if (func == 0)
                CV_Error(cv::Error::StsUnsupportedFormat, "Unsupported channel count or data type");

            for (int i = 0; i < channels; ++i)
                dst[i].create(src.size(), depth);

            func(*src_ptr, dst, stream);
        }
        else
        {
            typedef void (*multi_func_t)(const GpuMat& src, GpuMat* dst, Stream& stream);
            static const multi_func_t multi_funcs[5] =
            {
                SplitChannelsFunc<uchar>::call,
                SplitChannelsFunc<ushort>::call,
                SplitChannelsFunc<int>::call,
                0,
                SplitChannelsFunc<double>::call
            };

            const multi_func_t func = multi_funcs[elem_size / 2];

            if (func == 0)
                CV_Error(cv::Error::StsUnsupportedFormat, "Unsupported channel count or data type");

            for (int i = 0; i < channels; ++i)
                dst[i].create(src.size(), depth);

            func(*src_ptr, dst, stream);
        }
    }
}

void cv::cuda::split(InputArray _src, GpuMat* dst, Stream& stream)
{
    GpuMat src = getInputMat(_src, stream);
    splitImpl(src, dst, stream);
}

void cv::cuda::split(InputArray _src, std::vector<GpuMat>& dst, Stream& stream)
{
    GpuMat src = getInputMat(_src, stream);
    if (src.empty())
    {
        dst.clear();
        return;
    }
    dst.resize(src.channels());
    splitImpl(src, &dst[0], stream);
}

#endif
