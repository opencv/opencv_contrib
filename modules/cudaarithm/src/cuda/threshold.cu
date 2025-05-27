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

#include "opencv2/cudaarithm.hpp"
#include "opencv2/cudev.hpp"
#include "opencv2/core/private.cuda.hpp"

using namespace cv;
using namespace cv::cuda;
using namespace cv::cudev;

namespace
{
    template <typename ScalarDepth> struct TransformPolicy : DefaultTransformPolicy
    {
    };
    template <> struct TransformPolicy<double> : DefaultTransformPolicy
    {
        enum {
            shift = 1
        };
    };

    template <typename T>
    void thresholdImpl(const GpuMat& src, GpuMat& dst, double thresh, double maxVal, int type, Stream& stream)
    {
        const T thresh_ = static_cast<T>(thresh);
        const T maxVal_ = static_cast<T>(maxVal);

        switch (type)
        {
        case 0:
            gridTransformUnary_< TransformPolicy<T> >(globPtr<T>(src), globPtr<T>(dst), thresh_binary_func(thresh_, maxVal_), stream);
            break;
        case 1:
            gridTransformUnary_< TransformPolicy<T> >(globPtr<T>(src), globPtr<T>(dst), thresh_binary_inv_func(thresh_, maxVal_), stream);
            break;
        case 2:
            gridTransformUnary_< TransformPolicy<T> >(globPtr<T>(src), globPtr<T>(dst), thresh_trunc_func(thresh_), stream);
            break;
        case 3:
            gridTransformUnary_< TransformPolicy<T> >(globPtr<T>(src), globPtr<T>(dst), thresh_to_zero_func(thresh_), stream);
            break;
        case 4:
            gridTransformUnary_< TransformPolicy<T> >(globPtr<T>(src), globPtr<T>(dst), thresh_to_zero_inv_func(thresh_), stream);
            break;
        };
    }
}

template <uint n_bins>
__global__ void otsu_mean(uint *histogram, uint *threshold_sums, unsigned long long *sums)
{
    extern __shared__ unsigned long long shared_memory_u64[];

    uint bin_idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint threshold = blockIdx.y * blockDim.y + threadIdx.y;

    uint threshold_sum_above = 0;
    unsigned long long sum_above = 0;

    if (bin_idx >= threshold)
    {
        uint value = histogram[bin_idx];
        threshold_sum_above = value;
        sum_above = value * bin_idx;
    }

    blockReduce<n_bins>((uint *)shared_memory_u64, threshold_sum_above, bin_idx, plus<uint>());
    __syncthreads();
    blockReduce<n_bins>((unsigned long long *)shared_memory_u64, sum_above, bin_idx, plus<unsigned long long>());

    if (bin_idx == 0)
    {
        threshold_sums[threshold] = threshold_sum_above;
        sums[threshold] = sum_above;
    }
}

template <uint n_bins>
__global__ void
otsu_variance(longlong2 *variance, uint *histogram, uint *threshold_sums, unsigned long long *sums)
{
    extern __shared__ signed long long shared_memory_i64[];

    uint bin_idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint threshold = blockIdx.y * blockDim.y + threadIdx.y;

    uint n_samples = threshold_sums[0];
    uint n_samples_above = threshold_sums[threshold];
    uint n_samples_below = n_samples - n_samples_above;

    unsigned long long total_sum = sums[0];
    unsigned long long sum_above = sums[threshold];
    unsigned long long sum_below = total_sum - sum_above;

    float threshold_variance_above_f32 = 0;
    float threshold_variance_below_f32 = 0;
    if (bin_idx >= threshold)
    {
        float mean = (float) sum_above / n_samples_above;
        float sigma = bin_idx - mean;
        threshold_variance_above_f32 = sigma * sigma;
    }
    else
    {
        float mean = (float) sum_below / n_samples_below;
        float sigma = bin_idx - mean;
        threshold_variance_below_f32 = sigma * sigma;
    }

    uint bin_count = histogram[bin_idx];
    signed long long threshold_variance_above_i64 = (signed long long)(threshold_variance_above_f32 * bin_count);
    signed long long threshold_variance_below_i64 = (signed long long)(threshold_variance_below_f32 * bin_count);
    blockReduce<n_bins>((signed long long *)shared_memory_i64, threshold_variance_above_i64, bin_idx, plus<signed long long>());
    __syncthreads();
    blockReduce<n_bins>((signed long long *)shared_memory_i64, threshold_variance_below_i64, bin_idx, plus<signed long long>());

    if (bin_idx == 0)
    {
        variance[threshold] = make_longlong2(threshold_variance_above_i64, threshold_variance_below_i64);
    }
}

template <uint n_thresholds>
__global__ void
otsu_score(uint *otsu_threshold, uint *threshold_sums, longlong2 *variance)
{
    extern __shared__ float shared_memory_f32[];

    uint threshold = blockIdx.x * blockDim.x + threadIdx.x;

    uint n_samples = threshold_sums[0];
    uint n_samples_above = threshold_sums[threshold];
    uint n_samples_below = n_samples - n_samples_above;

    float threshold_mean_above = (float)n_samples_above / n_samples;
    float threshold_mean_below = (float)n_samples_below / n_samples;

    longlong2 variances = variance[threshold];
    float variance_above = (float)variances.x / n_samples_above;
    float variance_below = (float)variances.y / n_samples_below;

    float above = threshold_mean_above * variance_above;
    float below = threshold_mean_below * variance_below;
    float score = above + below;

    float original_score = score;

    blockReduce<n_thresholds>((float *)shared_memory_f32, score, threshold, minimum<float>());

    if (threshold == 0)
    {
        shared_memory_f32[0] = score;
    }
    __syncthreads();

    score = shared_memory_f32[0];

    // We found the minimum score, but we need to find the threshold. If we find the thread with the minimum score, we
    // know which threshold it is
    if (original_score == score)
    {
        *otsu_threshold = threshold - 1;
    }
}

template <uint n_bins, uint n_thresholds>
void compute_otsu_async(uint *histogram, uint *otsu_threshold, Stream &stream)
{
    CV_Assert(n_bins == 256);
    CV_Assert(n_thresholds == 256);

    cudaStream_t cuda_stream = StreamAccessor::getStream(stream);

    dim3 block_all = dim3(n_bins, 1, 1);
    dim3 grid_all = dim3(divUp((int)n_bins, block_all.x), divUp((int)n_thresholds, block_all.y), 1);
    dim3 block_score = dim3(n_thresholds, 1, 1);
    dim3 grid_score = dim3(divUp((int)n_thresholds, block_score.x), 1, 1);

    GpuMat gpu_threshold_sums = GpuMat(1, n_bins, CV_32SC1);
    GpuMat gpu_sums = GpuMat(1, n_bins, CV_64FC1);
    GpuMat gpu_variances = GpuMat(1, n_bins, CV_32SC4);

    uint shared_memory;
    shared_memory = n_bins * sizeof(unsigned long long);
    otsu_mean<256><<<grid_all, block_all, shared_memory, cuda_stream>>>(histogram, gpu_threshold_sums.ptr<uint>(), gpu_sums.ptr<unsigned long long>());
    otsu_variance<256><<<grid_all, block_all, shared_memory, cuda_stream>>>(gpu_variances.ptr<longlong2>(), histogram, gpu_threshold_sums.ptr<uint>(), gpu_sums.ptr<unsigned long long>());

    shared_memory = n_bins * sizeof(float);
    otsu_score<256><<<grid_score, block_score, shared_memory, cuda_stream>>>(
        otsu_threshold, gpu_threshold_sums.ptr<uint>(), gpu_variances.ptr<longlong2>());
}

// TODO: Replace this is cv::cuda::calcHist
template <uint n_bins>
__global__ void histogram_kernel(
    uint *histogram, const uint8_t *image, uint width,
    uint height, uint pitch)
{
    __shared__ uint local_histogram[n_bins];

    uint x = blockIdx.x * blockDim.x + threadIdx.x;
    uint y = blockIdx.y * blockDim.y + threadIdx.y;
    uint tid = threadIdx.y * blockDim.x + threadIdx.x;

    if (tid < n_bins)
    {
        local_histogram[tid] = 0;
    }

    __syncthreads();

    if (x < width && y < height)
    {
        uint8_t value = image[y * pitch + x];
        atomicInc(&local_histogram[value], 0xFFFFFFFF);
    }

    __syncthreads();

    if (tid < n_bins)
    {
        cv::cudev::atomicAdd(&histogram[tid], local_histogram[tid]);
    }
}

// TODO: Replace this with cv::cuda::calcHist
void calcHist(
    const GpuMat src, GpuMat histogram, Stream stream)
{
    const uint n_bins = 256;

    cudaStream_t cuda_stream = StreamAccessor::getStream(stream);

    dim3 block(128, 4, 1);
    dim3 grid = dim3(divUp(src.cols, block.x), divUp(src.rows, block.y), 1);
    CV_CUDEV_SAFE_CALL(cudaMemsetAsync(histogram.ptr<uint>(), 0, n_bins * sizeof(uint), cuda_stream));
    histogram_kernel<n_bins>
        <<<grid, block, 0, cuda_stream>>>(
            histogram.ptr<uint>(), src.ptr<uint8_t>(), (uint) src.cols, (uint) src.rows, (uint) src.step);
}

double cv::cuda::threshold(InputArray _src, OutputArray _dst, double thresh, double maxVal, int type, Stream &stream)
{
    GpuMat src = getInputMat(_src, stream);

    const int depth = src.depth();

    // TODO: How can we access precomp.hpp to avoid defining THRESH_OTSU?
    const int THRESH_OTSU = 8;
    if ((type & THRESH_OTSU) == THRESH_OTSU)
    {
        CV_Assert(depth == CV_8U);
        CV_Assert(src.channels() == 1);
        CV_Assert(maxVal == 255.0);

        const uint n_bins = 256;
        const uint n_thresholds = 256;

        // Find the threshold using Otsu and then run the normal thresholding algorithm
        GpuMat gpu_histogram = GpuMat(n_bins, 1, CV_32SC1);
        calcHist(src, gpu_histogram, stream);

        GpuMat gpu_otsu_threshold(1, 1, CV_32SC1);
        compute_otsu_async<n_bins, n_thresholds>(gpu_histogram.ptr<uint>(), gpu_otsu_threshold.ptr<uint>(), stream);

        cv::Mat mat_otsu_threshold;
        gpu_otsu_threshold.download(mat_otsu_threshold, stream);
        stream.waitForCompletion();

        // Overwrite the threshold value with the Otsu value and remove the Otsu flag from the type
        type = type & ~THRESH_OTSU;
        thresh = (double) mat_otsu_threshold.at<int>(0);
    }

    CV_Assert( depth <= CV_64F );
    CV_Assert( type <= 4 /*THRESH_TOZERO_INV*/ );

    GpuMat dst = getOutputMat(_dst, src.size(), src.type(), stream);
    src = src.reshape(1);
    dst = dst.reshape(1);

    if (depth == CV_32F && type == 2 /*THRESH_TRUNC*/)
    {
        NppStreamHandler h(StreamAccessor::getStream(stream));

        NppiSize sz;
        sz.width  = src.cols;
        sz.height = src.rows;

#if USE_NPP_STREAM_CTX
        nppSafeCall(nppiThreshold_32f_C1R_Ctx(src.ptr<Npp32f>(), static_cast<int>(src.step),
            dst.ptr<Npp32f>(), static_cast<int>(dst.step), sz, static_cast<Npp32f>(thresh), NPP_CMP_GREATER, h));
#else
        nppSafeCall( nppiThreshold_32f_C1R(src.ptr<Npp32f>(), static_cast<int>(src.step),
            dst.ptr<Npp32f>(), static_cast<int>(dst.step), sz, static_cast<Npp32f>(thresh), NPP_CMP_GREATER) );
#endif

        if (!stream)
            CV_CUDEV_SAFE_CALL( cudaDeviceSynchronize() );
    }
    else
    {
        typedef void (*func_t)(const GpuMat& src, GpuMat& dst, double thresh, double maxVal, int type, Stream& stream);
        static const func_t funcs[] =
        {
            thresholdImpl<uchar>,
            thresholdImpl<schar>,
            thresholdImpl<ushort>,
            thresholdImpl<short>,
            thresholdImpl<int>,
            thresholdImpl<float>,
            thresholdImpl<double>
        };

        if (depth != CV_32F && depth != CV_64F)
        {
            thresh = cvFloor(thresh);
            maxVal = cvRound(maxVal);
        }

        funcs[depth](src, dst, thresh, maxVal, type, stream);
    }

    syncOutput(dst, _dst, stream);

    return thresh;
}

#endif
