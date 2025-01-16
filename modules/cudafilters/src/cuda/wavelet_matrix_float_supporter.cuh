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

#ifndef __OPENCV_WAVELET_MATRIX_FLOAT_SUPPORTER_CUH__
#define __OPENCV_WAVELET_MATRIX_FLOAT_SUPPORTER_CUH__

// The CUB library is used for the Median Filter with Wavelet Matrix,
// which has become a standard library since CUDA 11.
#include "wavelet_matrix_feature_support_checks.h"
#ifdef __OPENCV_USE_WAVELET_MATRIX_FOR_MEDIAN_FILTER_CUDA__


namespace cv { namespace cuda { namespace device
{

namespace wavelet_matrix_median {
namespace WMMedianFloatSupporter {

template<int blockDim, typename IdxT>
__global__ void iota_idx1(IdxT *idx_in_cu, const IdxT hw) {
    const IdxT i = blockIdx.x * blockDim + threadIdx.x;
    if (i >= hw) return;
    idx_in_cu[i] = i;
}

template<int blockDim, int CH_NUM, typename IdxT, typename ValT>
__global__ void split_and_iota_idx(IdxT *idx_in_cu, const ValT* val_in_cu, ValT* val_out_cu, const IdxT hw) {
    const size_t i = blockIdx.x * blockDim + threadIdx.x;
    if (i >= hw) return;

    static_assert(is_same<ValT, float>::value, "");
    // static_assert(2 <= CH_NUM && CH_NUM <= 4);
    using SrcTU = std::conditional_t<CH_NUM == 2, float2, conditional_t<CH_NUM == 3, float3, float4>>;

    const SrcTU *src_u = (SrcTU*)val_in_cu;
    const SrcTU src_uv = src_u[i];

    if (CH_NUM >= 1) { // constexpr
        val_out_cu[i] = src_uv.x;
        idx_in_cu[i] = i;
    }
    if constexpr (CH_NUM >= 2) {
        val_out_cu += hw; idx_in_cu += hw;
        val_out_cu[i] = src_uv.y;
        idx_in_cu[i] = i;
    }
    if constexpr (CH_NUM >= 3) {
        val_out_cu += hw; idx_in_cu += hw;
        val_out_cu[i] = src_uv.z;
        idx_in_cu[i] = i;
    }
    if constexpr (CH_NUM >= 4) {
        val_out_cu += hw; idx_in_cu += hw;
        val_out_cu[i] = src_uv.w;
        idx_in_cu[i] = i;
    }
}

template<int blockDim, typename IdxT>
__global__ void set_wm_val_1(IdxT *wm_src_p, const IdxT *idx_out_cu, const IdxT hw) {
    const IdxT i = blockIdx.x * blockDim + threadIdx.x;
    if (i >= hw) return;
    const IdxT j = idx_out_cu[i];
    wm_src_p[j] = i;
}

template<int blockDim, int CH_NUM, typename IdxT>
__global__ void set_wm_val(IdxT *wm_src_p, const IdxT *idx_out_cu, const IdxT hw, const IdxT buf_byte_div32) {
    const IdxT i = blockIdx.x * blockDim + threadIdx.x;
    if (i >= hw) return;
    const size_t hwc = size_t(hw) * blockIdx.y;
    const IdxT j = idx_out_cu[i + hwc];
    const size_t src_offset = buf_byte_div32 * 32ull / sizeof(IdxT) * blockIdx.y;
    wm_src_p[src_offset + j] = i;
}

template<int blockDim, typename IdxT, typename ValT>
__global__ void conv_res_cu(ValT *dst, const ValT *val_out_cu, const IdxT *res_cu, const IdxT hw) {
    const IdxT i = blockIdx.x * blockDim + threadIdx.x;
    if (i >= hw) return;

    const IdxT r = res_cu[i];
    dst[i] = val_out_cu[r];
}

template<typename ValT, int CH_NUM, typename IdxT>
struct WMMedianFloatSupporter {
    constexpr static int blockDim = 512;
    int h = 0, w = 0;
    int hw_bit_len = -1;
    WMMedianFloatSupporter(){};
    WMMedianFloatSupporter(int h, int w) { reset(h, w); }
    ~WMMedianFloatSupporter(){
        free();
    }
    ValT *val_in_cu = nullptr;
    IdxT *idx_in_cu = nullptr;
    ValT *val_out_cu = nullptr;
    IdxT *idx_out_cu = nullptr;
    void *cub_temp_storage = nullptr;
    size_t cub_temp_storage_bytes;
    // set: val_in
    // get: val_out
    //  1ch
    //  [val_in][idx_in][val_out][idx_out][cub_temp]
    //  C ch

    //  [val_in][......][......][......]
    //  AaBbCcDd
    //  [^^^^^^][......][valin2][idxin2]
    //                  ABCDabcd01230123

    //  [val0in][val1in][idx0in][idx1in][val0out][val1out][idx0out][idx1out][cub_temp][d_offsets]

    void reset(const int H, const int W) {
        h = H; w = W;
        free();
    }
    void alloc(){
        const size_t hwc = size_t(CH_NUM) * h * w;
        if (CH_NUM == 1) { // constexpr
            cub::DeviceRadixSort::SortPairs(
                nullptr, cub_temp_storage_bytes, val_in_cu, val_out_cu, idx_in_cu, idx_out_cu, hwc);
            cudaMalloc(&val_in_cu, 2ull * hwc * (sizeof(ValT) + sizeof(IdxT)) + cub_temp_storage_bytes);
        } else {
            cub::DeviceSegmentedRadixSort::SortPairs(
                nullptr, cub_temp_storage_bytes, val_in_cu, val_out_cu, idx_in_cu, idx_out_cu, hwc, CH_NUM, (int*)nullptr, (int*)nullptr);
            const size_t offsets_arr_size = (CH_NUM + 1) * sizeof(int);
            cudaMalloc(&val_in_cu, 2ull * hwc * (sizeof(ValT) + sizeof(IdxT)) + cub_temp_storage_bytes + offsets_arr_size);
        }
        idx_in_cu = (IdxT*)(val_in_cu + hwc);
        val_out_cu = (ValT*)(idx_in_cu + hwc);
        idx_out_cu = (IdxT*)(val_out_cu + hwc);
        int *d_offsets = (int*)(idx_out_cu + hwc);
        cub_temp_storage = d_offsets + (CH_NUM + 1);
    }
    void free() {
        if (val_in_cu != nullptr) {
            cudaFree(val_in_cu);
        }
    }
    void sort_and_set(IdxT *wm_src_p, const IdxT buf_byte_div32 = 0){
        const IdxT hw = h * w;
        const size_t hwc = size_t(CH_NUM) * hw;
        const dim3 gridDim((hw + blockDim - 1) / blockDim, CH_NUM);

        if (CH_NUM == 1) { // constexpr
            iota_idx1<blockDim><<<gridDim, blockDim>>>(idx_in_cu, hw);
            cub::DeviceRadixSort::SortPairs(
                cub_temp_storage, cub_temp_storage_bytes, val_in_cu, val_out_cu, idx_in_cu, idx_out_cu, hw);
            set_wm_val_1<blockDim><<<gridDim, blockDim>>>(wm_src_p, idx_out_cu, hw);
        } else {
            auto idx2 = idx_out_cu;
            auto val2 = val_out_cu;
            auto idx3 = idx_in_cu;
            auto val3 = val_in_cu;
            split_and_iota_idx<blockDim, CH_NUM><<<gridDim.x, blockDim>>>(idx2, val_in_cu, val2, hw);

            int h_offsets[CH_NUM + 1];
            for (size_t i = 0; i <= CH_NUM; ++i) h_offsets[i] = i * hw;
            int *d_offsets = (int*)(idx_out_cu + hwc);
            cudaMemcpy(d_offsets, h_offsets, (CH_NUM + 1) * sizeof(int), cudaMemcpyHostToDevice);


            cub::DeviceSegmentedRadixSort::SortPairs(
                cub_temp_storage, cub_temp_storage_bytes, val2, val3, idx2, idx3, hwc, CH_NUM, d_offsets, d_offsets + 1);
            set_wm_val<blockDim, CH_NUM><<<gridDim, blockDim>>>(wm_src_p, idx3, hw, buf_byte_div32);
        }
        for(hw_bit_len = 1; ; ++hw_bit_len) {
            if ((1ull << hw_bit_len) >= hw) {
                break;
            }
        }
    }
    const ValT* get_res_table() const {
        if (CH_NUM == 1) { // constexpr
            return val_out_cu;
        } else {
            return val_in_cu;
        }
    }
};
} // end namespace WMMedianFloatSupporter
} // end namespace wavelet_matrix_median

}}} //end namespace cv::cuda::device
#endif
#endif // __OPENCV_WAVELET_MATRIX_FLOAT_SUPPORTER_CUH__
