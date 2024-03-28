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

#ifndef __OPENCV_WAVELET_MATRIX_2D_CUH__
#define __OPENCV_WAVELET_MATRIX_2D_CUH__


// The CUB library is used for the Median Filter with Wavelet Matrix,
// which has become a standard library since CUDA 11.
#include "wavelet_matrix_feature_support_checks.h"
#ifdef __OPENCV_USE_WAVELET_MATRIX_FOR_MEDIAN_FILTER_CUDA__


#include <cub/cub.cuh>
#include <vector>
#include "opencv2/core/cuda/warp_shuffle.hpp"

#include "wavelet_matrix_multi.cuh"

namespace cv { namespace cuda { namespace device
{

namespace wavelet_matrix_median {
    using std::vector;
    using namespace std;

template<int CH_NUM, int ThreadsDimY, int SRC_CACHE_DIV, int SRCB_S, typename SrcT, typename DstT, typename BlockT, typename XYIdxT, typename XIdxT>
__global__ void WaveletMatrix2dCu5C_UpSweep_gpu(const SrcT mask, const uint16_t block_pair_num, const XYIdxT size_div_w, const SrcT* __restrict__ src, DstT* __restrict__ dst, BlockT* __restrict__ nbit_bp, const XYIdxT* __restrict__ nsum_buf_test, XYIdxT* __restrict__ nsum_buf_test2, const uint32_t bv_block_byte_div32, const uint32_t buf_byte_div32, const XIdxT* __restrict__ idx_p, const XIdxT inf, XIdxT* __restrict__ wm, XIdxT* __restrict__ nxt_idx, XYIdxT* __restrict__ wm_nsum_scan_buf, const XYIdxT cwm_buf_byte_div32, BlockT* __restrict__ nbit_bp_pre) {
    using WordT = decltype(BlockT::nbit);
    using WarpWT = uint32_t;
    constexpr int WARP_SIZE = 8 * sizeof(WarpWT);
    static_assert(WARP_SIZE == 32, "");
    static constexpr int THREAD_PER_GRID = ThreadsDimY * WARP_SIZE;
    constexpr int WORD_SIZE = 8 * sizeof(WordT);
    static_assert(WORD_SIZE == 32 || WORD_SIZE == 64, "");
    constexpr uint32_t WORD_DIV_WARP = WORD_SIZE / WARP_SIZE;

    static_assert(ThreadsDimY % SRC_CACHE_DIV == 0, "");
    static_assert(ThreadsDimY != SRC_CACHE_DIV, "Warning: It's not efficient.");

    const size_t buf_byte_y_offset = (size_t)(CH_NUM==1?0:blockIdx.y) * (buf_byte_div32*32ull);
    const size_t bv_block_byte_y_offset = (size_t)(CH_NUM==1?0:blockIdx.y) * (bv_block_byte_div32*32ull);
    const size_t cwm_buf_byte_y_offset = (size_t)(CH_NUM==1?0:blockIdx.y) * cwm_buf_byte_div32 * 32u;
    src = (SrcT*)((uint8_t*)src + buf_byte_y_offset);
    dst = (DstT*)((uint8_t*)dst + buf_byte_y_offset);
    nsum_buf_test = (XYIdxT*)((uint8_t*)nsum_buf_test + buf_byte_y_offset);
    nsum_buf_test2 = (XYIdxT*)((uint8_t*)nsum_buf_test2 + buf_byte_y_offset);
    nbit_bp = (BlockT*)((uint8_t*)nbit_bp + bv_block_byte_y_offset);
    nbit_bp_pre = (BlockT*)((uint8_t*)nbit_bp_pre + bv_block_byte_y_offset);

    idx_p   = (XIdxT*)((uint8_t*)idx_p + buf_byte_y_offset);
    nxt_idx = (XIdxT*)((uint8_t*)nxt_idx + buf_byte_y_offset);
    if (wm != nullptr) wm = (XIdxT*)((uint8_t*)wm + cwm_buf_byte_y_offset);
    wm_nsum_scan_buf = (XYIdxT*)((uint8_t*)wm_nsum_scan_buf + cwm_buf_byte_y_offset);


    using WarpScanX = cub::WarpScan<XYIdxT, WARP_SIZE / SRC_CACHE_DIV>;
    using WarpScanY = cub::WarpScan<XYIdxT, ThreadsDimY>;
    using WarpReduce = cub::WarpReduce<uint32_t>;
    using WarpReduceY = cub::WarpReduce<uint32_t, ThreadsDimY>;

    static_assert(SRCB_S < 64 * 1024, "");

    __shared__ SrcT   src_val_cache[ThreadsDimY][(WARP_SIZE/SRC_CACHE_DIV)-1][WARP_SIZE];
    __shared__ XIdxT vidx_val_cache[ThreadsDimY][(WARP_SIZE/SRC_CACHE_DIV)-1][WARP_SIZE];

    __shared__ uint4 nsum_count_sh[ThreadsDimY];
    __shared__ XYIdxT wm_zero_count_sh[ThreadsDimY];
    __shared__ XYIdxT pre_sum_share[2];
    __shared__ XYIdxT warp_scan_sums[ThreadsDimY];
    __shared__ typename WarpScanX::TempStorage s_scanStorage;
    __shared__ typename WarpScanY::TempStorage s_scanStorage2;
    __shared__ typename WarpReduce::TempStorage WarpReduce_temp_storage[ThreadsDimY];
    __shared__ typename WarpReduceY::TempStorage WarpReduceY_temp_storage;
    // shmem ------ end ------

    XYIdxT wm_zero_count = 0;

    const XYIdxT size_div_warp = size_div_w * WORD_DIV_WARP;
    const XYIdxT nsum = nbit_bp[size_div_w].nsum;
    const XYIdxT nsum_offset = nsum_buf_test[blockIdx.x];
    const XYIdxT nsum_pre = nbit_bp_pre[size_div_w].nsum;


    XYIdxT nsum_idx0_org = nsum_offset;
    XYIdxT nsum_idx1_org = (XYIdxT)blockIdx.x * block_pair_num * THREAD_PER_GRID + nsum - nsum_idx0_org;
    nsum_idx0_org /= (XYIdxT)block_pair_num * ThreadsDimY * WARP_SIZE;
    nsum_idx1_org /= (XYIdxT)block_pair_num * ThreadsDimY * WARP_SIZE;
    const XYIdxT nsum_idx0_bound = (nsum_idx0_org + 1) * block_pair_num * ThreadsDimY * WARP_SIZE;
    const XYIdxT nsum_idx1_bound = (nsum_idx1_org + 1) * block_pair_num * ThreadsDimY * WARP_SIZE;
    uint4 nsum_count = make_uint4(0, 0, 0, 0);

    const unsigned short th_idx = threadIdx.y * WARP_SIZE + threadIdx.x;
    if (th_idx == 0) {
        pre_sum_share[0] = nsum_offset;
    }

    for (XYIdxT ka = 0; ka < block_pair_num; ka += WARP_SIZE / SRC_CACHE_DIV) {
        const XYIdxT ibb = ((XYIdxT)blockIdx.x * block_pair_num + ka) * ThreadsDimY;
        if (ibb >= size_div_warp) break;

        WarpWT my_bits = 0;
        SrcT  first_val;
        XIdxT first_idxval;

        for (XYIdxT kb = 0, i = ibb + WARP_SIZE / SRC_CACHE_DIV * threadIdx.y; kb < WARP_SIZE / SRC_CACHE_DIV; ++kb, ++i) {
            if (i >= size_div_warp) break;
            WarpWT bits;
            const XYIdxT ij = i * WARP_SIZE + threadIdx.x;
            const SrcT v = src[ij];
            const XIdxT wm_idxv = idx_p[ij];
            if (kb == 0) {
                first_val = v;
                first_idxval = wm_idxv;
            } else {
                src_val_cache[threadIdx.y][kb - 1][threadIdx.x] = v;
                vidx_val_cache[threadIdx.y][kb - 1][threadIdx.x] = wm_idxv;
            }
            if (v <= mask) {
                bits = __activemask();
            } else {
                bits = ~__activemask();
            }
            if (threadIdx.x == kb) {
                my_bits = bits;
            }
            if (wm != nullptr) {
                if (ij < nsum_pre) {
                    wm[ij] = wm_idxv;
                    if (wm_idxv * 2 <= inf) {
                        ++wm_zero_count;
                    }
                } else {
                    wm[ij] = inf;
                }
            }
        }

        XYIdxT c, t = 0;
        if (threadIdx.y < ThreadsDimY) {
            c = __popc(my_bits);

            WarpScanX(s_scanStorage).ExclusiveSum(c, t);
            if (threadIdx.x == WARP_SIZE / SRC_CACHE_DIV - 1) {
                warp_scan_sums[threadIdx.y] = c + t;
            }
        }

        __syncthreads();

        XYIdxT pre_sum = pre_sum_share[(ka & (WARP_SIZE / SRC_CACHE_DIV)) > 0 ? 1 : 0];
        XYIdxT s = threadIdx.x < ThreadsDimY ? warp_scan_sums[threadIdx.x] : 0;
        WarpScanY(s_scanStorage2).ExclusiveSum(s, s);

        s = cv::cuda::device::shfl(s, threadIdx.y, WARP_SIZE);
        s += t + pre_sum;

        if (SRC_CACHE_DIV == 1 || threadIdx.x < WARP_SIZE / SRC_CACHE_DIV) {
            if (th_idx == THREAD_PER_GRID - WARP_SIZE + WARP_SIZE / SRC_CACHE_DIV - 1) {
                pre_sum_share[(ka & (WARP_SIZE / SRC_CACHE_DIV)) == 0 ? 1 : 0] = s + c;
            }
            const XYIdxT bi = ibb + threadIdx.y * WARP_SIZE / SRC_CACHE_DIV + threadIdx.x;
            if (bi < size_div_warp) {
                static_assert(WORD_SIZE == 32, "");
                nbit_bp[bi] = BlockT{s, my_bits};
            }
        }

        if (mask == 0) {
            SrcT  vo = first_val;
            XIdxT idx_v = first_idxval;
            for (XYIdxT j = 0, i = ibb + WARP_SIZE / SRC_CACHE_DIV * threadIdx.y; j < WARP_SIZE / SRC_CACHE_DIV; ++j, ++i) {
                if (i >= size_div_warp) break;
                const WarpWT e_nbit = cv::cuda::device::shfl(my_bits, j, WARP_SIZE);
                const XYIdxT e_nsum = cv::cuda::device::shfl(s, j, WARP_SIZE);
                XYIdxT rank = __popc(e_nbit << (WARP_SIZE - threadIdx.x));
                const XYIdxT idx0 = e_nsum + rank;
                XYIdxT idx = idx0;
                if (vo > mask) { // 1
                    const XYIdxT ij = i * WARP_SIZE + threadIdx.x;
                    idx = ij + nsum - idx;
                }
                if (idx < size_div_warp * WARP_SIZE) {
                    nxt_idx[idx] = idx_v;
                }
                if (j == WARP_SIZE / SRC_CACHE_DIV - 1) break;
                vo    = src_val_cache[threadIdx.y][j][threadIdx.x];
                idx_v = vidx_val_cache[threadIdx.y][j][threadIdx.x];
            }
            continue;
        }
        const SrcT mask_2 = mask >> 1;
        SrcT  vo = first_val;
        XIdxT idx_v = first_idxval;
        for (XYIdxT j = 0, i = ibb + WARP_SIZE / SRC_CACHE_DIV * threadIdx.y; j < WARP_SIZE / SRC_CACHE_DIV; ++j, ++i) {
            if (i >= size_div_warp) break;
            const WarpWT e_nbit = cv::cuda::device::shfl(my_bits, j, WARP_SIZE);
            const XYIdxT e_nsum = cv::cuda::device::shfl(s, j, WARP_SIZE);
            XYIdxT rank = __popc(e_nbit << (WARP_SIZE - threadIdx.x));
            const XYIdxT idx0 = e_nsum + rank;

            DstT v = (DstT)vo;
            XYIdxT idx = idx0;
            if (vo > mask) { // 1
                const XYIdxT ij = i * WARP_SIZE + threadIdx.x;
                idx = ij + nsum - idx;
                v &= mask;
            }
            if (idx < size_div_warp * WARP_SIZE) {
                if (mask != 0) {
                    dst[idx] = v;
                }
                nxt_idx[idx] = idx_v;
            }

            if (v <= mask_2) {
                if (vo <= mask) {
                    if (idx < nsum_idx0_bound) {
                        nsum_count.x++;
                    } else {
                        nsum_count.y++;
                    }
                } else {
                    if (idx < nsum_idx1_bound) {
                        nsum_count.z++;
                    } else {
                        nsum_count.w++;
                    }
                }
            }
            if (j == WARP_SIZE / SRC_CACHE_DIV - 1) break;
            vo    = src_val_cache[threadIdx.y][j][threadIdx.x];
            idx_v = vidx_val_cache[threadIdx.y][j][threadIdx.x];
        }
    }
    if (blockIdx.x == gridDim.x - 1 && th_idx == 0) {
        nbit_bp[size_div_warp / WORD_DIV_WARP].nsum = nsum;
    }

    nsum_count.x = WarpReduce(WarpReduce_temp_storage[threadIdx.y]).Sum(nsum_count.x);
    nsum_count.y = WarpReduce(WarpReduce_temp_storage[threadIdx.y]).Sum(nsum_count.y);
    nsum_count.z = WarpReduce(WarpReduce_temp_storage[threadIdx.y]).Sum(nsum_count.z);
    nsum_count.w = WarpReduce(WarpReduce_temp_storage[threadIdx.y]).Sum(nsum_count.w);
    wm_zero_count = WarpReduce(WarpReduce_temp_storage[threadIdx.y]).Sum(wm_zero_count);
    if (threadIdx.x == 0) {
        nsum_count_sh[threadIdx.y]    = nsum_count;
        wm_zero_count_sh[threadIdx.y] = wm_zero_count;
    }
    __syncthreads();
    if (threadIdx.x < ThreadsDimY) {
        nsum_count = nsum_count_sh[threadIdx.x];
        nsum_count.x = WarpReduceY(WarpReduceY_temp_storage).Sum(nsum_count.x);
        nsum_count.y = WarpReduceY(WarpReduceY_temp_storage).Sum(nsum_count.y);
        nsum_count.z = WarpReduceY(WarpReduceY_temp_storage).Sum(nsum_count.z);
        nsum_count.w = WarpReduceY(WarpReduceY_temp_storage).Sum(nsum_count.w);
        wm_zero_count = WarpReduceY(WarpReduceY_temp_storage).Sum(wm_zero_count_sh[threadIdx.x]);
        if (th_idx == 0) {
            const XYIdxT nsum_idx0_org = nsum_idx0_bound / ((XYIdxT)block_pair_num * ThreadsDimY * WARP_SIZE);
            const XYIdxT nsum_idx1_org = nsum_idx1_bound / ((XYIdxT)block_pair_num * ThreadsDimY * WARP_SIZE);
            if (mask != 0) {
                if (nsum_count.x > 0) atomicAdd(nsum_buf_test2 + nsum_idx0_org - 1, nsum_count.x);
                if (nsum_count.y > 0) atomicAdd(nsum_buf_test2 + nsum_idx0_org - 0, nsum_count.y);
                if (nsum_count.z > 0) atomicAdd(nsum_buf_test2 + nsum_idx1_org - 1, nsum_count.z);
                if (nsum_count.w > 0) atomicAdd(nsum_buf_test2 + nsum_idx1_org - 0, nsum_count.w);
            }
            if (wm != nullptr) {
                wm_nsum_scan_buf[blockIdx.x] = wm_zero_count;
            }
        }
    }
}

template<int CH_NUM, int ThreadsDimY, typename BlockT, typename XYIdxT, typename XIdxT>
__global__ void WaveletMatrix2dCu5C_last_gpu(const uint16_t block_pair_num, const XYIdxT size_div_w, const uint32_t buf_byte_div32, const XIdxT* __restrict__ idx_p, const XIdxT inf, XIdxT* __restrict__ wm, XYIdxT* __restrict__ wm_nsum_scan_buf, const XYIdxT cwm_buf_byte_div32, BlockT* __restrict__ nbit_bp_pre, const uint32_t bv_block_byte_div32) {
    using WordT = decltype(BlockT::nbit);
    using WarpWT = uint32_t;
    constexpr int WARP_SIZE = 8 * sizeof(WarpWT);
    static_assert(WARP_SIZE == 32, "");
    static constexpr int THREAD_PER_GRID = ThreadsDimY * WARP_SIZE;
    constexpr int WORD_SIZE = 8 * sizeof(WordT);
    static_assert(WORD_SIZE == 32 || WORD_SIZE == 64, "");
    constexpr uint32_t WORD_DIV_WARP = WORD_SIZE / WARP_SIZE;

    const size_t buf_byte_y_offset = (size_t)(CH_NUM==1?0:blockIdx.y) * (buf_byte_div32*32ull);
    const size_t bv_block_byte_y_offset = (size_t)(CH_NUM==1?0:blockIdx.y) * (bv_block_byte_div32*32ull);
    const size_t cwm_buf_byte_y_offset = (size_t)(CH_NUM==1?0:blockIdx.y) * cwm_buf_byte_div32 * 32u;

    idx_p   = (XIdxT*)((uint8_t*)idx_p + buf_byte_y_offset);
    wm = (XIdxT*)((uint8_t*)wm + cwm_buf_byte_y_offset);
    wm_nsum_scan_buf = (XYIdxT*)((uint8_t*)wm_nsum_scan_buf + cwm_buf_byte_y_offset);
    nbit_bp_pre = (BlockT*)((uint8_t*)nbit_bp_pre + bv_block_byte_y_offset);
    const XYIdxT nsum_pre = nbit_bp_pre[size_div_w].nsum;

    using WarpReduce = cub::WarpReduce<uint32_t>;
    using WarpReduceY = cub::WarpReduce<uint32_t, ThreadsDimY>;

    __shared__ XYIdxT wm_zero_count_sh[ThreadsDimY];
    __shared__ typename WarpReduce::TempStorage WarpReduce_temp_storage[ThreadsDimY];
    __shared__ typename WarpReduceY::TempStorage WarpReduceY_temp_storage;
    // shmem ------ end ------

    XYIdxT wm_zero_count = 0;
    const XYIdxT size_div_warp = size_div_w * WORD_DIV_WARP;
    const unsigned short th_idx = threadIdx.y * WARP_SIZE + threadIdx.x;

    const int block_num = block_pair_num/WARP_SIZE;
    for (XYIdxT ka = 0; ka < block_num; ++ka) {
        const XYIdxT ibb = ((XYIdxT)blockIdx.x * block_num + ka) * THREAD_PER_GRID + WARP_SIZE * threadIdx.y;
        if (ibb >= size_div_warp) break;

        for (XYIdxT kb = 0; kb < WARP_SIZE; ++kb) {
            XYIdxT i = ibb + kb;
            if (i >= size_div_warp) break;

            const XYIdxT ij = i * WARP_SIZE + threadIdx.x;

            if (ij < nsum_pre) {
                const XIdxT wm_idxv = idx_p[ij];
                wm[ij] = wm_idxv;
                if (wm_idxv * 2 <= inf) {
                    ++wm_zero_count;
                }
            } else {
                wm[ij] = inf;
            }
        }
    }
    wm_zero_count = WarpReduce(WarpReduce_temp_storage[threadIdx.y]).Sum(wm_zero_count);
    if (threadIdx.x == 0) {
        wm_zero_count_sh[threadIdx.y] = wm_zero_count;
    }
    __syncthreads();
    if (threadIdx.x < ThreadsDimY) {
        wm_zero_count = WarpReduceY(WarpReduceY_temp_storage).Sum(wm_zero_count_sh[threadIdx.x]);
        if (th_idx == 0) {
            wm_nsum_scan_buf[blockIdx.x] = wm_zero_count;
        }
    }
}

template<int CH_NUM, int MAX_BLOCK_X, typename XYIdxT, typename BlockT>
__global__ void WaveletMatrix2dCu5C_ExclusiveSum(XYIdxT* __restrict__ nsum_scan_buf, XYIdxT* __restrict__ nsum_buf_test2, BlockT* __restrict__ nsum_p, const uint32_t buf_byte_div32, const uint32_t bv_block_byte_div32) {

    typedef cub::BlockScan<XYIdxT, MAX_BLOCK_X> BlockScan;
    __shared__ typename BlockScan::TempStorage temp_storage;

    XYIdxT thread_data1;
    XYIdxT thread_data2;

    nsum_scan_buf = (XYIdxT*)((uint8_t*)nsum_scan_buf + (size_t)blockIdx.x * (buf_byte_div32*32ull));
    nsum_buf_test2 = (XYIdxT*)((uint8_t*)nsum_buf_test2 + (size_t)blockIdx.x * (buf_byte_div32*32ull));

    thread_data1 = nsum_scan_buf[threadIdx.x];
    BlockScan(temp_storage).ExclusiveSum(thread_data1, thread_data2);

    nsum_scan_buf[threadIdx.x] = thread_data2;
    nsum_buf_test2[threadIdx.x] = 0;

    if (threadIdx.x == blockDim.x - 1) {
        thread_data2 += thread_data1;
        nsum_p = (BlockT*)((uint8_t*)nsum_p + (size_t)blockIdx.x * (bv_block_byte_div32*32ull));
        nsum_p->nsum = thread_data2;
    }
}


template<int CH_NUM, int ThreadsDimY, bool MOV_SRC,typename SrcT, typename XYIdxT, typename XIdxT>
__global__ void WaveletMatrix2dCu5C_first_gpu_multi(const SrcT mask, uint16_t block_pair_num, const XYIdxT size_div_warp, SrcT* __restrict__ src, SrcT* __restrict__ dst, XYIdxT* __restrict__ nsum_scan_buf, const uint32_t buf_byte_div32, XIdxT* __restrict__ buf_idx, const int W, const XYIdxT WH, const SrcT* __restrict__ src_const) {
    using WarpWT = uint32_t;
    constexpr int WARP_SIZE = 8 * sizeof(WarpWT);
    static_assert(WARP_SIZE == 32, "");
    static constexpr int THREAD_PER_GRID = ThreadsDimY * WARP_SIZE;

    SrcT* __restrict__ dsts[CH_NUM];
    XIdxT* __restrict__ buf_idxs[CH_NUM];
    XYIdxT cs[CH_NUM];

    __shared__ SrcT src_vbuf_org[ThreadsDimY][CH_NUM * WARP_SIZE];
    SrcT* __restrict__ src_vbuf = src_vbuf_org[threadIdx.y];

    for (int c = 0; c < CH_NUM; ++c) {
        if (CH_NUM > 1) { // constexpr
            dsts[c]     = (SrcT*)((uint8_t*)dst + (size_t)c * (buf_byte_div32*32ull));
        }
        buf_idxs[c] = (XIdxT*)((uint8_t*)buf_idx + (size_t)c * (buf_byte_div32*32ull));
        cs[c] = 0;
    }

    XYIdxT ibb = (XYIdxT)blockIdx.x * block_pair_num * ThreadsDimY;
    for (XYIdxT ka = 0; ka < block_pair_num; ka += WARP_SIZE, ibb += THREAD_PER_GRID) {
        for (XYIdxT kb = 0, i = ibb + WARP_SIZE * threadIdx.y; kb < WARP_SIZE; ++kb, ++i) {
            if (i >= size_div_warp) break;
            const XYIdxT iw = i * WARP_SIZE;
            const XYIdxT idx = iw + threadIdx.x;
            const XIdxT idx_v = (idx >= WH ? 0 : idx % W);

            for (int c = 0; c < CH_NUM; ++c) {
                if (MOV_SRC) { // constexpr
                    const XYIdxT s_idx = iw * CH_NUM + threadIdx.x + c * WARP_SIZE;
                    src_vbuf[threadIdx.x + c * WARP_SIZE] = src[s_idx] = src_const[s_idx];
                } else {
                    src_vbuf[threadIdx.x + c * WARP_SIZE] = src[iw * CH_NUM + threadIdx.x + c * WARP_SIZE];
                }
            }
            __syncwarp();
            for (int c = 0; c < CH_NUM; ++c) {
                const SrcT v = src_vbuf[threadIdx.x * CH_NUM + c];
                if (v <= mask) {
                    ++cs[c];
                }
                buf_idxs[c][idx] = idx_v;
                if constexpr(CH_NUM > 1) {
                    dsts[c][idx] = v;
                }
            }
        }
    }

    using WarpReduce = cub::WarpReduce<uint32_t>;
    __shared__ typename WarpReduce::TempStorage WarpReduce_temp_storage[ThreadsDimY];
    __shared__ XYIdxT cs_sum_sh[CH_NUM][ThreadsDimY];
    for (int c = 0; c < CH_NUM; ++c) {
        cs[c] = WarpReduce(WarpReduce_temp_storage[threadIdx.y]).Sum(cs[c]);
    }
    if (threadIdx.x == 0) {
        for (int c = 0; c < CH_NUM; ++c) {
            cs_sum_sh[c][threadIdx.y] = cs[c];
        }
    }
    __syncthreads();
    if (threadIdx.y != 0) return;
    for (int c = 0; c < CH_NUM; ++c) {
        XYIdxT cs_bsum = (threadIdx.x < ThreadsDimY ? cs_sum_sh[c][threadIdx.x] : 0);
        cs_bsum = WarpReduce(WarpReduce_temp_storage[0]).Sum(cs_bsum);
        if (threadIdx.x == 0) {
            nsum_scan_buf[blockIdx.x] = cs_bsum;
            nsum_scan_buf = (XYIdxT*)((uint8_t*)nsum_scan_buf + (buf_byte_div32*32ull));
        }
    }

}


template<int ThreadsDimY, typename SrcT, typename XYIdxT, typename XIdxT>
__global__ void WaveletMatrix2dCu5C_first_gpu_multi_srcunpacked(const SrcT mask, uint16_t block_pair_num, const XYIdxT size_div_warp, const SrcT* __restrict__ src, XYIdxT* __restrict__ nsum_scan_buf, const uint32_t buf_byte_div32, XIdxT* __restrict__ buf_idx, const int W, const XYIdxT WH) {
    using WarpWT = uint32_t;
    constexpr int WARP_SIZE = 8 * sizeof(WarpWT);
    static_assert(WARP_SIZE == 32, "");
    static constexpr int THREAD_PER_GRID = ThreadsDimY * WARP_SIZE;

    XYIdxT cs = 0;

    const int c = blockIdx.y;
    buf_idx = buf_idx + c * (buf_byte_div32*32ull / sizeof(XIdxT));
    src = src + c * (buf_byte_div32*32ull / sizeof(SrcT));


    XYIdxT i = (XYIdxT)blockIdx.x * block_pair_num * ThreadsDimY + threadIdx.y;

    XIdxT x_idx = (i * WARP_SIZE + threadIdx.x) % W;
    const XIdxT x_diff = THREAD_PER_GRID % W;

    for (XYIdxT k = 0; k < block_pair_num; ++k, i += ThreadsDimY) {
        if (i >= size_div_warp) break;
        const XYIdxT idx = i * WARP_SIZE + threadIdx.x;

        if (idx >= WH) x_idx = 0;

        const SrcT v = src[idx];
        if (v <= mask) {
            ++cs;
        }
        buf_idx[idx] = x_idx;

        x_idx += x_diff;
        if (x_idx >= W) x_idx -= W;
    }

    using WarpReduce = cub::WarpReduce<uint32_t>;
    __shared__ typename WarpReduce::TempStorage WarpReduce_temp_storage[ThreadsDimY];
    __shared__ XYIdxT cs_sum_sh[ThreadsDimY];
    cs = WarpReduce(WarpReduce_temp_storage[threadIdx.y]).Sum(cs);
    if (threadIdx.x == 0) {
        cs_sum_sh[threadIdx.y] = cs;
    }
    __syncthreads();
    if (threadIdx.y != 0) return;
    XYIdxT cs_bsum = (threadIdx.x < ThreadsDimY ? cs_sum_sh[threadIdx.x] : 0);
    cs_bsum = WarpReduce(WarpReduce_temp_storage[0]).Sum(cs_bsum);
    if (threadIdx.x == 0) {
        nsum_scan_buf += (buf_byte_div32*32ull / sizeof(XYIdxT)) * (blockIdx.y);
        nsum_scan_buf[blockIdx.x] = cs_bsum;
    }
}

template<typename IdxType, typename BlockT>
__device__
inline IdxType WaveletMatrix2dCu5C_median2d_rank0(const IdxType i, const BlockT* __restrict__ nbit_bp) {
    using WordT = decltype(BlockT::nbit);
    constexpr int WORD_SIZE = 8 * sizeof(WordT);
    static_assert(WORD_SIZE == 32 || WORD_SIZE == 64, "");

    const IdxType bi = i / WORD_SIZE;

    const int ai = i % WORD_SIZE;
    const BlockT block = nbit_bp[bi];
    if constexpr(WORD_SIZE == 32) {
        return block.nsum + __popc(block.nbit & ((1u << ai) - 1));
    }
    if constexpr(WORD_SIZE == 64) {
        return block.nsum + __popcll(block.nbit & ((1ull << ai) - 1ull));
    }
}


template<int CH_NUM, int THREADS_NUM_W, int THREADS_NUM_H, int VAL_BIT_LEN, bool CUT_BORDER, typename XYIdxT, typename XIdxT, typename ValT, typename BlockT, typename ResT, typename ResTableT>
__global__ void WaveletMatrix2dCu5C_median2d_cu(
    const int H, const int W, const int res_step_num, const int r, ResT* __restrict__ res_cu, const BlockT* __restrict__ wm_nbit_bp, const uint32_t nsum_pos,
    const uint32_t bv_block_h_byte_div32, const uint32_t bv_block_len,
    const BlockT* __restrict__ bv_nbit_bp, const uint8_t w_bit_len, const uint8_t val_bit_len,
    const ResTableT* __restrict__ res_table
) {

    const int y = blockIdx.y * THREADS_NUM_H + threadIdx.y;
    if (y >= H) return;
    const int x = blockIdx.x * THREADS_NUM_W + threadIdx.x;
    if (x >= W) return;

    if (CH_NUM >= 2) { // constexpr
        bv_nbit_bp = (BlockT*)((uint8_t*)bv_nbit_bp + bv_block_h_byte_div32 * 32ull * blockIdx.z * (VAL_BIT_LEN >= 0 ? VAL_BIT_LEN : val_bit_len)); // TODO
        wm_nbit_bp = (BlockT*)((uint8_t*)wm_nbit_bp + bv_block_h_byte_div32 * 32ull * blockIdx.z * w_bit_len);
    }

    XYIdxT ya, yb, k;
    XIdxT  xa, xb;
    if (CUT_BORDER) { // constexpr
        ya = y;
        xa = x;
        yb = y + r * 2 + 1;
        xb = x + r * 2 + 1;
        k = (r * 2 + 1) * (r * 2 + 1) / 2;
    } else {
        ya = (y < r ? 0 : y - r);
        xa = (x < r ? 0 : x - r);
        yb = y + r + 1; if (yb > H) yb = H;
        xb = x + r + 1; if (xb > W) xb = W;
        k = XYIdxT(yb - ya) * (xb - xa) / 2;
    }
    ValT res = 0;
    ya *= (CUT_BORDER ? W + 2 * r : W);
    yb *= (CUT_BORDER ? W + 2 * r : W);

    for (int8_t h = (VAL_BIT_LEN >= 0 ? VAL_BIT_LEN : val_bit_len); h--; ) {
        const XYIdxT top0 = WaveletMatrix2dCu5C_median2d_rank0(ya, bv_nbit_bp);
        const XYIdxT bot0 = WaveletMatrix2dCu5C_median2d_rank0(yb, bv_nbit_bp);
        XYIdxT l_ya_xa = top0;
        XYIdxT l_yb_xa = bot0;
        XYIdxT l_ya_xb = top0;
        XYIdxT l_yb_xb = bot0;
        XYIdxT d = 0;
        for (int8_t j = w_bit_len; j--; ) {
            const XYIdxT zeros = wm_nbit_bp[nsum_pos].nsum;
            const XYIdxT l_ya_xa_rank0 = WaveletMatrix2dCu5C_median2d_rank0(l_ya_xa, wm_nbit_bp);
            const XYIdxT l_ya_xb_rank0 = WaveletMatrix2dCu5C_median2d_rank0(l_ya_xb, wm_nbit_bp);
            const XYIdxT l_yb_xb_rank0 = WaveletMatrix2dCu5C_median2d_rank0(l_yb_xb, wm_nbit_bp);
            const XYIdxT l_yb_xa_rank0 = WaveletMatrix2dCu5C_median2d_rank0(l_yb_xa, wm_nbit_bp);

            if (((xa >> j) & 1) == 0) {
                l_ya_xa = l_ya_xa_rank0;
                l_yb_xa = l_yb_xa_rank0;
            } else {
                d += l_ya_xa_rank0; l_ya_xa += zeros - l_ya_xa_rank0;
                d -= l_yb_xa_rank0; l_yb_xa += zeros - l_yb_xa_rank0;
            }
            if (((xb >> j) & 1) == 0) {
                l_ya_xb = l_ya_xb_rank0;
                l_yb_xb = l_yb_xb_rank0;
            } else {
                d -= l_ya_xb_rank0; l_ya_xb += zeros - l_ya_xb_rank0;
                d += l_yb_xb_rank0; l_yb_xb += zeros - l_yb_xb_rank0;
            }
            wm_nbit_bp = (BlockT*)((uint8_t*)wm_nbit_bp - bv_block_h_byte_div32 * 32ull);
        }
        if (CH_NUM >= 2) {
            wm_nbit_bp = (BlockT*)((uint8_t*)wm_nbit_bp - bv_block_h_byte_div32 * 32ull * w_bit_len * (CH_NUM - 1));
        }
        const XYIdxT bv_h_zeros = bv_nbit_bp[nsum_pos].nsum;
        if (k < d) {
            ya = top0;
            yb = bot0;
        } else {
            k -= d;
            res |= (ValT)1 << h;
            ya += bv_h_zeros - top0;
            yb += bv_h_zeros - bot0;
        }
        bv_nbit_bp = (BlockT*)((uint8_t*)bv_nbit_bp - bv_block_h_byte_div32 * 32ull);
    }




    if constexpr(is_same<ResTableT, std::nullptr_t>::value) {
        res_cu[(XYIdxT)y * res_step_num + x * CH_NUM + blockIdx.z] = res;
    } else if (CH_NUM == 1){
        res_cu[(XYIdxT)y * res_step_num + x * CH_NUM] = res_table[res];
    } else {
        const size_t offset = size_t(CUT_BORDER ? W + 2 * r : W) * (CUT_BORDER ? H + 2 * r : H) * blockIdx.z;
        res_cu[(XYIdxT)y * res_step_num + x * CH_NUM + blockIdx.z] = res_table[res + offset];
    }
}




template <typename ValT, int CH_NUM, class MultiWaveletMatrixImpl, int TH_NUM = 512, int WORD_SIZE = 32>
struct WaveletMatrix2dCu5C {
    static_assert(is_same<ValT, uint32_t>() || is_same<ValT, uint16_t>() || is_same<ValT, uint8_t>(), "Supports 32, 16, or 8 bits only");
    static constexpr int MAX_BIT_LEN = 8 * sizeof(ValT);

    static constexpr uint32_t WSIZE = WORD_SIZE;
    static constexpr int WARP_SIZE = 32;
    using T_Type = ValT;
    static constexpr int THREAD_PER_GRID = TH_NUM;
    static constexpr int SRC_CACHE_DIV = 2;
    static constexpr int MAX_BLOCK_X = MultiWaveletMatrixImpl::MAX_BLOCK_X;
    static_assert(WORD_SIZE == 32 || WORD_SIZE == 64, "WORD_SIZE must be 32 or 64");
    using WordT = typename std::conditional<WORD_SIZE == 32, uint32_t, uint64_t>::type;

    static_assert(MAX_BLOCK_X <= 1024, "");
    static_assert(TH_NUM == 1024 || TH_NUM == 512 || TH_NUM == 256 || TH_NUM == 128 || TH_NUM == 64 || TH_NUM == 32, "");
    static_assert(THREAD_PER_GRID == MultiWaveletMatrixImpl::THREAD_PER_GRID, "");

    using BlockT = typename MultiWaveletMatrixImpl::BlockT;
    using WarpWT = uint32_t;
    using XIdxT = uint16_t;
    using YIdxT = uint16_t;
    using XYIdxT = uint32_t;
    static constexpr int BLOCK_TYPE = 2;
    using MultiWaveletMatrixImplClass = MultiWaveletMatrixImpl;
    static_assert(is_same<XIdxT, typename MultiWaveletMatrixImpl::T_Type>::value, "");
    static_assert(8 * sizeof(WarpWT) == WARP_SIZE, "");

    int H, W;
    XYIdxT size = 0;
    MultiWaveletMatrixImpl WM;
    XYIdxT bv_zeros[MAX_BIT_LEN];

    int w_bit_len = 0;
    int val_bit_len = 0;
    static constexpr int wm_num = CH_NUM;

private:
    uint8_t* bv_block_nbit_and_nsum_base_cu = nullptr; // GPU mem
    uint32_t bv_block_byte_div32;
    uint32_t buf_byte_div32;
    uint32_t nsum_scan_buf_len;
    size_t input_buf_byte;
public:
    ValT* src_cu = nullptr; // GPU mem
    ValT* res_cu = nullptr;
    size_t bv_block_len = 0;
    size_t bv_chunk_len = 0;

#if _MSC_VER >= 1920 || __INTEL_COMPILER
    inline static int bitCount64(uint64_t bits) {
        return (int)_mm_popcnt_u64(bits);
    }
#else
    inline static int bitCount64(uint64_t bits) {
        return __builtin_popcountll(bits);
    }
#endif
    static constexpr int get_bit_len(uint64_t val) {
        return (
            (val |= val >> 1),
            (val |= val >> 2),
            (val |= val >> 4),
            (val |= val >> 8),
            (val |= val >> 16),
            (val |= val >> 32),
            bitCount64(val));
        // val |= val >> 1;
        // val |= val >> 2;
        // val |= val >> 4;
        // val |= val >> 8;
        // val |= val >> 16;
        // val |= val >> 32;
        // return bitCount64(val);
    }

    WaveletMatrix2dCu5C() {
        reset(0, 0);
    }
    WaveletMatrix2dCu5C(const int rows, const int cols, const bool use_hw_bit_len = false, const bool alloc_res = true) {
        reset(rows, cols, use_hw_bit_len, alloc_res);
    }

    void reset(const int rows, const int cols, const bool use_hw_bit_len = false, const bool alloc_res = true) {
        H = rows;
        W = cols;
        if (rows == 0 || cols == 0) return;
        val_bit_len = (use_hw_bit_len ? get_bit_len((uint64_t)H * W - 1) : MAX_BIT_LEN);
        assert(size == 0 && src_cu == nullptr);

        size = div_ceil<size_t>((uint64_t)H * W, WORD_SIZE) * WORD_SIZE;
        assert(W < 65535); // That is, less than 65534.
        w_bit_len = get_bit_len(W); // w=7 [0,6] bl=3; w=8 [0,7] bl=4
        WM.reset(size, w_bit_len, val_bit_len * wm_num);
        if (val_bit_len == 0) return;


        bv_block_len = div_ceil<size_t>(size, THREAD_PER_GRID) * THREAD_PER_GRID / WORD_SIZE + 1;
        bv_block_len = div_ceil<size_t>(bv_block_len, 8*2) * 8*2;
        const size_t bv_block_byte = (sizeof(BlockT)) * val_bit_len * bv_block_len;
        bv_block_byte_div32 = div_ceil<size_t>(bv_block_byte, 32);

        cudaMalloc(&bv_block_nbit_and_nsum_base_cu, (size_t)(bv_block_byte_div32*32ull) * CH_NUM);
        if (bv_block_nbit_and_nsum_base_cu == nullptr) { printf("GPU Memory Alloc Error! %s:%d\n", __FILE__, __LINE__); release(); return; }

        const uint16_t block_pair_num = get_block_pair_num();
        nsum_scan_buf_len = div_ceil<size_t>(size, (size_t)THREAD_PER_GRID * block_pair_num);
        nsum_scan_buf_len = div_ceil<size_t>(nsum_scan_buf_len, 4) * 4;

        const size_t buf_byte =
            sizeof(XYIdxT) * 2 * nsum_scan_buf_len
            + sizeof(XIdxT) * 2 * size
            + sizeof(ValT) * (CH_NUM == 1 ? 1 : 2) * size;
        buf_byte_div32 = div_ceil<size_t>(buf_byte, 32);


        input_buf_byte = sizeof(ValT) * size * CH_NUM;
        cudaMalloc(&src_cu, (size_t)(buf_byte_div32*32ull) * CH_NUM + input_buf_byte);
        if (src_cu == nullptr) { printf("GPU Memory Alloc Error! %s:%d\n", __FILE__, __LINE__); release(); return; }

        if (alloc_res) {
            cudaMalloc(&res_cu, sizeof(ValT) * size * CH_NUM);
            if (res_cu == nullptr) { printf("GPU Memory Alloc Error! %s:%d\n", __FILE__, __LINE__); release(); return; }
        }
    }
    void release() {
        size = 0;
        if (src_cu != nullptr) cudaFree(src_cu);
        if (bv_block_nbit_and_nsum_base_cu != nullptr) cudaFree(bv_block_nbit_and_nsum_base_cu);
        if (res_cu != nullptr) cudaFree(res_cu);
        src_cu = nullptr;
        bv_block_nbit_and_nsum_base_cu = nullptr;
        res_cu = nullptr;
    }
    ~WaveletMatrix2dCu5C() { release(); }

    BlockT*  get_bv_block_cu(int h) const { return (BlockT*)(bv_block_nbit_and_nsum_base_cu + (bv_block_len * (sizeof(BlockT))) * h); }

    BlockT*  get_bv_block_cu(int h, int c) const { return (BlockT*)((uint8_t*)get_bv_block_cu(h) + (size_t)c * (bv_block_byte_div32*32ull)); }


    uint16_t get_block_pair_num() const {
        return WM.get_block_pair_num() * MultiWaveletMatrixImpl::THREAD_PER_GRID / THREAD_PER_GRID;
    }
    std::pair<ValT*, XYIdxT> get_nowcu_and_buf_byte_div32() {
        ValT *now_cu = src_cu + (CH_NUM == 1 ? 0ull : size * (size_t)CH_NUM);
        return make_pair(now_cu, buf_byte_div32);
    }

    // Set data in src_cu before calling (data will be destroyed). Or set src_cu_const.
    void construct(const ValT *src_cu_const = nullptr, const cudaStream_t main_stream = 0, const bool src_unpacked = false) {
        assert(size > 0 && src_cu != nullptr);
        if (val_bit_len == 0) return;
        if (src_cu == nullptr) { printf("Build Error: memory not alloced."); return;}

        const XIdxT inf = ((XIdxT)1u << w_bit_len) - 1;
        assert(W <= inf);
        assert(size % WORD_SIZE == 0);

        ValT mask = ((ValT)1u << val_bit_len) - 1;

        const uint16_t block_pair_num = get_block_pair_num();
        const int grid_x = div_ceil<int>(size, THREAD_PER_GRID * block_pair_num);
        if (grid_x > MAX_BLOCK_X) { printf("over grid_x %d\n", grid_x); exit(1); }

        const dim3 grid(grid_x, wm_num);
        const dim3 thread(WARP_SIZE, THREAD_PER_GRID / WARP_SIZE);
        const XYIdxT size_div_w = size / WORD_SIZE;
        const XYIdxT size_div_warp = size / WARP_SIZE;
        assert(size % WARP_SIZE == 0);
        constexpr int ThreadsDimY = THREAD_PER_GRID / WARP_SIZE;


#define CALC_SRCB_SIZE(SrcT)   (0)
        constexpr int SRCB_S_8  = CALC_SRCB_SIZE(uint8_t);
        constexpr int SRCB_S_16 = CALC_SRCB_SIZE(uint16_t);
        constexpr int SRCB_S_32 = CALC_SRCB_SIZE(uint32_t);
#undef CALC_SRCB_SIZE
        {   using SrcT = uint8_t; using DstT = uint8_t;
            cudaFuncSetAttribute(&WaveletMatrix2dCu5C_UpSweep_gpu<CH_NUM, ThreadsDimY, SRC_CACHE_DIV, SRCB_S_8, SrcT, DstT, BlockT, XYIdxT, XIdxT>, cudaFuncAttributeMaxDynamicSharedMemorySize, SRCB_S_8);
        } if (!is_same<ValT, uint8_t>::value) { using SrcT = uint16_t; using DstT = uint8_t;
            cudaFuncSetAttribute(&WaveletMatrix2dCu5C_UpSweep_gpu<CH_NUM, ThreadsDimY, SRC_CACHE_DIV, SRCB_S_16, SrcT, DstT, BlockT, XYIdxT, XIdxT>, cudaFuncAttributeMaxDynamicSharedMemorySize, SRCB_S_16);
        } if (!is_same<ValT, uint8_t>::value) { using SrcT = uint16_t; using DstT = uint16_t;
            cudaFuncSetAttribute(&WaveletMatrix2dCu5C_UpSweep_gpu<CH_NUM, ThreadsDimY, SRC_CACHE_DIV, SRCB_S_16, SrcT, DstT, BlockT, XYIdxT, XIdxT>, cudaFuncAttributeMaxDynamicSharedMemorySize, SRCB_S_16);
        } if (is_same<ValT, uint32_t>::value) { using SrcT = uint32_t; using DstT = uint16_t;
            cudaFuncSetAttribute(&WaveletMatrix2dCu5C_UpSweep_gpu<CH_NUM, ThreadsDimY, SRC_CACHE_DIV, SRCB_S_32, SrcT, DstT, BlockT, XYIdxT, XIdxT>, cudaFuncAttributeMaxDynamicSharedMemorySize, SRCB_S_32);
        } if (is_same<ValT, uint32_t>::value) { using SrcT = uint32_t; using DstT = uint32_t;
            cudaFuncSetAttribute(&WaveletMatrix2dCu5C_UpSweep_gpu<CH_NUM, ThreadsDimY, SRC_CACHE_DIV, SRCB_S_32, SrcT, DstT, BlockT, XYIdxT, XIdxT>, cudaFuncAttributeMaxDynamicSharedMemorySize, SRCB_S_32);
        }

        {   using SrcT = uint8_t; using DstT = uint8_t;
            cudaFuncSetCacheConfig(&WaveletMatrix2dCu5C_UpSweep_gpu<CH_NUM, ThreadsDimY, SRC_CACHE_DIV, SRCB_S_8, SrcT, DstT, BlockT, XYIdxT, XIdxT>, cudaFuncCachePreferShared);
        } if (!is_same<ValT, uint8_t>::value) { using SrcT = uint16_t; using DstT = uint8_t;
            cudaFuncSetCacheConfig(&WaveletMatrix2dCu5C_UpSweep_gpu<CH_NUM, ThreadsDimY, SRC_CACHE_DIV, SRCB_S_16, SrcT, DstT, BlockT, XYIdxT, XIdxT>, cudaFuncCachePreferShared);
        } if (!is_same<ValT, uint8_t>::value) { using SrcT = uint16_t; using DstT = uint16_t;
            cudaFuncSetCacheConfig(&WaveletMatrix2dCu5C_UpSweep_gpu<CH_NUM, ThreadsDimY, SRC_CACHE_DIV, SRCB_S_16, SrcT, DstT, BlockT, XYIdxT, XIdxT>, cudaFuncCachePreferShared);
        } if (is_same<ValT, uint32_t>::value) { using SrcT = uint32_t; using DstT = uint16_t;
            cudaFuncSetCacheConfig(&WaveletMatrix2dCu5C_UpSweep_gpu<CH_NUM, ThreadsDimY, SRC_CACHE_DIV, SRCB_S_32, SrcT, DstT, BlockT, XYIdxT, XIdxT>, cudaFuncCachePreferShared);
        } if (is_same<ValT, uint32_t>::value) { using SrcT = uint32_t; using DstT = uint32_t;
            cudaFuncSetCacheConfig(&WaveletMatrix2dCu5C_UpSweep_gpu<CH_NUM, ThreadsDimY, SRC_CACHE_DIV, SRCB_S_32, SrcT, DstT, BlockT, XYIdxT, XIdxT>, cudaFuncCachePreferShared);
        }

        const uint32_t nsum_pos = get_nsum_pos();

        ValT *now_cu = src_cu + (CH_NUM == 1 ? 0ull : size * (size_t)CH_NUM);
        ValT *nxt_cu = now_cu + size;
        XYIdxT *nsum_buf_test = (XYIdxT*)(nxt_cu + size);
        XYIdxT *nsum_buf_test2 = nsum_buf_test + nsum_scan_buf_len;
        XIdxT *buf_idx = (XIdxT*)(nsum_buf_test2 + nsum_scan_buf_len);
        XIdxT *nxt_idx = (XIdxT*)(buf_idx + size);


        const int val_bit_len_m1 = val_bit_len - 1;
        int h = val_bit_len_m1;
        if (src_unpacked == true) {
            if (src_cu_const != nullptr) {
                printf("[Error!] not support. %s:%d\n", __FILE__, __LINE__);
                exit(-1);
            }
            WaveletMatrix2dCu5C_first_gpu_multi_srcunpacked<ThreadsDimY> <<<grid, thread, 0, main_stream >>> (ValT(mask / 2),block_pair_num, size_div_warp, now_cu, nsum_buf_test, buf_byte_div32, buf_idx, W, (XYIdxT)W * H);
        } else if (src_cu_const == nullptr) {
            WaveletMatrix2dCu5C_first_gpu_multi<CH_NUM, ThreadsDimY, 0> <<<grid_x, thread, 0, main_stream >>> (ValT(mask / 2),block_pair_num, size_div_warp, src_cu, now_cu, nsum_buf_test, buf_byte_div32, buf_idx, W, (XYIdxT)W * H, src_cu_const);
        } else {
            WaveletMatrix2dCu5C_first_gpu_multi<CH_NUM, ThreadsDimY, 1> <<<grid_x, thread, 0, main_stream >>> (ValT(mask / 2),block_pair_num, size_div_warp, src_cu, now_cu, nsum_buf_test, buf_byte_div32, buf_idx, W, (XYIdxT)W * H, src_cu_const);
        }
        BlockT * nsum_p = get_bv_block_cu(h) + nsum_pos;
        WaveletMatrix2dCu5C_ExclusiveSum<CH_NUM, MAX_BLOCK_X, XYIdxT> <<< wm_num, grid_x, 0, main_stream >>> (nsum_buf_test, nsum_buf_test2, nsum_p, buf_byte_div32, bv_block_byte_div32);

        const XYIdxT cwm_buf_byte_div32 = WM.get_buf_byte_div32();

        if constexpr (sizeof(ValT) >= 4) for (; h > 16; --h) {
            using SrcT = uint32_t;
            using DstT = uint32_t;
            mask >>= 1;
            BlockT* bv_block_nbit_cu_h = get_bv_block_cu(h);
            const int hp1 = std::min(val_bit_len - 1, h+1);
            XYIdxT* WM_nsum_buf = WM.get_nsum_buf(hp1 * CH_NUM);
            XIdxT* cwm = (h + 1 == val_bit_len ? nullptr : WM.get_src_p(hp1 * CH_NUM));
            BlockT* bv_block_nbit_pre_cu_h = get_bv_block_cu(hp1);
            WaveletMatrix2dCu5C_UpSweep_gpu<CH_NUM, ThreadsDimY, SRC_CACHE_DIV, SRCB_S_32> <<<grid, thread, SRCB_S_32, main_stream >>> ((SrcT)mask, block_pair_num, size_div_w, (SrcT*)now_cu, (DstT*)nxt_cu, bv_block_nbit_cu_h, nsum_buf_test, nsum_buf_test2, bv_block_byte_div32, buf_byte_div32, buf_idx, inf, cwm, nxt_idx, WM_nsum_buf, cwm_buf_byte_div32, bv_block_nbit_pre_cu_h);

            if (h == 0) break;
            BlockT* nsum_p = get_bv_block_cu(h - 1) + nsum_pos;
            WaveletMatrix2dCu5C_ExclusiveSum<CH_NUM, MAX_BLOCK_X, XYIdxT> <<< wm_num, grid_x, 0, main_stream >>> (nsum_buf_test2, nsum_buf_test, nsum_p, buf_byte_div32, bv_block_byte_div32);
            swap(nsum_buf_test, nsum_buf_test2);
            swap(now_cu, nxt_cu);
            swap(buf_idx, nxt_idx);
        }
        if constexpr (sizeof(ValT) >= 4) if (h == 16 || (is_same<ValT, uint32_t>::value && h >= 0)) do {
            using SrcT = uint32_t;
            using DstT = uint16_t;
            mask >>= 1;
            BlockT* bv_block_nbit_cu_h = get_bv_block_cu(h);
            const int hp1 = std::min(val_bit_len - 1, h+1);
            XYIdxT* WM_nsum_buf = WM.get_nsum_buf(hp1 * CH_NUM);
            XIdxT* cwm = (h + 1 == val_bit_len ? nullptr : WM.get_src_p(hp1 * CH_NUM));
            BlockT* bv_block_nbit_pre_cu_h = get_bv_block_cu(hp1);
            WaveletMatrix2dCu5C_UpSweep_gpu<CH_NUM, ThreadsDimY, SRC_CACHE_DIV, SRCB_S_32> <<<grid, thread, SRCB_S_32, main_stream >>> ((SrcT)mask, block_pair_num, size_div_w, (SrcT*)now_cu, (DstT*)nxt_cu, bv_block_nbit_cu_h, nsum_buf_test, nsum_buf_test2, bv_block_byte_div32, buf_byte_div32, buf_idx, inf, cwm, nxt_idx, WM_nsum_buf, cwm_buf_byte_div32, bv_block_nbit_pre_cu_h);

            if (h == 0) break;
            BlockT* nsum_p = get_bv_block_cu(h - 1) + nsum_pos;
            WaveletMatrix2dCu5C_ExclusiveSum<CH_NUM, MAX_BLOCK_X, XYIdxT> <<< wm_num, grid_x, 0, main_stream >>> (nsum_buf_test2, nsum_buf_test, nsum_p, buf_byte_div32, bv_block_byte_div32);
            swap(nsum_buf_test, nsum_buf_test2);
            swap(now_cu, nxt_cu);
            swap(buf_idx, nxt_idx);
            --h;
        } while(0);
        if constexpr (sizeof(ValT) >= 2) for (; h > 8; --h) {
            using SrcT = uint16_t;
            using DstT = uint16_t;
            mask >>= 1;
            BlockT* bv_block_nbit_cu_h = get_bv_block_cu(h);
            const int hp1 = std::min(val_bit_len - 1, h+1);
            XYIdxT* WM_nsum_buf = WM.get_nsum_buf(hp1 * CH_NUM);
            XIdxT* cwm = (h + 1 == val_bit_len ? nullptr : WM.get_src_p(hp1 * CH_NUM));
            BlockT* bv_block_nbit_pre_cu_h = get_bv_block_cu(hp1);
            WaveletMatrix2dCu5C_UpSweep_gpu<CH_NUM, ThreadsDimY, SRC_CACHE_DIV, SRCB_S_16> <<<grid, thread, SRCB_S_16, main_stream >>> ((SrcT)mask, block_pair_num, size_div_w, (SrcT*)now_cu, (DstT*)nxt_cu, bv_block_nbit_cu_h, nsum_buf_test, nsum_buf_test2, bv_block_byte_div32, buf_byte_div32, buf_idx, inf, cwm, nxt_idx, WM_nsum_buf, cwm_buf_byte_div32, bv_block_nbit_pre_cu_h);

            if (h == 0) break;
            BlockT* nsum_p = get_bv_block_cu(h - 1) + nsum_pos;
            WaveletMatrix2dCu5C_ExclusiveSum<CH_NUM, MAX_BLOCK_X, XYIdxT> <<< wm_num, grid_x, 0, main_stream >>> (nsum_buf_test2, nsum_buf_test, nsum_p, buf_byte_div32, bv_block_byte_div32);
            swap(nsum_buf_test, nsum_buf_test2);
            swap(now_cu, nxt_cu);
            swap(buf_idx, nxt_idx);
        }
        if constexpr (sizeof(ValT) >= 2) if (h == 8 || (is_same<ValT, uint32_t>::value && h >= 0)) do {
            using SrcT = uint16_t;
            using DstT = uint8_t;
            mask >>= 1;
            BlockT* bv_block_nbit_cu_h = get_bv_block_cu(h);
            const int hp1 = std::min(val_bit_len - 1, h+1);
            XYIdxT* WM_nsum_buf = WM.get_nsum_buf(hp1 * CH_NUM);
            XIdxT* cwm = (h + 1 == val_bit_len ? nullptr : WM.get_src_p(hp1 * CH_NUM));
            BlockT* bv_block_nbit_pre_cu_h = get_bv_block_cu(hp1);
            WaveletMatrix2dCu5C_UpSweep_gpu<CH_NUM, ThreadsDimY, SRC_CACHE_DIV, SRCB_S_16> <<<grid, thread, SRCB_S_16, main_stream >>> ((SrcT)mask, block_pair_num, size_div_w, (SrcT*)now_cu, (DstT*)nxt_cu, bv_block_nbit_cu_h, nsum_buf_test, nsum_buf_test2, bv_block_byte_div32, buf_byte_div32, buf_idx, inf, cwm, nxt_idx, WM_nsum_buf, cwm_buf_byte_div32, bv_block_nbit_pre_cu_h);

            if (h == 0) break;
            BlockT* nsum_p = get_bv_block_cu(h - 1) + nsum_pos;
            WaveletMatrix2dCu5C_ExclusiveSum<CH_NUM, MAX_BLOCK_X, XYIdxT> <<< wm_num, grid_x, 0, main_stream >>> (nsum_buf_test2, nsum_buf_test, nsum_p, buf_byte_div32, bv_block_byte_div32);
            swap(nsum_buf_test, nsum_buf_test2);
            swap(now_cu, nxt_cu);
            swap(buf_idx, nxt_idx);
            --h;
        } while(0);
        for (; h >= 0; --h) {
            using SrcT = uint8_t;
            using DstT = uint8_t;
            mask >>= 1;
            BlockT* bv_block_nbit_cu_h = get_bv_block_cu(h);
            const int hp1 = std::min(val_bit_len - 1, h+1);
            XYIdxT* WM_nsum_buf = WM.get_nsum_buf(hp1 * CH_NUM);
            XIdxT* cwm = (h + 1 == val_bit_len ? nullptr : WM.get_src_p(hp1 * CH_NUM));
            BlockT* bv_block_nbit_pre_cu_h = get_bv_block_cu(hp1);
            WaveletMatrix2dCu5C_UpSweep_gpu<CH_NUM, ThreadsDimY, SRC_CACHE_DIV, SRCB_S_8> <<<grid, thread, SRCB_S_8, main_stream >>> ((SrcT)mask, block_pair_num, size_div_w, (SrcT*)now_cu, (DstT*)nxt_cu, bv_block_nbit_cu_h, nsum_buf_test, nsum_buf_test2, bv_block_byte_div32, buf_byte_div32, buf_idx, inf, cwm, nxt_idx, WM_nsum_buf, cwm_buf_byte_div32, bv_block_nbit_pre_cu_h);

            if (h == 0) break;
            BlockT* nsum_p = get_bv_block_cu(h - 1) + nsum_pos;
            WaveletMatrix2dCu5C_ExclusiveSum<CH_NUM, MAX_BLOCK_X, XYIdxT> <<< wm_num, grid_x, 0, main_stream >>> (nsum_buf_test2, nsum_buf_test, nsum_p, buf_byte_div32, bv_block_byte_div32);
            swap(nsum_buf_test, nsum_buf_test2);
            swap(now_cu, nxt_cu);
            swap(buf_idx, nxt_idx);
        }
        {
            const int h = 0;
            XYIdxT* WM_nsum_buf = WM.get_nsum_buf(h * CH_NUM);
            XIdxT* cwm = WM.get_src_p(h * CH_NUM);
            BlockT* bv_block_nbit_pre_cu_h = get_bv_block_cu(h);

            WaveletMatrix2dCu5C_last_gpu<CH_NUM, ThreadsDimY, BlockT> <<<grid, thread, SRCB_S_8, main_stream >>> (block_pair_num, size_div_w, buf_byte_div32, nxt_idx, inf, cwm, WM_nsum_buf, cwm_buf_byte_div32, bv_block_nbit_pre_cu_h, bv_block_byte_div32);
        }
        WM.construct(main_stream, false);
    }

    XYIdxT get_nsum_pos() const {
        const XYIdxT size_div_w = size / WORD_SIZE;
        return size_div_w;
    }

    template<int TH_W = 8, int TH_H = 32, typename ResTableT = std::nullptr_t, bool CUT_BORDER = false>
    void median2d(const int r, const ResTableT* res_table = nullptr) {
        median2d<TH_W, TH_H, ResTableT, CUT_BORDER>(r, -1, res_table);
    }

    template<int TH_W = 8, int TH_H = 32, typename ResTableT = std::nullptr_t, bool CUT_BORDER = false>
    void median2d(const int r, int res_step_num = -1, const ResTableT* res_table = nullptr, const cudaStream_t main_stream = 0) {
        if (bv_block_nbit_and_nsum_base_cu == nullptr) { printf("Median2d Error: memory not alloced."); return;}
        if (is_same<ResTableT, std::nullptr_t>::value == false && res_table == nullptr) {printf("Median2d Error: res_table is null."); return;}
        static_assert(is_same<ResTableT, std::nullptr_t>::value || (sizeof(ResTableT) <= sizeof(ValT)), "");

        static_assert(TH_W * TH_H <= 1024, "max number of threads in block");

        if (res_step_num < 0) res_step_num = W * CH_NUM;

        constexpr int THREADS_NUM_W = TH_W;
        const dim3 thread(THREADS_NUM_W, TH_H);
        const dim3 grid(div_ceil<size_t>((CUT_BORDER ? W - 2 * r: W), THREADS_NUM_W), div_ceil<size_t>((CUT_BORDER ? H - 2 * r : H), TH_H), CH_NUM);


        const uint32_t bv_nsum_pos = get_nsum_pos();
        const BlockT*  bv_bv_block_nbit_cu_first = get_bv_block_cu(val_bit_len - 1);

        const BlockT* wm_bv_block_nbit_cu_first = WM.get_bv_block_cu(w_bit_len - 1, (val_bit_len - 1) * CH_NUM); //
        const uint32_t nsum_pos = WM.get_nsum_pos();
        const uint64_t wm_bv_block_byte = WM.get_bv_block_byte();

        if (bv_nsum_pos != nsum_pos) { printf("err! line %d", __LINE__); exit(-1); }
        if (WM.get_bv_block_byte()  != WM.get_bv_block_h_byte_div32() * 32ull * w_bit_len) { printf("err! line %d", __LINE__); exit(-1); }

        if (bv_block_len != WM.bv_block_len) {printf("bv_block_len error!\n"); exit(1);}

        using ResT = typename std::conditional<is_same<ResTableT, std::nullptr_t>::value, ValT, ResTableT>::type;

        const int Wc = (CUT_BORDER ? W - 2 * r : W);
        const int Hc = (CUT_BORDER ? H - 2 * r : H);

        constexpr int VAL_BIT_LEN = (sizeof(ValT) < 4) ? MAX_BIT_LEN : -1;
        WaveletMatrix2dCu5C_median2d_cu<CH_NUM, THREADS_NUM_W, TH_H, VAL_BIT_LEN, CUT_BORDER, XYIdxT, XIdxT, ValT> <<<grid, thread, 0, main_stream>>>
            (Hc, Wc, res_step_num, r, (ResT*)res_cu, wm_bv_block_nbit_cu_first, nsum_pos, WM.get_bv_block_h_byte_div32(), bv_block_len,
                bv_bv_block_nbit_cu_first, w_bit_len, val_bit_len, res_table);
    }

    template<typename ResTableT = ValT>
    vector<vector<ResTableT>> get_res() {
        static_assert(sizeof(ResTableT) <= sizeof(ValT), "");
        auto res = vector<vector<ResTableT>>(H, vector<ResTableT>(W));
        if (res_cu == nullptr) { printf("get_res Error: memory not alloced."); return res;}

        for (int i = 0; i < H; ++i) {
            cudaMemcpy(res[i].data(), res_cu + (XYIdxT)W * i, W * sizeof(ResTableT), cudaMemcpyDeviceToHost);
        }
        return res;
    }
};

} // end namespace wavelet_median
}}} //end namespace cv::cuda::device

#endif
#endif // __OPENCV_WAVELET_MATRIX_2D_CUH__
