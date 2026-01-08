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

#ifndef __OPENCV_WAVELET_MATRIX_MULTI_CUH__
#define __OPENCV_WAVELET_MATRIX_MULTI_CUH__

// The CUB library is used for the Median Filter with Wavelet Matrix,
// which has become a standard library since CUDA 11.
#include "wavelet_matrix_feature_support_checks.h"
#ifdef __OPENCV_USE_WAVELET_MATRIX_FOR_MEDIAN_FILTER_CUDA__


#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cub/cub.cuh>
#include <cub/device/device_scan.cuh>
#include <cub/block/block_scan.cuh>
#include "opencv2/core/cuda/warp_shuffle.hpp"


#include <vector>

namespace cv { namespace cuda { namespace device
{

namespace wavelet_matrix_median {
    using std::vector;
    using namespace std;

    template <typename T>
    constexpr T div_ceil(T a, T b) {
        return (a + b - 1) / b;
    }


template<int ThreadsDimY, int SRCB_S, int DSTB_S, typename SrcT, typename DstT, class BlockT, typename IdxType>
__global__ void WaveletMatrixMultiCu4G_UpSweep_gpu(const SrcT mask, const uint16_t block_pair_num, const IdxType size_div_w, const SrcT* __restrict__ src, DstT* __restrict__ dst, BlockT* __restrict__ nbit_bp, const IdxType* __restrict__ nsum_zeros_buf, IdxType* __restrict__ nsum_zeros_buf2, const uint32_t bv_block_byte_div32, const uint32_t buf_byte_div32) {
    using WordT = decltype(BlockT::nbit);
    using WarpWT = uint32_t;
    constexpr int WARP_SIZE = 8 * sizeof(WarpWT);
    static_assert(WARP_SIZE == 32, "");
    static constexpr int THREAD_PER_GRID = ThreadsDimY * WARP_SIZE;
    constexpr int WORD_SIZE = 8 * sizeof(WordT);
    static_assert(WORD_SIZE == 32 || WORD_SIZE == 64, "");
    constexpr uint32_t WORD_DIV_WARP = WORD_SIZE / WARP_SIZE;

    src            = (SrcT*)((uint8_t*)src + (size_t)blockIdx.y * (buf_byte_div32*32ull));
    dst            = (DstT*)((uint8_t*)dst + (size_t)blockIdx.y * (buf_byte_div32*32ull));
    nsum_zeros_buf  = (IdxType*)((uint8_t*)nsum_zeros_buf + (size_t)blockIdx.y * (buf_byte_div32*32ull));
    nsum_zeros_buf2 = (IdxType*)((uint8_t*)nsum_zeros_buf2 + (size_t)blockIdx.y * (buf_byte_div32*32ull));
    nbit_bp        = (BlockT*)((uint8_t*)nbit_bp + (size_t)blockIdx.y * (bv_block_byte_div32*32ull)); // TODO: rename

    using WarpScan = cub::WarpScan<IdxType>;
    using WarpScanY = cub::WarpScan<IdxType, ThreadsDimY>;
    using WarpReduce = cub::WarpReduce<uint32_t>;
    using WarpReduceY = cub::WarpReduce<uint32_t, ThreadsDimY>;

    constexpr size_t shmem_size = sizeof(SrcT) * (ThreadsDimY * (WARP_SIZE - 1) * WARP_SIZE);
    static_assert(SRCB_S == shmem_size, "");
    static_assert(SRCB_S + DSTB_S < 64 * 1024, "");

    constexpr int DST_BUF_SIZE = DSTB_S;
    constexpr int DST_BUF_NUM_PER_WARP = DST_BUF_SIZE / (ThreadsDimY * sizeof(DstT)); // [32k/32/2=512] [48k/8/1=6114]
    constexpr int DST_BUF_NUM_PER_THREAD = DST_BUF_NUM_PER_WARP / WARP_SIZE;
    static_assert(DST_BUF_NUM_PER_THREAD <= WARP_SIZE, "");


    extern __shared__ uint8_t shmem_base[];
    SrcT* __restrict__ src_val_cache = (SrcT*)shmem_base;
    DstT* __restrict__ dst_buf = (DstT*)&src_val_cache[SRCB_S] + threadIdx.y * DST_BUF_NUM_PER_WARP;  //[ThreadsDimY][DST_BUF_NUM_PER_WARP];

    __shared__ uint4 nsum_count_sh[ThreadsDimY];
    __shared__ IdxType pre_sum_share[2];
    __shared__ IdxType warp_scan_sums[WARP_SIZE];
    __shared__ typename WarpScan::TempStorage s_scanStorage;
    __shared__ typename WarpScanY::TempStorage s_scanStorage2;
    __shared__ typename WarpReduce::TempStorage WarpReduce_temp_storage[ThreadsDimY];
    __shared__ typename WarpReduceY::TempStorage WarpReduceY_temp_storage;
    // shmem ------ end ------

    const IdxType size_div_warp = size_div_w * WORD_DIV_WARP;
    const IdxType nsum = nbit_bp[size_div_w].nsum;
    const IdxType nsum_offset = nsum_zeros_buf[blockIdx.x];


    IdxType nsum_idx0_org = nsum_offset;
    IdxType nsum_idx1_org = (IdxType)blockIdx.x * block_pair_num * THREAD_PER_GRID + nsum - nsum_idx0_org;
    nsum_idx0_org /= (IdxType)block_pair_num * ThreadsDimY * WARP_SIZE;
    nsum_idx1_org /= (IdxType)block_pair_num * ThreadsDimY * WARP_SIZE;
    const IdxType nsum_idx0_bound = (nsum_idx0_org + 1) * block_pair_num * ThreadsDimY * WARP_SIZE;
    const IdxType nsum_idx1_bound = (nsum_idx1_org + 1) * block_pair_num * ThreadsDimY * WARP_SIZE;
    uint4 nsum_count = make_uint4(0, 0, 0, 0);

    const unsigned short th_idx = threadIdx.y * WARP_SIZE + threadIdx.x;
    if (th_idx == 0) {
        pre_sum_share[0] = nsum_offset;
    }


    for (IdxType ka = 0; ka < block_pair_num; ka += WARP_SIZE) {
        const IdxType ibb = ((IdxType)blockIdx.x * block_pair_num + ka) * ThreadsDimY;
        if (ibb >= size_div_warp) break;
        WarpWT my_bits = 0;
        SrcT first_val;

        const IdxType src_val_cache_offset = IdxType(threadIdx.y * (WARP_SIZE - 1) - 1) * WARP_SIZE + threadIdx.x;
        for (IdxType kb = 0, i = ibb + WARP_SIZE * threadIdx.y; kb < WARP_SIZE; ++kb, ++i) {
            if (i >= size_div_warp) break;
            WarpWT bits;
            const SrcT v = src[i * WARP_SIZE + threadIdx.x];
            if (kb == 0) {
                first_val = v;
            } else {
                src_val_cache[src_val_cache_offset + kb * WARP_SIZE] = v;
            }
            if (v <= mask) {
                bits = __activemask();
            } else {
                bits = ~__activemask();
            }
            if (threadIdx.x == kb) {
                my_bits = bits;
            }
        }
        IdxType t, c = __popc(my_bits);
        WarpScan(s_scanStorage).ExclusiveSum(c, t);

        if (threadIdx.x == WARP_SIZE - 1) {
            warp_scan_sums[threadIdx.y] = c + t;
        }
        __syncthreads();
        IdxType pre_sum = pre_sum_share[(ka & WARP_SIZE) > 0];
        IdxType s = threadIdx.x < ThreadsDimY ? warp_scan_sums[threadIdx.x] : 0;
        WarpScanY(s_scanStorage2).ExclusiveSum(s, s);

        s = cv::cuda::device::shfl(s, threadIdx.y, WARP_SIZE);

        s += t + pre_sum;
        if (th_idx == THREAD_PER_GRID - 1) {
            pre_sum_share[(ka & WARP_SIZE) == 0] = s + c;
        }
        const IdxType bi = ibb + th_idx;

        if (bi < size_div_warp) {
            static_assert(WORD_SIZE == 32, "");
            nbit_bp[bi] = BlockT{s, my_bits};
        }
        if (mask == 0) continue;

        const SrcT mask_2 = mask >> 1;
        SrcT vo = first_val;
        for (IdxType j = 0, i = ibb + WARP_SIZE * threadIdx.y; j < WARP_SIZE;) {
            if (i >= size_div_warp) break;


            IdxType idx0_begin, idx0_num, idx1_offset, idx01_num, idx1_offset0;
            if (DST_BUF_SIZE > 0) { // constexpr
                IdxType idx1_begin, idx0_end;
                const IdxType ib = ::min(size_div_warp, i + DST_BUF_NUM_PER_THREAD);
                const IdxType jb = j + ib - i - 1;
                idx0_begin = cv::cuda::device::shfl(s, j, WARP_SIZE);
                idx1_begin = i * WARP_SIZE + nsum - idx0_begin;
                idx0_end = cv::cuda::device::shfl(s + c, jb, WARP_SIZE);

                idx0_num = idx0_end - idx0_begin;
                idx1_offset = idx1_begin - idx0_num;
                idx01_num = (ib - i) * WARP_SIZE;
                idx1_offset0 = nsum - idx1_begin + idx0_num;
            }
            constexpr int DST_LOOP_NUM = (DST_BUF_SIZE == 0 ? 1 : DST_BUF_NUM_PER_THREAD);
            for (IdxType kb = 0; kb < DST_LOOP_NUM; ++kb, ++j, ++i) {
                if (i >= size_div_warp) break;

                const WarpWT  e_nbit = cv::cuda::device::shfl(my_bits, j, WARP_SIZE);
                const IdxType e_nsum = cv::cuda::device::shfl(s, j, WARP_SIZE);
                IdxType rank = __popc(e_nbit << (WARP_SIZE - threadIdx.x));
                const IdxType idx0 = e_nsum + rank;

                DstT v = (DstT)vo;
                IdxType idx;
                IdxType buf_idx;
                if (vo > mask) { // 1
                    const IdxType ij = i * WARP_SIZE + threadIdx.x;
                    idx = ij + nsum - idx0;
                    v &= mask;
                    buf_idx = ij - idx0 + idx1_offset0;
                } else {
                    idx = idx0;
                    buf_idx = idx0 - idx0_begin;
                }
                if (DST_BUF_SIZE == 0) {
                    dst[idx] = v;
                } else {
                    dst_buf[buf_idx] = (DstT)v;
                }

                if (v <= mask_2) {
                    if (vo <= mask) {
                        if (idx < nsum_idx0_bound) {
                            nsum_count.x++;
                        } else {
                            assert(idx < nsum_idx0_bound + block_pair_num * ThreadsDimY * WARP_SIZE);
                            nsum_count.y++;
                        }
                    } else {
                        if (idx < nsum_idx1_bound) {
                            nsum_count.z++;
                        } else {
                            assert(idx < nsum_idx1_bound + block_pair_num * ThreadsDimY * WARP_SIZE);
                            nsum_count.w++;
                        }
                    }
                }
                if (j == WARP_SIZE - 1) { j = WARP_SIZE; break; }
                vo = src_val_cache[(threadIdx.y * (WARP_SIZE - 1) + j) * WARP_SIZE + threadIdx.x];
            }
            if (DST_BUF_SIZE > 0) { // constexpr
                for (IdxType j = threadIdx.x; (int)j < DST_BUF_NUM_PER_WARP; j += WARP_SIZE) {
                    if (j >= idx01_num) break;
                    IdxType idx;
                    if (j < idx0_num) { // 0
                        idx = j + idx0_begin;
                    } else {    // 1
                        idx = j + idx1_offset;
                    }
                    dst[idx] = dst_buf[j];
                }
            }
        }
    }

    if (blockIdx.x == gridDim.x - 1 && th_idx == 0) {
        nbit_bp[size_div_warp / WORD_DIV_WARP].nsum = nsum;
    }
    if (mask == 0) return;

    nsum_count.x = WarpReduce(WarpReduce_temp_storage[threadIdx.y]).Sum(nsum_count.x);
    nsum_count.y = WarpReduce(WarpReduce_temp_storage[threadIdx.y]).Sum(nsum_count.y);
    nsum_count.z = WarpReduce(WarpReduce_temp_storage[threadIdx.y]).Sum(nsum_count.z);
    nsum_count.w = WarpReduce(WarpReduce_temp_storage[threadIdx.y]).Sum(nsum_count.w);
    if (threadIdx.x == 0) {
        nsum_count_sh[threadIdx.y] = nsum_count;
    }
    __syncthreads();


    if (threadIdx.x < ThreadsDimY) {
        nsum_count = nsum_count_sh[threadIdx.x];
        nsum_count.x = WarpReduceY(WarpReduceY_temp_storage).Sum(nsum_count.x);
        nsum_count.y = WarpReduceY(WarpReduceY_temp_storage).Sum(nsum_count.y);
        nsum_count.z = WarpReduceY(WarpReduceY_temp_storage).Sum(nsum_count.z);
        nsum_count.w = WarpReduceY(WarpReduceY_temp_storage).Sum(nsum_count.w);
        if (th_idx == 0) {
            const IdxType nsum_idx0_org = nsum_idx0_bound / ((IdxType)block_pair_num * ThreadsDimY * WARP_SIZE);
            const IdxType nsum_idx1_org = nsum_idx1_bound / ((IdxType)block_pair_num * ThreadsDimY * WARP_SIZE);
            if (nsum_count.x > 0) atomicAdd(nsum_zeros_buf2 + nsum_idx0_org - 1, nsum_count.x);
            if (nsum_count.y > 0) atomicAdd(nsum_zeros_buf2 + nsum_idx0_org - 0, nsum_count.y);
            if (nsum_count.z > 0) atomicAdd(nsum_zeros_buf2 + nsum_idx1_org - 1, nsum_count.z);
            if (nsum_count.w > 0) atomicAdd(nsum_zeros_buf2 + nsum_idx1_org - 0, nsum_count.w);
        }
    }
}


template<int MAX_BLOCK_X, typename IdxType, typename BlockT>
__global__ void WaveletMatrixMultiCu4G_ExclusiveSum(IdxType* __restrict__ nsum_scan_buf, IdxType* __restrict__ nsum_zeros_buf2, BlockT* __restrict__ nsum_p, const uint32_t buf_byte_div32, const uint32_t bv_block_byte_div32) {

    typedef cub::BlockScan<IdxType, MAX_BLOCK_X> BlockScan;
    __shared__ typename BlockScan::TempStorage temp_storage;

    IdxType thread_data1;
    IdxType thread_data2;
    nsum_scan_buf  = (IdxType*)((uint8_t*)nsum_scan_buf + (size_t)blockIdx.x * (buf_byte_div32*32ull));
    nsum_zeros_buf2 = (IdxType*)((uint8_t*)nsum_zeros_buf2 + (size_t)blockIdx.x * (buf_byte_div32*32ull));

    thread_data1 = nsum_scan_buf[threadIdx.x];
    BlockScan(temp_storage).ExclusiveSum(thread_data1, thread_data2);

    nsum_scan_buf[threadIdx.x] = thread_data2;
    nsum_zeros_buf2[threadIdx.x] = 0;

    if (threadIdx.x == blockDim.x - 1) {
        thread_data2 += thread_data1;
        nsum_p = (BlockT*)((uint8_t*)nsum_p + (size_t)blockIdx.x * (bv_block_byte_div32*32ull));
        nsum_p->nsum = thread_data2;
    }
}


template<int ThreadsDimY, typename SrcT, typename IdxType>
__global__ void WaveletMatrixMultiCu4G_first_gpu(const SrcT mask, uint16_t block_pair_num, const IdxType size_div_warp, const SrcT* __restrict__ src, IdxType* __restrict__ nsum_scan_buf, const uint32_t buf_byte_div32) {
    using WarpWT = uint32_t;
    constexpr int WARP_SIZE = 8 * sizeof(WarpWT);
    static_assert(WARP_SIZE == 32, "");
    static constexpr int THREAD_PER_GRID = ThreadsDimY * WARP_SIZE;

    src = (SrcT*)((uint8_t*)src + (size_t)blockIdx.y * (buf_byte_div32*32ull));
    IdxType cs = 0;
    IdxType ibb = (IdxType)blockIdx.x * block_pair_num * ThreadsDimY;
    for (IdxType ka = 0; ka < block_pair_num; ka += WARP_SIZE, ibb += THREAD_PER_GRID) {
        for (IdxType kb = 0, i = ibb + WARP_SIZE * threadIdx.y; kb < WARP_SIZE; ++kb, ++i) {
            if (i >= size_div_warp) break;
            const SrcT v = src[i * WARP_SIZE + threadIdx.x];
            if (v <= mask) {
                ++cs;
            }
        }
    }
    using BlockReduce = cub::BlockReduce<IdxType, WARP_SIZE, cub::BLOCK_REDUCE_WARP_REDUCTIONS, ThreadsDimY>;
    __shared__ typename BlockReduce::TempStorage s_reduceStorage;
    IdxType reducedValue = BlockReduce(s_reduceStorage).Sum(cs);

    if (threadIdx.y == 0 && threadIdx.x == 0) {
        nsum_scan_buf = (IdxType*)((uint8_t*)nsum_scan_buf + (size_t)blockIdx.y * (buf_byte_div32*32ull));
        nsum_scan_buf[blockIdx.x] = reducedValue;
    }
}



template<int MIN_DSTBUF_KB>
constexpr int WaveletMatrixMultiCu4G_get_dstbuf_kb_internal_Z(uint32_t r) {
    return (r >= MIN_DSTBUF_KB) ? r * 1024 : 0;
}

template<int MIN_DSTBUF_KB>
constexpr int WaveletMatrixMultiCu4G_get_dstbuf_kb_internal_B(uint32_t r) {
    return WaveletMatrixMultiCu4G_get_dstbuf_kb_internal_Z<MIN_DSTBUF_KB>((r + 1) / 2);
}
template<int MIN_DSTBUF_KB>
constexpr int WaveletMatrixMultiCu4G_get_dstbuf_kb_internal_A16(uint32_t r) {
    return WaveletMatrixMultiCu4G_get_dstbuf_kb_internal_B<MIN_DSTBUF_KB>(r | (r >> 16));
}
template<int MIN_DSTBUF_KB>
constexpr int WaveletMatrixMultiCu4G_get_dstbuf_kb_internal_A8(uint32_t r) {
    return WaveletMatrixMultiCu4G_get_dstbuf_kb_internal_A16<MIN_DSTBUF_KB>(r | (r >> 8));
}
template<int MIN_DSTBUF_KB>
constexpr int WaveletMatrixMultiCu4G_get_dstbuf_kb_internal_A4(uint32_t r) {
    return WaveletMatrixMultiCu4G_get_dstbuf_kb_internal_A8<MIN_DSTBUF_KB>(r | (r >> 4));
}
template<int MIN_DSTBUF_KB>
constexpr int WaveletMatrixMultiCu4G_get_dstbuf_kb_internal_A2(uint32_t r) {
    return WaveletMatrixMultiCu4G_get_dstbuf_kb_internal_A4<MIN_DSTBUF_KB>(r | (r >> 2));
}
template<int MIN_DSTBUF_KB>
constexpr int WaveletMatrixMultiCu4G_get_dstbuf_kb_internal_A1(uint32_t r) {
    return WaveletMatrixMultiCu4G_get_dstbuf_kb_internal_A2<MIN_DSTBUF_KB>(r | (r >> 1));
}
template<int SHMEM_USE_KB, int MIN_DSTBUF_KB>
constexpr int WaveletMatrixMultiCu4G_get_dstbuf_kb(int SRCB_S) {
    return WaveletMatrixMultiCu4G_get_dstbuf_kb_internal_A1<MIN_DSTBUF_KB>(SHMEM_USE_KB - (SRCB_S + 1023) / 1024);
}

// template<int SHMEM_USE_KB, int MIN_DSTBUF_KB>
// constexpr int WaveletMatrixMultiCu4G_get_dstbuf_kb(int SRCB_S) {
//     uint32_t r = (SHMEM_USE_KB - (SRCB_S + 1023) / 1024);
//     r |= r >> 1;
//     r |= r >> 2;
//     r |= r >> 4;
//     r |= r >> 8;
//     r |= r >> 16;
//     r = (r + 1) / 2;
//     return (r >= MIN_DSTBUF_KB) ? r * 1024 : 0;
// }


template <typename T = uint64_t, int TH_NUM = 512, int WORD_SIZE = 32, typename IdxType = uint32_t>
struct WaveletMatrixMultiCu4G {
    static constexpr int MAX_BIT_LEN = 8 * sizeof(T);

    static constexpr uint32_t WSIZE = WORD_SIZE;
    using T_Type = T;
    static constexpr int WARP_SIZE       = 32;
    static constexpr int THREAD_PER_GRID = TH_NUM;
    static constexpr int MAX_BLOCK_X     = 1024;
    static_assert(WORD_SIZE == 32 || WORD_SIZE == 64, "WORD SIZE must be 32 or 64");
    using WordT = typename std::conditional<WORD_SIZE == 32, uint32_t, uint64_t>::type;

    static constexpr int SHMEM_USE_KB  = 64;

    struct __align__(8) BLOCK32_T { uint32_t nsum; union { uint32_t nbit; uint32_t nbit_a[1];}; };
    struct __align__(4) BLOCK64_T { uint32_t nsum; union { uint64_t nbit; uint32_t nbit_a[2];}; };
    using BlockT = typename std::conditional<WORD_SIZE == 32, BLOCK32_T, BLOCK64_T>::type;

    static constexpr int MIN_DSTBUF_KB =  4;
    static constexpr int BLOCK_TYPE = 2;
    using WarpWT = uint32_t;
    static_assert(8 * sizeof(WarpWT) == WARP_SIZE, "bits of WarpWT must be WARP_SIZE");

    IdxType size = 0;
    int wm_num = 0;

private:
    T* src_cu = nullptr;
    uint8_t* bv_block_nbit_and_nsum_base_cu = nullptr;
    uint32_t bv_block_byte_div32;
    uint32_t buf_byte_div32;
public:
    size_t bv_block_len = 0;
    IdxType bv_zeros[MAX_BIT_LEN];
    int bit_len = 0;

    WaveletMatrixMultiCu4G(IdxType _n = 0, int _bit_len = 0, int num = 0) {
        reset(_n, _bit_len, num);
    }
    void reset(IdxType _n, int _bit_len, int _num) {
        cudaError_t err;
        assert(size == 0 && src_cu == nullptr && 0 <= _bit_len && _bit_len <= MAX_BIT_LEN);
        bit_len = _bit_len;
        wm_num = _num;
        if (_n == 0 || wm_num == 0) return;
        size = div_ceil<size_t>(_n, WORD_SIZE) * WORD_SIZE;
        bv_block_len = div_ceil<size_t>(size, THREAD_PER_GRID) * THREAD_PER_GRID / WORD_SIZE + 1;
        bv_block_len = div_ceil<size_t>(bv_block_len, 8*2) * 8*2;

        const size_t bv_block_byte = (sizeof(BlockT)) * bit_len * bv_block_len;
        if (bv_block_byte % 32 != 0) { printf("bv_block_byte not 32n!"); exit(-1); }
        bv_block_byte_div32 = div_ceil<size_t>(bv_block_byte, 32);

        err = cudaMalloc(&bv_block_nbit_and_nsum_base_cu, (size_t)(bv_block_byte_div32*32ull) * _num);
        if (bv_block_nbit_and_nsum_base_cu == nullptr) { printf("GPU Memory Alloc Error! %s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); release(); return; }

        const uint16_t block_pair_num = get_block_pair_num();
        const IdxType nsum_scan_buf_len = div_ceil<size_t>(size, (size_t)THREAD_PER_GRID * block_pair_num);

        const size_t buf_byte = sizeof(IdxType) * 2 * nsum_scan_buf_len + sizeof(T) * size * 2;
        buf_byte_div32 = div_ceil<size_t>(buf_byte, 32);
        err = cudaMalloc(&src_cu, (size_t)(buf_byte_div32*32ull) * _num);
        if (src_cu == nullptr) { printf("GPU Memory Alloc Error! %s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); release(); return; }

    }
    void release() {
        size = 0;
        if (src_cu != nullptr) cudaFree(src_cu);
        if (bv_block_nbit_and_nsum_base_cu != nullptr) cudaFree(bv_block_nbit_and_nsum_base_cu);
        src_cu = nullptr;
        bv_block_nbit_and_nsum_base_cu = nullptr;
    }
    ~WaveletMatrixMultiCu4G() { release(); }

    BlockT*  get_bv_block_cu(int h) const { return (BlockT*)(bv_block_nbit_and_nsum_base_cu + (sizeof(BlockT)) * bv_block_len * h); }
    BlockT*  get_bv_block_cu(int h, int c) const { return (BlockT*)((uint8_t*)get_bv_block_cu(h) + (size_t)c * (bv_block_byte_div32*32ull)); }
    uint64_t get_bv_block_byte() const { return (bv_block_byte_div32*32ull); }


    T* get_src_p(int c) const { return src_cu + (size_t)(buf_byte_div32*32ull) / (sizeof(T)) * c; }

    uint16_t get_block_pair_num() const {
        constexpr int x_chunk = 65536 / THREAD_PER_GRID / WARP_SIZE; // To make the pixels assigned per grid a multiple of 65536.
        static_assert(x_chunk > 0, "");
        const uint64_t total_gridx = div_ceil<uint64_t>(size, THREAD_PER_GRID * WARP_SIZE);
        uint64_t block_pair_num_org = div_ceil<uint64_t>(total_gridx, MAX_BLOCK_X);
        if (block_pair_num_org <= x_chunk) {
            block_pair_num_org--;
            block_pair_num_org |= block_pair_num_org >> 1;
            block_pair_num_org |= block_pair_num_org >> 2;
            block_pair_num_org |= block_pair_num_org >> 4;
            block_pair_num_org |= block_pair_num_org >> 8;
            block_pair_num_org++;
        } else {
            block_pair_num_org = div_ceil<uint64_t>(block_pair_num_org, x_chunk) * x_chunk;
        }
        block_pair_num_org *= WARP_SIZE;

        if (block_pair_num_org >= (1LL << (8 * sizeof(uint16_t)))) { printf("over block_pair_num %ld\n", block_pair_num_org); exit(1); }
        return (uint16_t)block_pair_num_org;
    }
    std::pair<IdxType*, size_t> get_nsum_buf_and_buf_byte() const {
        IdxType* nsum_buf = (IdxType*)(src_cu + 2ull * size);
        return { nsum_buf, (buf_byte_div32*32ull) };
    }
    IdxType* get_nsum_buf(int c) const {
        IdxType* nsum_buf = (IdxType*)(src_cu + 2ull * size);
        return (IdxType*)((uint8_t*)nsum_buf + (size_t)(buf_byte_div32*32ull) * c);
    }
    uint64_t get_buf_byte() const { return (buf_byte_div32*32ull); }
    uint32_t get_buf_byte_div32() const { return buf_byte_div32; }

    IdxType get_nsum_scan_buf_len(uint16_t block_pair_num) const {
        return div_ceil<IdxType>(size, THREAD_PER_GRID * block_pair_num);
    }
    IdxType get_nsum_scan_buf_len() const {
        const uint16_t block_pair_num = get_block_pair_num();
        return get_nsum_scan_buf_len(block_pair_num);
    }

    // Set data in src_cu before calling (data will be destroyed).
    void construct(const cudaStream_t main_stream = 0, const bool run_first = true) {
        assert(size > 0 && src_cu != nullptr);
        if (size == 0 || wm_num == 0) return;
        if (src_cu == nullptr) { printf("Build Error: memory not alloced."); return;}

        T mask = ((T)1 << bit_len) - 1;

        const uint16_t block_pair_num = get_block_pair_num();
        const int grid_x = div_ceil<int>(size, THREAD_PER_GRID * block_pair_num);
        if (grid_x > MAX_BLOCK_X) { printf("over grid_x %d\n", grid_x); exit(1); }


        const dim3 grid(grid_x, wm_num);
        const dim3 thread(WARP_SIZE, THREAD_PER_GRID / WARP_SIZE);
        const IdxType size_div_w = size / WORD_SIZE;
        const IdxType size_div_warp = size / WARP_SIZE;
        assert(size % WARP_SIZE == 0);
        constexpr int ThreadsDimY = THREAD_PER_GRID / WARP_SIZE;


        const int nsum_scan_buf_len = get_nsum_scan_buf_len(block_pair_num); // same grid_x


#define CALC_SRCB_SIZE(SrcT)   (sizeof(SrcT) * (ThreadsDimY * (WARP_SIZE - 1) * WARP_SIZE))
        constexpr int SRCB_S_T = CALC_SRCB_SIZE(T);
        constexpr int SRCB_S_8 = CALC_SRCB_SIZE(uint8_t);
#undef CALC_SRCB_SIZE
        constexpr int BLOCK_SHMEM_KB = SHMEM_USE_KB * THREAD_PER_GRID / 1024;
        constexpr int DSTB_S_T = WaveletMatrixMultiCu4G_get_dstbuf_kb<BLOCK_SHMEM_KB, MIN_DSTBUF_KB>(SRCB_S_T);
        constexpr int DSTB_S_8 = WaveletMatrixMultiCu4G_get_dstbuf_kb<BLOCK_SHMEM_KB, MIN_DSTBUF_KB>(SRCB_S_8);
        static_assert(SHMEM_USE_KB >= 64 || THREAD_PER_GRID == 1024, "if SHMEM_USE_KB < 64, THREAD_PER_GRID must 1024");
        static_assert(SRCB_S_T + DSTB_S_T <= BLOCK_SHMEM_KB * 1024 && ((DSTB_S_T == 0 && SRCB_S_T + MIN_DSTBUF_KB * 1024> BLOCK_SHMEM_KB * 1024) || (DSTB_S_T >= MIN_DSTBUF_KB * 1024 && SRCB_S_T + DSTB_S_T * 2 > BLOCK_SHMEM_KB * 1024)), "");
        static_assert(SRCB_S_8 + DSTB_S_8 <= BLOCK_SHMEM_KB * 1024 && ((DSTB_S_8 == 0 && SRCB_S_8 + MIN_DSTBUF_KB * 1024> BLOCK_SHMEM_KB * 1024) || (DSTB_S_8 >= MIN_DSTBUF_KB * 1024 && SRCB_S_8 + DSTB_S_8 * 2 > BLOCK_SHMEM_KB * 1024)), "");

        {   using SrcT = T; using DstT = T;
            cudaFuncSetAttribute(&WaveletMatrixMultiCu4G_UpSweep_gpu<ThreadsDimY, SRCB_S_T, DSTB_S_T, SrcT, DstT, BlockT, IdxType>, cudaFuncAttributeMaxDynamicSharedMemorySize, SRCB_S_T + DSTB_S_T);
        } { using SrcT = T; using DstT = uint8_t;
            cudaFuncSetAttribute(&WaveletMatrixMultiCu4G_UpSweep_gpu<ThreadsDimY, SRCB_S_T, DSTB_S_T, SrcT, DstT, BlockT, IdxType>, cudaFuncAttributeMaxDynamicSharedMemorySize, SRCB_S_T + DSTB_S_T);
        } { using SrcT = uint8_t; using DstT = uint8_t;
            cudaFuncSetAttribute(&WaveletMatrixMultiCu4G_UpSweep_gpu<ThreadsDimY, SRCB_S_8, DSTB_S_8, SrcT, DstT, BlockT, IdxType>, cudaFuncAttributeMaxDynamicSharedMemorySize, SRCB_S_8 + DSTB_S_8);
        }

        T* now_cu = src_cu;
        T* nxt_cu = src_cu + size;
        IdxType* nsum_zeros_buf = (IdxType*)(nxt_cu + size);
        IdxType* nsum_zeros_buf2 = nsum_zeros_buf + nsum_scan_buf_len;

        const uint32_t nsum_pos = get_nsum_pos();

        int h = bit_len - 1;
        if (run_first) {
            WaveletMatrixMultiCu4G_first_gpu<ThreadsDimY> <<<grid, thread, 0, main_stream >>> (T(mask / 2), block_pair_num, size_div_warp, src_cu, nsum_zeros_buf, buf_byte_div32);
        }
        WaveletMatrixMultiCu4G_ExclusiveSum<MAX_BLOCK_X, IdxType> <<< wm_num, grid_x, 0, main_stream >>> (nsum_zeros_buf, nsum_zeros_buf2, get_bv_block_cu(h) + nsum_pos, buf_byte_div32, bv_block_byte_div32);

        for (; h > 8; --h) {
            using SrcT = T;
            using DstT = T;
            mask >>= 1;
            BlockT* bv_block_nbit_cu_h = get_bv_block_cu(h);
            WaveletMatrixMultiCu4G_UpSweep_gpu<ThreadsDimY, SRCB_S_T, DSTB_S_T> <<<grid, thread, SRCB_S_T + DSTB_S_T, main_stream >>> ((SrcT)mask, block_pair_num, size_div_w, (SrcT*)now_cu, (DstT*)nxt_cu, bv_block_nbit_cu_h, nsum_zeros_buf, nsum_zeros_buf2, bv_block_byte_div32, buf_byte_div32);

            BlockT* nsum_p = get_bv_block_cu(h - 1) + nsum_pos;
            WaveletMatrixMultiCu4G_ExclusiveSum<MAX_BLOCK_X, IdxType> <<< wm_num, grid_x, 0, main_stream >>> (nsum_zeros_buf2, nsum_zeros_buf, nsum_p, buf_byte_div32, bv_block_byte_div32);
            swap(nsum_zeros_buf, nsum_zeros_buf2);
            swap(now_cu, nxt_cu);
        }
        if (h == 8 || (is_same<T, uint16_t>::value && bit_len <= 8 && h == bit_len - 1)) {
            using SrcT = T;
            using DstT = uint8_t;
            mask >>= 1;
            BlockT* bv_block_nbit_cu_h = get_bv_block_cu(h);
            WaveletMatrixMultiCu4G_UpSweep_gpu<ThreadsDimY, SRCB_S_T, DSTB_S_T> <<<grid, thread, SRCB_S_T + DSTB_S_T, main_stream >>> ((SrcT)mask, block_pair_num, size_div_w, (SrcT*)now_cu, (DstT*)nxt_cu, bv_block_nbit_cu_h, nsum_zeros_buf, nsum_zeros_buf2, bv_block_byte_div32, buf_byte_div32);
            if (h == 0) return;
            BlockT* nsum_p = get_bv_block_cu(h - 1) + nsum_pos;
            WaveletMatrixMultiCu4G_ExclusiveSum<MAX_BLOCK_X, IdxType> <<< wm_num, grid_x, 0, main_stream >>> (nsum_zeros_buf2, nsum_zeros_buf, nsum_p, buf_byte_div32, bv_block_byte_div32);
            swap(nsum_zeros_buf, nsum_zeros_buf2);
            swap(now_cu, nxt_cu);
            --h;
        }

        for (; h >= 0; --h) {
            using SrcT = uint8_t;
            using DstT = uint8_t;
            mask >>= 1;
            BlockT* bv_block_nbit_cu_h = get_bv_block_cu(h);
            WaveletMatrixMultiCu4G_UpSweep_gpu<ThreadsDimY, SRCB_S_8, DSTB_S_8> <<<grid, thread, SRCB_S_8 + DSTB_S_8, main_stream >>> ((SrcT)mask, block_pair_num, size_div_w, (SrcT*)now_cu, (DstT*)nxt_cu, bv_block_nbit_cu_h, nsum_zeros_buf, nsum_zeros_buf2, bv_block_byte_div32, buf_byte_div32);
            if (h == 0) break;
            BlockT* nsum_p = get_bv_block_cu(h - 1) + nsum_pos;
            WaveletMatrixMultiCu4G_ExclusiveSum<MAX_BLOCK_X, IdxType> <<< wm_num, grid_x, 0, main_stream >>> (nsum_zeros_buf2, nsum_zeros_buf, nsum_p, buf_byte_div32, bv_block_byte_div32);
            swap(nsum_zeros_buf, nsum_zeros_buf2);
            swap(now_cu, nxt_cu);
        }
    }

    IdxType get_nsum_pos() const {
        const IdxType size_div_w = size / WORD_SIZE;
        return size_div_w;
    }
    IdxType get_bv_block_h_byte_div32() const {
        return (bv_block_len * (sizeof(WordT) + sizeof(IdxType))) / 32u;
    }
};

} // end namespace wavelet_median
}}} //end namespace cv::cuda::device

#endif
#endif // __OPENCV_WAVELET_MATRIX_MULTI_CUH__
