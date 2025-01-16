/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                          License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
// Copyright (C) 2013, OpenCV Foundation, all rights reserved.
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

#pragma once

#ifndef OPENCV_CUDEV_BLOCK_REDUCE_HPP
#define OPENCV_CUDEV_BLOCK_REDUCE_HPP

#include "../common.hpp"
#include "../util/tuple.hpp"
#include "../warp/reduce.hpp"
#include "detail/reduce.hpp"
#include "detail/reduce_key_val.hpp"
#include <cuda_runtime_api.h>

namespace cv { namespace cudev {

//! @addtogroup cudev
//! @{

// blockReduce

template <int N, typename T, class Op>
__device__ __forceinline__ void blockReduce(volatile T* smem, T& val, uint tid, const Op& op)
{
    block_reduce_detail::Dispatcher<N>::reductor::template reduce<volatile T*, T&, const Op&>(smem, val, tid, op);
}

#if (CUDART_VERSION < 12040)
template <int N,
          typename P0, typename P1, typename P2, typename P3, typename P4, typename P5, typename P6, typename P7, typename P8, typename P9,
          typename R0, typename R1, typename R2, typename R3, typename R4, typename R5, typename R6, typename R7, typename R8, typename R9,
          class Op0, class Op1, class Op2, class Op3, class Op4, class Op5, class Op6, class Op7, class Op8, class Op9>
__device__ __forceinline__ void blockReduce(const tuple<P0, P1, P2, P3, P4, P5, P6, P7, P8, P9>& smem,
                                            const tuple<R0, R1, R2, R3, R4, R5, R6, R7, R8, R9>& val,
                                            uint tid,
                                            const tuple<Op0, Op1, Op2, Op3, Op4, Op5, Op6, Op7, Op8, Op9>& op)
{
    block_reduce_detail::Dispatcher<N>::reductor::template reduce<
            const tuple<P0, P1, P2, P3, P4, P5, P6, P7, P8, P9>&,
            const tuple<R0, R1, R2, R3, R4, R5, R6, R7, R8, R9>&,
            const tuple<Op0, Op1, Op2, Op3, Op4, Op5, Op6, Op7, Op8, Op9>&>(smem, val, tid, op);
}

// blockReduceKeyVal

template <int N, typename K, typename V, class Cmp>
__device__ __forceinline__ void blockReduceKeyVal(volatile K* skeys, K& key, volatile V* svals, V& val, uint tid, const Cmp& cmp)
{
    block_reduce_key_val_detail::Dispatcher<N>::reductor::template reduce<volatile K*, K&, volatile V*, V&, const Cmp&>(skeys, key, svals, val, tid, cmp);
}

template <int N,
          typename K,
          typename VP0, typename VP1, typename VP2, typename VP3, typename VP4, typename VP5, typename VP6, typename VP7, typename VP8, typename VP9,
          typename VR0, typename VR1, typename VR2, typename VR3, typename VR4, typename VR5, typename VR6, typename VR7, typename VR8, typename VR9,
          class Cmp>
__device__ __forceinline__ void blockReduceKeyVal(volatile K* skeys, K& key,
                                                  const tuple<VP0, VP1, VP2, VP3, VP4, VP5, VP6, VP7, VP8, VP9>& svals,
                                                  const tuple<VR0, VR1, VR2, VR3, VR4, VR5, VR6, VR7, VR8, VR9>& val,
                                                  uint tid, const Cmp& cmp)
{
    block_reduce_key_val_detail::Dispatcher<N>::reductor::template reduce<volatile K*, K&,
            const tuple<VP0, VP1, VP2, VP3, VP4, VP5, VP6, VP7, VP8, VP9>&,
            const tuple<VR0, VR1, VR2, VR3, VR4, VR5, VR6, VR7, VR8, VR9>&,
            const Cmp&>(skeys, key, svals, val, tid, cmp);
}

template <int N,
          typename KP0, typename KP1, typename KP2, typename KP3, typename KP4, typename KP5, typename KP6, typename KP7, typename KP8, typename KP9,
          typename KR0, typename KR1, typename KR2, typename KR3, typename KR4, typename KR5, typename KR6, typename KR7, typename KR8, typename KR9,
          typename VP0, typename VP1, typename VP2, typename VP3, typename VP4, typename VP5, typename VP6, typename VP7, typename VP8, typename VP9,
          typename VR0, typename VR1, typename VR2, typename VR3, typename VR4, typename VR5, typename VR6, typename VR7, typename VR8, typename VR9,
          class Cmp0, class Cmp1, class Cmp2, class Cmp3, class Cmp4, class Cmp5, class Cmp6, class Cmp7, class Cmp8, class Cmp9>
__device__ __forceinline__ void blockReduceKeyVal(const tuple<KP0, KP1, KP2, KP3, KP4, KP5, KP6, KP7, KP8, KP9>& skeys,
                                                  const tuple<KR0, KR1, KR2, KR3, KR4, KR5, KR6, KR7, KR8, KR9>& key,
                                                  const tuple<VP0, VP1, VP2, VP3, VP4, VP5, VP6, VP7, VP8, VP9>& svals,
                                                  const tuple<VR0, VR1, VR2, VR3, VR4, VR5, VR6, VR7, VR8, VR9>& val,
                                                  uint tid,
                                                  const tuple<Cmp0, Cmp1, Cmp2, Cmp3, Cmp4, Cmp5, Cmp6, Cmp7, Cmp8, Cmp9>& cmp)
{
    block_reduce_key_val_detail::Dispatcher<N>::reductor::template reduce<
            const tuple<KP0, KP1, KP2, KP3, KP4, KP5, KP6, KP7, KP8, KP9>&,
            const tuple<KR0, KR1, KR2, KR3, KR4, KR5, KR6, KR7, KR8, KR9>&,
            const tuple<VP0, VP1, VP2, VP3, VP4, VP5, VP6, VP7, VP8, VP9>&,
            const tuple<VR0, VR1, VR2, VR3, VR4, VR5, VR6, VR7, VR8, VR9>&,
            const tuple<Cmp0, Cmp1, Cmp2, Cmp3, Cmp4, Cmp5, Cmp6, Cmp7, Cmp8, Cmp9>&
            >(skeys, key, svals, val, tid, cmp);
}

#else

template <int N, typename... P, typename... R, typename... Op>
__device__ __forceinline__ void blockReduce(const tuple<P...>& smem,
                                            const tuple<R...>& val,
                                            uint tid,
                                            const tuple<Op...>& op)
{
    block_reduce_detail::Dispatcher<N>::reductor::template reduce<const tuple<P...>&, const tuple<R...>&, const tuple<Op...>&>(smem, val, tid, op);
}

// blockReduceKeyVal

template <int N, typename K, typename V, class Cmp>
__device__ __forceinline__ void blockReduceKeyVal(volatile K* skeys, K& key, volatile V* svals, V& val, uint tid, const Cmp& cmp)
{
    block_reduce_key_val_detail::Dispatcher<N>::reductor::template reduce<volatile K*, K&, volatile V*, V&, const Cmp&>(skeys, key, svals, val, tid, cmp);
}

template <int N, typename K, typename... VP, typename... VR, class Cmp>
__device__ __forceinline__ void blockReduceKeyVal(volatile K* skeys, K& key, const tuple<VP...>& svals, const tuple<VR...>& val, uint tid, const Cmp& cmp)
{
    block_reduce_key_val_detail::Dispatcher<N>::reductor::template reduce<volatile K*, K&, const tuple<VP...>&, const tuple<VR...>&, const Cmp&>(skeys, key, svals, val, tid, cmp);
}

template <int N, typename... KP, typename... KR, typename... VP, typename... VR, class... Cmp>
__device__ __forceinline__ void blockReduceKeyVal(const tuple<KP...>& skeys, const tuple<KR...>& key, const tuple<VP...>& svals, const tuple<VR...>& val, uint tid, const tuple<Cmp...>& cmp)
{
    block_reduce_key_val_detail::Dispatcher<N>::reductor::template reduce< const tuple<KP...>&, const tuple<KR...>&, const tuple<VP...>&, const tuple<VR...>&, const tuple<Cmp...>&>(skeys, key, svals, val, tid, cmp);
}

#endif

//! @}

}}

#endif
