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

#ifndef OPENCV_CUDEV_BLOCK_REDUCE_KEY_VAL_DETAIL_HPP
#define OPENCV_CUDEV_BLOCK_REDUCE_KEY_VAL_DETAIL_HPP

#include "../../common.hpp"
#include "../../util/tuple.hpp"
#include "../../util/type_traits.hpp"
#include "../../warp/warp.hpp"

namespace cv { namespace cudev {

namespace block_reduce_key_val_detail
{
    // GetType

    template <typename T> struct GetType;

    template <typename T> struct GetType<T*>
    {
        typedef T type;
    };

    template <typename T> struct GetType<volatile T*>
    {
        typedef T type;
    };

    template <typename T> struct GetType<T&>
    {
        typedef T type;
    };

    // For

    template <int I, int N> struct For
    {
        template <class PointerTuple, class ReferenceTuple>
        __device__ static void loadToSmem(const PointerTuple& smem, const ReferenceTuple& data, uint tid)
        {
            get<I>(smem)[tid] = get<I>(data);

            For<I + 1, N>::loadToSmem(smem, data, tid);
        }

        template <class PointerTuple, class ReferenceTuple>
        __device__ static void loadFromSmem(const PointerTuple& smem, const ReferenceTuple& data, uint tid)
        {
            get<I>(data) = get<I>(smem)[tid];

            For<I + 1, N>::loadFromSmem(smem, data, tid);
        }

        template <class PointerTuple, class ReferenceTuple>
        __device__ static void copy(const PointerTuple& svals, const ReferenceTuple& val, uint tid, uint delta)
        {
            get<I>(svals)[tid] = get<I>(val) = get<I>(svals)[tid + delta];

            For<I + 1, N>::copy(svals, val, tid, delta);
        }

        template <class KeyPointerTuple, class KeyReferenceTuple, class ValPointerTuple, class ValReferenceTuple, class CmpTuple>
        __device__ static void merge(const KeyPointerTuple& skeys, const KeyReferenceTuple& key,
                                     const ValPointerTuple& svals, const ValReferenceTuple& val,
                                     const CmpTuple& cmp,
                                     uint tid, uint delta)
        {
            typename GetType<typename tuple_element<I, KeyPointerTuple>::type>::type reg = get<I>(skeys)[tid + delta];

            if (get<I>(cmp)(reg, get<I>(key)))
            {
                get<I>(skeys)[tid] = get<I>(key) = reg;
                get<I>(svals)[tid] = get<I>(val) = get<I>(svals)[tid + delta];
            }

            For<I + 1, N>::merge(skeys, key, svals, val, cmp, tid, delta);
        }
    };

    template <int N> struct For<N, N>
    {
        template <class PointerTuple, class ReferenceTuple>
        __device__ static void loadToSmem(const PointerTuple&, const ReferenceTuple&, uint)
        {
        }

        template <class PointerTuple, class ReferenceTuple>
        __device__ static void loadFromSmem(const PointerTuple&, const ReferenceTuple&, uint)
        {
        }

        template <class PointerTuple, class ReferenceTuple>
        __device__ static void copy(const PointerTuple&, const ReferenceTuple&, uint, uint)
        {
        }

        template <class KeyPointerTuple, class KeyReferenceTuple, class ValPointerTuple, class ValReferenceTuple, class CmpTuple>
        __device__ static void merge(const KeyPointerTuple&, const KeyReferenceTuple&,
                                     const ValPointerTuple&, const ValReferenceTuple&,
                                     const CmpTuple&,
                                     uint, uint)
        {
        }
    };

    // loadToSmem / loadFromSmem

    template <typename T>
    __device__ __forceinline__ void loadToSmem(volatile T* smem, T& data, uint tid)
    {
        smem[tid] = data;
    }

    template <typename T>
    __device__ __forceinline__ void loadFromSmem(volatile T* smem, T& data, uint tid)
    {
        data = smem[tid];
    }

#if (CUDART_VERSION < 12040)
    template <typename VP0, typename VP1, typename VP2, typename VP3, typename VP4, typename VP5, typename VP6, typename VP7, typename VP8, typename VP9,
              typename VR0, typename VR1, typename VR2, typename VR3, typename VR4, typename VR5, typename VR6, typename VR7, typename VR8, typename VR9>
    __device__ __forceinline__ void loadToSmem(const tuple<VP0, VP1, VP2, VP3, VP4, VP5, VP6, VP7, VP8, VP9>& smem,
                                               const tuple<VR0, VR1, VR2, VR3, VR4, VR5, VR6, VR7, VR8, VR9>& data,
                                               uint tid)
    {
        For<0, tuple_size<tuple<VP0, VP1, VP2, VP3, VP4, VP5, VP6, VP7, VP8, VP9> >::value>::loadToSmem(smem, data, tid);
    }

    template <typename VP0, typename VP1, typename VP2, typename VP3, typename VP4, typename VP5, typename VP6, typename VP7, typename VP8, typename VP9,
              typename VR0, typename VR1, typename VR2, typename VR3, typename VR4, typename VR5, typename VR6, typename VR7, typename VR8, typename VR9>
    __device__ __forceinline__ void loadFromSmem(const tuple<VP0, VP1, VP2, VP3, VP4, VP5, VP6, VP7, VP8, VP9>& smem,
                                                 const tuple<VR0, VR1, VR2, VR3, VR4, VR5, VR6, VR7, VR8, VR9>& data,
                                                 uint tid)
    {
        For<0, tuple_size<tuple<VP0, VP1, VP2, VP3, VP4, VP5, VP6, VP7, VP8, VP9> >::value>::loadFromSmem(smem, data, tid);
    }

    // copyVals

    template <typename V>
    __device__ __forceinline__ void copyVals(volatile V* svals, V& val, uint tid, uint delta)
    {
        svals[tid] = val = svals[tid + delta];
    }

    template <typename VP0, typename VP1, typename VP2, typename VP3, typename VP4, typename VP5, typename VP6, typename VP7, typename VP8, typename VP9,
              typename VR0, typename VR1, typename VR2, typename VR3, typename VR4, typename VR5, typename VR6, typename VR7, typename VR8, typename VR9>
    __device__ __forceinline__ void copyVals(const tuple<VP0, VP1, VP2, VP3, VP4, VP5, VP6, VP7, VP8, VP9>& svals,
                                             const tuple<VR0, VR1, VR2, VR3, VR4, VR5, VR6, VR7, VR8, VR9>& val,
                                             uint tid, uint delta)
    {
        For<0, tuple_size<tuple<VP0, VP1, VP2, VP3, VP4, VP5, VP6, VP7, VP8, VP9> >::value>::copy(svals, val, tid, delta);
    }

    // merge

    template <typename K, typename V, class Cmp>
    __device__ void merge(volatile K* skeys, K& key, volatile V* svals, V& val, const Cmp& cmp, uint tid, uint delta)
    {
        K reg = skeys[tid + delta];

        if (cmp(reg, key))
        {
            skeys[tid] = key = reg;
            copyVals(svals, val, tid, delta);
        }
    }

    template <typename K,
              typename VP0, typename VP1, typename VP2, typename VP3, typename VP4, typename VP5, typename VP6, typename VP7, typename VP8, typename VP9,
              typename VR0, typename VR1, typename VR2, typename VR3, typename VR4, typename VR5, typename VR6, typename VR7, typename VR8, typename VR9,
              class Cmp>
    __device__ void merge(volatile K* skeys, K& key,
                          const tuple<VP0, VP1, VP2, VP3, VP4, VP5, VP6, VP7, VP8, VP9>& svals,
                          const tuple<VR0, VR1, VR2, VR3, VR4, VR5, VR6, VR7, VR8, VR9>& val,
                          const Cmp& cmp, uint tid, uint delta)
    {
        K reg = skeys[tid + delta];

        if (cmp(reg, key))
        {
            skeys[tid] = key = reg;
            copyVals(svals, val, tid, delta);
        }
    }

    template <typename KP0, typename KP1, typename KP2, typename KP3, typename KP4, typename KP5, typename KP6, typename KP7, typename KP8, typename KP9,
              typename KR0, typename KR1, typename KR2, typename KR3, typename KR4, typename KR5, typename KR6, typename KR7, typename KR8, typename KR9,
              typename VP0, typename VP1, typename VP2, typename VP3, typename VP4, typename VP5, typename VP6, typename VP7, typename VP8, typename VP9,
              typename VR0, typename VR1, typename VR2, typename VR3, typename VR4, typename VR5, typename VR6, typename VR7, typename VR8, typename VR9,
              class Cmp0, class Cmp1, class Cmp2, class Cmp3, class Cmp4, class Cmp5, class Cmp6, class Cmp7, class Cmp8, class Cmp9>
    __device__ __forceinline__ void merge(const tuple<KP0, KP1, KP2, KP3, KP4, KP5, KP6, KP7, KP8, KP9>& skeys,
                                          const tuple<KR0, KR1, KR2, KR3, KR4, KR5, KR6, KR7, KR8, KR9>& key,
                                          const tuple<VP0, VP1, VP2, VP3, VP4, VP5, VP6, VP7, VP8, VP9>& svals,
                                          const tuple<VR0, VR1, VR2, VR3, VR4, VR5, VR6, VR7, VR8, VR9>& val,
                                          const tuple<Cmp0, Cmp1, Cmp2, Cmp3, Cmp4, Cmp5, Cmp6, Cmp7, Cmp8, Cmp9>& cmp,
                                          uint tid, uint delta)
    {
        For<0, tuple_size<tuple<VP0, VP1, VP2, VP3, VP4, VP5, VP6, VP7, VP8, VP9> >::value>::merge(skeys, key, svals, val, cmp, tid, delta);
    }
#else
    template <typename... VP, typename... VR>
    __device__ __forceinline__ void loadToSmem(const tuple<VP...>& smem, const tuple<VR...>& data, uint tid)
    {
        For<0, tuple_size<tuple<VP...> >::value>::loadToSmem(smem, data, tid);
    }

    template <typename... VP, typename... VR>
    __device__ __forceinline__ void loadFromSmem(const tuple<VP...>& smem, const tuple<VR...>& data, uint tid)
    {
        For<0, tuple_size<tuple<VP...> >::value>::loadFromSmem(smem, data, tid);
    }

    // copyVals

    template <typename V>
    __device__ __forceinline__ void copyVals(volatile V* svals, V& val, uint tid, uint delta)
    {
        svals[tid] = val = svals[tid + delta];
    }

    template <typename... VP, typename... VR>
    __device__ __forceinline__ void copyVals(const tuple<VP...>& svals, const tuple<VR...>& val, uint tid, uint delta)
    {
        For<0, tuple_size<tuple<VP...> >::value>::copy(svals, val, tid, delta);
    }

    // merge

    template <typename K, typename V, class Cmp>
    __device__ void merge(volatile K* skeys, K& key, volatile V* svals, V& val, const Cmp& cmp, uint tid, uint delta)
    {
        K reg = skeys[tid + delta];

        if (cmp(reg, key))
        {
            skeys[tid] = key = reg;
            copyVals(svals, val, tid, delta);
        }
    }

    template <typename K, typename... VP, typename... VR, class Cmp>
    __device__ void merge(volatile K* skeys, K& key, const tuple<VP...>& svals, const tuple<VR...>& val, const Cmp& cmp, uint tid, uint delta)
    {
        K reg = skeys[tid + delta];

        if (cmp(reg, key))
        {
            skeys[tid] = key = reg;
            copyVals(svals, val, tid, delta);
        }
    }

    template <typename... KP, typename... KR, typename... VP, typename... VR, class... Cmp>
    __device__ __forceinline__ void merge(const tuple<KP...>& skeys, const tuple<KR...>& key,
                                          const tuple<VP...>& svals, const tuple<VR...>& val,
                                          const tuple<Cmp...>& cmp, uint tid, uint delta)
    {
        For<0, tuple_size<tuple<VP...> >::value>::merge(skeys, key, svals, val, cmp, tid, delta);
    }
#endif

    // Generic

    template <int N> struct Generic
    {
        template <class KP, class KR, class VP, class VR, class Cmp>
        __device__ static void reduce(KP skeys, KR key, VP svals, VR val, uint tid, Cmp cmp)
        {
            loadToSmem(skeys, key, tid);
            loadValsToSmem(svals, val, tid);
            if (N >= 32)
                __syncthreads();

            if (N >= 2048)
            {
                if (tid < 1024)
                    merge(skeys, key, svals, val, cmp, tid, 1024);

                __syncthreads();
            }
            if (N >= 1024)
            {
                if (tid < 512)
                    merge(skeys, key, svals, val, cmp, tid, 512);

                __syncthreads();
            }
            if (N >= 512)
            {
                if (tid < 256)
                    merge(skeys, key, svals, val, cmp, tid, 256);

                __syncthreads();
            }
            if (N >= 256)
            {
                if (tid < 128)
                    merge(skeys, key, svals, val, cmp, tid, 128);

                __syncthreads();
            }
            if (N >= 128)
            {
                if (tid < 64)
                    merge(skeys, key, svals, val, cmp, tid, 64);

                __syncthreads();
            }
            if (N >= 64)
            {
                if (tid < 32)
                    merge(skeys, key, svals, val, cmp, tid, 32);
            }

            if (tid < 16)
            {
                merge(skeys, key, svals, val, cmp, tid, 16);
                merge(skeys, key, svals, val, cmp, tid, 8);
                merge(skeys, key, svals, val, cmp, tid, 4);
                merge(skeys, key, svals, val, cmp, tid, 2);
                merge(skeys, key, svals, val, cmp, tid, 1);
            }
        }
    };

    // Unroll

    template <int I, class KP, class KR, class VP, class VR, class Cmp> struct Unroll
    {
        __device__ static void loop(KP skeys, KR key, VP svals, VR val, uint tid, Cmp cmp)
        {
            merge(skeys, key, svals, val, cmp, tid, I);
            Unroll<I / 2, KP, KR, VP, VR, Cmp>::loop(skeys, key, svals, val, tid, cmp);
        }
    };

    template <class KP, class KR, class VP, class VR, class Cmp> struct Unroll<0, KP, KR, VP, VR, Cmp>
    {
        __device__ __forceinline__ static void loop(KP, KR, VP, VR, uint, Cmp)
        {
        }
    };

    // WarpOptimized

    template <int N> struct WarpOptimized
    {
        template <class KP, class KR, class VP, class VR, class Cmp>
        __device__ static void reduce(KP skeys, KR key, VP svals, VR val, uint tid, Cmp cmp)
        {
            loadToSmem(skeys, key, tid);
            loadToSmem(svals, val, tid);

            if (tid < N / 2)
                Unroll<N / 2, KP, KR, VP, VR, Cmp>::loop(skeys, key, svals, val, tid, cmp);
        }
    };

    // GenericOptimized32

    template <uint N> struct GenericOptimized32
    {
        enum { M = N / 32 };

        template <class KP, class KR, class VP, class VR, class Cmp>
        __device__ static void reduce(KP skeys, KR key, VP svals, VR val, uint tid, Cmp cmp)
        {
            const uint laneId = Warp::laneId();

            loadToSmem(skeys, key, tid);
            loadToSmem(svals, val, tid);

            if (laneId < 16)
                Unroll<16, KP, KR, VP, VR, Cmp>::loop(skeys, key, svals, val, tid, cmp);

            __syncthreads();

            if (laneId == 0)
            {
                loadToSmem(skeys, key, tid / 32);
                loadToSmem(svals, val, tid / 32);
            }

            __syncthreads();

            loadFromSmem(skeys, key, tid);

            if (tid < 32)
            {
                Unroll<M / 2, KP, KR, VP, VR, Cmp>::loop(skeys, key, svals, val, tid, cmp);
            }
        }
    };

    template <int N> struct Dispatcher
    {
        typedef typename SelectIf<
            (N <= 32) && IsPowerOf2<N>::value,
            WarpOptimized<N>,
            typename SelectIf<
                (N <= 1024) && IsPowerOf2<N>::value,
                GenericOptimized32<N>,
                Generic<N>
            >::type
        >::type reductor;
    };
}

}}

#endif
