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

#ifndef OPENCV_CUDEV_GRID_TRANSPOSE_DETAIL_HPP
#define OPENCV_CUDEV_GRID_TRANSPOSE_DETAIL_HPP

#include <algorithm>
#include "../../common.hpp"
#include "../../util/saturate_cast.hpp"
#include "../../ptr2d/glob.hpp"
#include "../../ptr2d/traits.hpp"

namespace cv { namespace cudev {

namespace transpose_detail
{
    template <int TILE_DIM, int BLOCK_DIM_Y, class SrcPtr, typename DstType>
    __global__ void transpose(const SrcPtr src, GlobPtr<DstType> dst, const int rows, const int cols)
    {
        typedef typename PtrTraits<SrcPtr>::value_type src_type;

        __shared__ src_type tile[TILE_DIM][TILE_DIM + 1];

        const uint numTilesX = (static_cast<uint>(cols) + TILE_DIM - 1u) / TILE_DIM;
        const uint numTilesY = (static_cast<uint>(rows) + TILE_DIM - 1u) / TILE_DIM;

        for (uint by = blockIdx.y; by < numTilesY; by += gridDim.y)
        {
            uint blockIdx_x, blockIdx_y;

            // do diagonal reordering over logical tile grid (numTilesX, numTilesY)
            if (numTilesX == numTilesY)
            {
                blockIdx_y = blockIdx.x;
                blockIdx_x = (blockIdx.x + by) % numTilesX;
            }
            else
            {
                const unsigned long long bid = static_cast<unsigned long long>(blockIdx.x) +
                                               static_cast<unsigned long long>(numTilesX) * by;
                blockIdx_y = static_cast<uint>(bid % numTilesY);
                blockIdx_x = static_cast<uint>(((bid / numTilesY) + blockIdx_y) % numTilesX);
            }

            uint xIndex = blockIdx_x * TILE_DIM + threadIdx.x;
            uint yIndex = blockIdx_y * TILE_DIM + threadIdx.y;

            if (xIndex < static_cast<uint>(cols))
            {
                for (int i = 0; i < TILE_DIM; i += BLOCK_DIM_Y)
                {
                    if (yIndex + static_cast<uint>(i) < static_cast<uint>(rows))
                    {
                        tile[threadIdx.y + i][threadIdx.x] = src(static_cast<int>(yIndex) + i, static_cast<int>(xIndex));
                    }
                }
            }

            __syncthreads();

            xIndex = blockIdx_y * TILE_DIM + threadIdx.x;
            yIndex = blockIdx_x * TILE_DIM + threadIdx.y;

            if (xIndex < static_cast<uint>(rows))
            {
                for (int i = 0; i < TILE_DIM; i += BLOCK_DIM_Y)
                {
                    if (yIndex + static_cast<uint>(i) < static_cast<uint>(cols))
                    {
                        dst(static_cast<int>(yIndex) + i, static_cast<int>(xIndex)) = saturate_cast<DstType>(tile[threadIdx.x][threadIdx.y + i]);
                    }
                }
            }

            __syncthreads();
        }
    }

    template <class Policy, class SrcPtr, typename DstType>
    __host__ void transpose(const SrcPtr& src, const GlobPtr<DstType>& dst, int rows, int cols, cudaStream_t stream)
    {
        if (rows <= 0 || cols <= 0)
            return;

        const dim3 block(Policy::tile_dim, Policy::block_dim_y);
        const uint numTilesX = (static_cast<uint>(cols) + Policy::tile_dim - 1u) / Policy::tile_dim;
        const uint numTilesY = (static_cast<uint>(rows) + Policy::tile_dim - 1u) / Policy::tile_dim;
        const dim3 grid(numTilesX, std::min(numTilesY, 65535u));

        transpose<Policy::tile_dim, Policy::block_dim_y><<<grid, block, 0, stream>>>(src, dst, rows, cols);
        CV_CUDEV_SAFE_CALL( cudaGetLastError() );

        if (stream == 0)
            CV_CUDEV_SAFE_CALL( cudaDeviceSynchronize() );
    }
}

}}

#endif
