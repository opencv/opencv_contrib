/*M///////////////////////////////////////////////////////////////////////////////////////
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (c) 2026 Advanced Micro Devices, Inc.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties are disclaimed.
//
//M*/

// Generic GPU non-maximum suppression for cv::cuda::nms. The GPU kernel builds
// an N x N IoU bitmask (class-aware: boxes of different classes never suppress
// each other); the host then does a cheap greedy sweep over the bitmask in
// score order to pick survivors. This mirrors the classic torchvision / OpenCV
// dnn grid-NMS design but works on plain GpuMats so it is usable as a public,
// Python-wrapped primitive.

#include "opencv2/opencv_modules.hpp"

#ifndef HAVE_OPENCV_CUDEV

#error "opencv_cudev is required"

#else

#include "opencv2/cudev.hpp"
#include "opencv2/core/private.cuda.hpp"

using namespace cv;
using namespace cv::cuda;
using namespace cv::cudev;

namespace
{
    // One thread per (i, j) pair. mask[i * maskCols + (j>>5)] gets bit (j&31)
    // set when box j should be suppressed by box i, i.e. same class and
    // IoU(i, j) > threshold. Only the upper triangle (j > i) is written; the
    // greedy host sweep only ever reads j > i.
    __global__ void nmsMaskKernel(const uchar* boxes, size_t boxStep,
                                  const uchar* classes, size_t classStep,
                                  int n, int maskCols, float iouThr,
                                  unsigned int* mask)
    {
        const int i = blockIdx.y * blockDim.y + threadIdx.y;
        const int j = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= n || j >= n || j <= i)
            return;

        const int ci = *reinterpret_cast<const int*>(classes + (size_t)i * classStep);
        const int cj = *reinterpret_cast<const int*>(classes + (size_t)j * classStep);
        if (ci != cj)
            return;

        const float* bi = reinterpret_cast<const float*>(boxes + (size_t)i * boxStep);
        const float* bj = reinterpret_cast<const float*>(boxes + (size_t)j * boxStep);

        const float ix1 = bi[0], iy1 = bi[1], ix2 = bi[2], iy2 = bi[3];
        const float jx1 = bj[0], jy1 = bj[1], jx2 = bj[2], jy2 = bj[3];

        const float xx1 = fmaxf(ix1, jx1);
        const float yy1 = fmaxf(iy1, jy1);
        const float xx2 = fminf(ix2, jx2);
        const float yy2 = fminf(iy2, jy2);

        const float w = fmaxf(0.0f, xx2 - xx1);
        const float h = fmaxf(0.0f, yy2 - yy1);
        const float inter = w * h;

        const float ai = fmaxf(0.0f, ix2 - ix1) * fmaxf(0.0f, iy2 - iy1);
        const float aj = fmaxf(0.0f, jx2 - jx1) * fmaxf(0.0f, jy2 - jy1);
        const float uni = ai + aj - inter;

        const float iou = uni > 0.0f ? inter / uni : 0.0f;
        if (iou > iouThr)
            atomicOr(&mask[i * maskCols + (j >> 5)], 1u << (j & 31));
    }
}

namespace cv { namespace cuda { namespace detail {

// boxes: (n,4) CV_32F, already sorted by descending score by the caller.
// classes: (n) CV_32S. Returns the IoU bitmask as a host vector (row-major,
// n rows x maskCols uint each). The greedy selection is done host-side.
void nmsBuildMaskHip(const GpuMat& boxes, const GpuMat& classes,
                     int n, float iouThr,
                     std::vector<unsigned int>& hostMask, int& maskCols,
                     cudaStream_t stream)
{
    maskCols = (n + 31) / 32;
    // Single contiguous row so device step == width (no per-row padding); the
    // kernel indexes the mask as a flat i*maskCols + word buffer.
    GpuMat mask(1, n * maskCols, CV_32S);
    mask.setTo(Scalar::all(0));

    const dim3 block(16, 16);
    const dim3 grid(divUp(n, block.x), divUp(n, block.y));
    nmsMaskKernel<<<grid, block, 0, stream>>>(
        boxes.data, boxes.step, classes.data, classes.step,
        n, maskCols, iouThr,
        reinterpret_cast<unsigned int*>(mask.ptr<int>()));
    CV_CUDEV_SAFE_CALL(cudaGetLastError());
    if (stream == 0)
        CV_CUDEV_SAFE_CALL(cudaDeviceSynchronize());

    // Pull the bitmask back to host for the greedy sweep (n<=few thousand,
    // so this is a small transfer).
    Mat maskHost;
    mask.download(maskHost);
    hostMask.resize((size_t)n * maskCols);
    memcpy(hostMask.data(), maskHost.ptr<int>(), (size_t)n * maskCols * sizeof(unsigned int));
}

}}}

#endif
