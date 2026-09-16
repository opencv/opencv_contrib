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

#include "precomp.hpp"

using namespace cv;
using namespace cv::cuda;

#if !defined (HAVE_CUDA) || defined (CUDA_DISABLER)

void cv::cuda::nms(InputArray, InputArray, InputArray, OutputArray, float, float, Stream&) { throw_no_cuda(); }

#else

#ifdef __HIP_PLATFORM_AMD__

namespace cv { namespace cuda { namespace detail {
    void nmsBuildMaskHip(const GpuMat& boxes, const GpuMat& classes,
                         int n, float iouThr,
                         std::vector<unsigned int>& hostMask, int& maskCols,
                         cudaStream_t stream);
}}}

void cv::cuda::nms(InputArray _boxes, InputArray _scores, InputArray _classes,
                   OutputArray _indices, float scoreThr, float iouThr, Stream& stream)
{
    GpuMat boxes   = getInputMat(_boxes, stream);
    GpuMat scores  = getInputMat(_scores, stream);
    GpuMat classes = getInputMat(_classes, stream);

    CV_Assert(boxes.type() == CV_32FC1 && boxes.cols == 4);
    const int total = boxes.rows;
    CV_Assert((size_t)(scores.rows * scores.cols) == (size_t)total);
    CV_Assert((size_t)(classes.rows * classes.cols) == (size_t)total);

    // Pull scores/classes to host for the (cheap) score filter + score sort.
    // The expensive O(N^2) IoU work stays on the GPU.
    Mat scoresH, classesH, boxesH;
    scores.download(scoresH);   scoresH  = scoresH.reshape(1, total);
    classes.download(classesH); classesH = classesH.reshape(1, total);
    boxes.download(boxesH);

    const float* sp = scoresH.ptr<float>();

    // Candidate indices passing the score threshold, sorted by descending score.
    std::vector<int> order;
    order.reserve(total);
    for (int i = 0; i < total; ++i)
        if (sp[i] >= scoreThr)
            order.push_back(i);

    if (order.empty())
    {
        _indices.release();
        return;
    }

    std::sort(order.begin(), order.end(),
              [&](int a, int b) { return sp[a] > sp[b]; });

    const int n = (int)order.size();

    // Build the score-sorted, filtered box/class matrices and upload them.
    Mat boxesSorted(n, 4, CV_32F);
    Mat classesSorted(n, 1, CV_32S);
    for (int r = 0; r < n; ++r)
    {
        const int src = order[r];
        memcpy(boxesSorted.ptr<float>(r), boxesH.ptr<float>(src), 4 * sizeof(float));
        classesSorted.at<int>(r, 0) = classesH.ptr<int>(src)[0];
    }

    GpuMat dBoxes, dClasses;
    dBoxes.upload(boxesSorted, stream);
    dClasses.upload(classesSorted, stream);
    cudaStream_t cudaStream = StreamAccessor::getStream(stream);

    std::vector<unsigned int> mask;
    int maskCols = 0;
    detail::nmsBuildMaskHip(dBoxes, dClasses, n, iouThr, mask, maskCols, cudaStream);

    // Greedy sweep over survivors in score order (host, O(N^2/32) word ops).
    std::vector<unsigned char> removed(n, 0);
    std::vector<int> keep;
    for (int i = 0; i < n; ++i)
    {
        if (removed[i])
            continue;
        keep.push_back(order[i]);   // map back to the caller's original index
        const unsigned int* row = &mask[(size_t)i * maskCols];
        for (int j = i + 1; j < n; ++j)
            if (!removed[j] && (row[j >> 5] & (1u << (j & 31))))
                removed[j] = 1;
    }

    if (keep.empty())
    {
        _indices.release();
        return;
    }

    Mat out(1, (int)keep.size(), CV_32S, keep.data());
    if (_indices.isGpuMat())
    {
        _indices.create(1, (int)keep.size(), CV_32S);
        _indices.getGpuMatRef().upload(out);
    }
    else
    {
        out.copyTo(_indices);
    }
}

#else  // NVIDIA CUDA build: not implemented here (AMD/HIP demo primitive)

void cv::cuda::nms(InputArray, InputArray, InputArray, OutputArray, float, float, Stream&)
{
    CV_Error(Error::StsNotImplemented, "cv::cuda::nms is currently implemented for the HIP/ROCm build only");
}

#endif  // __HIP_PLATFORM_AMD__

#endif  // HAVE_CUDA
