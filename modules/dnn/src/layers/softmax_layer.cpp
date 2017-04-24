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

#include "../precomp.hpp"
#include "layers_common.hpp"
#include <algorithm>
#include <stdlib.h>
using std::max;

namespace cv
{
namespace dnn
{

class SoftMaxLayerImpl : public SoftmaxLayer
{
public:

    SoftMaxLayerImpl(const LayerParams& params)
    {
        axisRaw = params.get<int>("axis", 1);
        setParamsFrom(params);
    }

    void allocate(const std::vector<Mat*> &inputs, std::vector<Mat> &outputs)
    {
        CV_Assert(inputs.size() == 1);
        const Mat& inp0 = *inputs[0];
        int dims = inp0.dims;
        axis = axisRaw < 0 ? axisRaw + dims : axisRaw;

        outerSize = inp0.total(0, axis);
        channels = inp0.size[axis];
        innerSize = inp0.total(axis + 1);

        std::vector<int> shape(inp0.size.p, inp0.size.p + dims);
        shape[axis] = 1;
        buf.create(shape, inp0.type());

        outputs.resize(1);
        outputs[0].create(inp0.dims, inp0.size.p, inp0.type());
    }

    void forward(std::vector<Mat*> &inputs, std::vector<Mat> &outputs)
    {
        const Mat &src = *inputs[0];
        Mat &dst = outputs[0];

        CV_Assert(src.type() == CV_32F);
        CV_Assert(src.isContinuous() && dst.isContinuous());

        const float *srcPtr = src.ptr<float>();
        float *dstPtr = dst.ptr<float>();
        float *bufPtr = buf.ptr<float>();

        size_t outerStep = src.total(axis);
        size_t cnStep = src.total(axis + 1);

        //compute max along axis
        for (size_t outerDim = 0; outerDim < outerSize; outerDim++)
        {
            size_t srcOffset = outerDim * outerStep;
            size_t bufOffset = outerDim * cnStep;

            memcpy(bufPtr + bufOffset, srcPtr + srcOffset, innerSize * sizeof(float));

            for (size_t cnDim = 1; cnDim < channels; cnDim++)
            {
                for (size_t i = 0; i < innerSize; i++)
                    bufPtr[bufOffset + i] = std::max(bufPtr[bufOffset + i], srcPtr[srcOffset + cnDim * cnStep + i]);
            }
        }

        //subtract max
        for (size_t outerDim = 0; outerDim < outerSize; outerDim++)
        {
            size_t srcOffset = outerDim * outerStep;
            size_t bufOffset = outerDim * cnStep;

            for (size_t cnDim = 0; cnDim < channels; cnDim++)
            {
                for (size_t i = 0; i < innerSize; i++)
                    dstPtr[srcOffset + cnDim * cnStep + i] = srcPtr[srcOffset + cnDim * cnStep + i] - bufPtr[bufOffset + i];
            }
        }

        cv::exp(dst, dst);

        for (size_t outerDim = 0; outerDim < outerSize; outerDim++)
        {
            size_t srcOffset = outerDim * outerStep;
            size_t bufOffset = outerDim * cnStep;

            //sum exp along axis
            for (size_t i = 0; i < innerSize; i++)
                bufPtr[bufOffset + i] = 0.f;

            for (size_t cnDim = 0; cnDim < channels; cnDim++)
            {
                for (size_t i = 0; i < innerSize; i++)
                    bufPtr[bufOffset + i] += dstPtr[srcOffset + cnDim * cnStep + i];
            }

            //divide by computed sum
            for (size_t cnDim = 0; cnDim < channels; cnDim++)
            {
                for (size_t i = 0; i < innerSize; i++)
                    dstPtr[srcOffset + cnDim * cnStep + i] /= bufPtr[bufOffset + i];
            }
        }
    }

    int axis, axisRaw;
    Mat buf;
    size_t outerSize, channels, innerSize;
};

Ptr<SoftmaxLayer> SoftmaxLayer::create(const LayerParams& params)
{
    return Ptr<SoftmaxLayer>(new SoftMaxLayerImpl(params));
}

}
}
