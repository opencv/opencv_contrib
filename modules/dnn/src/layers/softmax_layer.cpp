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
#include "softmax_layer.hpp"
#include <algorithm>
#include <stdlib.h>
using std::max;

namespace cv
{
namespace dnn
{
    //TODO: set default axis number to 1, and add custom shape length in FullyConnected
    SoftMaxLayer::SoftMaxLayer(LayerParams &params) : Layer(params)
    {
        //hotfix!!!
        axis_ = params.get<int>("axis", 1);
    }

    void SoftMaxLayer::allocate(const std::vector<Blob*> &inputs, std::vector<Blob> &outputs)
    {
        CV_Assert(inputs.size() == 1);
        axis = inputs[0]->canonicalAxis(axis_);

        BlobShape shape = inputs[0]->shape();
        outputs.resize(1);
        outputs[0].create(shape);

        shape[axis] = 1;
        maxAggregator.create(shape);
    }

    void SoftMaxLayer::forward(std::vector<Blob*> &inputs, std::vector<Blob> &outputs)
    {
        Blob &src = *inputs[0];
        Blob &dst = outputs[0];

        float *srcPtr = src.ptrf();
        float *dstPtr = dst.ptrf();
        float *bufPtr = maxAggregator.ptrf();

        size_t outerSize = src.total(0, axis);
        size_t channels = src.size(axis);
        size_t innerSize = src.total(axis + 1);

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

        cv::exp(dst.matRef(), dst.matRef());

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

}
}
