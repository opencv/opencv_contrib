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
#include "mvn_layer.hpp"
#include <opencv2/dnn/shape_utils.hpp>

namespace cv
{
namespace dnn
{

MVNLayerImpl::MVNLayerImpl(bool normVariance_, bool acrossChannels_, double eps_)
{
    normVariance = normVariance_;
    acrossChannels = acrossChannels_;
    eps = eps_;
}

void MVNLayerImpl::allocate(const std::vector<Blob *> &inputs, std::vector<Blob> &outputs)
{
    outputs.resize(inputs.size());
    for (size_t i = 0; i < inputs.size(); i++)
    {
        CV_Assert(!acrossChannels || inputs[i]->dims() >= 2);
        outputs[i].create(inputs[i]->shape(), inputs[i]->type());
    }
}

void MVNLayerImpl::forward(std::vector<Blob *> &inputs, std::vector<Blob> &outputs)
{
    for (size_t inpIdx = 0; inpIdx < inputs.size(); inpIdx++)
    {
        Blob &inpBlob = *inputs[inpIdx];
        Blob &outBlob = outputs[inpIdx];

        int splitDim = (acrossChannels) ? 1 : 2;
        Shape workSize((int)inpBlob.total(0, splitDim), (int)inpBlob.total(splitDim));
        Mat inpMat = reshaped(inpBlob.matRefConst(), workSize);
        Mat outMat = reshaped(outBlob.matRef(), workSize);

        Scalar mean, dev;
        for (int i = 0; i < workSize[0]; i++)
        {
            Mat inpRow = inpMat.row(i);
            Mat outRow = outMat.row(i);

            cv::meanStdDev(inpRow, mean, (normVariance) ? dev : noArray());
            double alpha = (normVariance) ? 1/(eps + dev[0]) : 1;
            inpRow.convertTo(outRow, outRow.type(), alpha, -mean[0] * alpha);
        }
    }
}


Ptr<MVNLayer> MVNLayer::create(bool normVariance, bool acrossChannels, double eps)
{
    return Ptr<MVNLayer>(new MVNLayerImpl(normVariance, acrossChannels, eps));
}

}
}
