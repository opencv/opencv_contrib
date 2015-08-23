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

namespace cv
{
namespace dnn
{

MVNLayer::MVNLayer(LayerParams &params) : Layer(params)
{
    eps = params.get<double>("eps", 1e-9);
    acrossChannels = params.get<bool>("across_channels", false);
    normalizeVariance = params.get<bool>("normalize_variance", true);
}

void MVNLayer::allocate(const std::vector<Blob *> &inputs, std::vector<Blob> &outputs)
{
    outputs.resize(inputs.size());
    for (size_t i = 0; i < inputs.size(); i++)
    {
        CV_Assert(!acrossChannels || inputs[i]->dims() >= 2);
        outputs[i].create(inputs[i]->shape(), inputs[i]->type());
    }
}

void MVNLayer::forward(std::vector<Blob *> &inputs, std::vector<Blob> &outputs)
{
    for (size_t inpIdx = 0; inpIdx < inputs.size(); inpIdx++)
    {
        Blob &inpBlob = *inputs[inpIdx];
        Blob &outBlob = outputs[inpIdx];

        int workSize[2];
        int splitDim = (acrossChannels) ? 1 : 2;
        workSize[0] = (int)inpBlob.total(0, splitDim);
        workSize[1] = (int)inpBlob.total(splitDim);

        Mat inpMat = inpBlob.matRef().reshape(1, 2, workSize);
        Mat outMat = outBlob.matRef().reshape(1, 2, workSize);

        Scalar mean, dev;
        for (int i = 0; i < workSize[0]; i++)
        {
            Mat inpRow = inpMat.row(i);
            Mat outRow = outMat.row(i);

            cv::meanStdDev(inpRow, mean, (normalizeVariance) ? dev : noArray());
            double alpha = (normalizeVariance) ? 1/(eps + dev[0]) : 1;
            inpRow.convertTo(outRow, outRow.type(), alpha, -mean[0] * alpha);
        }
    }
}

}
}
