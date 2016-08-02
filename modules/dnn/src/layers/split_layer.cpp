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
#include "split_layer.hpp"
#include <opencv2/core/ocl.hpp>

namespace cv
{
namespace dnn
{

SplitLayerImpl::SplitLayerImpl(int outputsCount_ /*= -1*/)
{
    outputsCount = outputsCount_;
}

void SplitLayerImpl::allocate(const std::vector<Blob*> &inputs, std::vector<Blob> &outputs)
{
    CV_Assert(inputs.size() == 1);
    useOpenCL = ocl::useOpenCL() && inputs[0]->getState() == Blob::HEAD_AT_UMAT;
    int allocFlags = useOpenCL ? Blob::ALLOC_UMAT : Blob::ALLOC_MAT;

    if (outputsCount >= 0)
        outputs.resize(outputsCount);

    for (size_t i = 0; i < outputs.size(); i++)
        outputs[i].create(inputs[0]->shape(), inputs[0]->type(), allocFlags);
}

void SplitLayerImpl::forward(std::vector<Blob*> &inputs, std::vector<Blob> &outputs)
{
    for (size_t i = 0; i < outputs.size(); i++)
    {
        if (useOpenCL)
            inputs[0]->umatRefConst().copyTo(outputs[i].umatRef());
        else
            inputs[0]->matRefConst().copyTo(outputs[i].matRef());
    }
}


Ptr<SplitLayer> SplitLayer::create(int outputsCount)
{
    return Ptr<SplitLayer>(new SplitLayerImpl(outputsCount));
}

}
}
