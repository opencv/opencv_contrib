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
#include "concat_layer.hpp"
#include <opencv2/core/ocl.hpp>

namespace cv
{
namespace dnn
{

ConcatLayerImpl::ConcatLayerImpl(int axis_ /*= 1*/)
{
    axis = axis_;
}

void ConcatLayerImpl::allocate(const std::vector<Blob *> &inputs, std::vector<Blob> &outputs)
{
    CV_Assert(inputs.size() > 0);

    BlobShape refShape = inputs[0]->shape();
    axisIdx = inputs[0]->canonicalAxis(axis);

    int axisSum = 0;
    useOpenCL = false;
    for (size_t i = 0; i < inputs.size(); i++)
    {
        BlobShape curShape = inputs[i]->shape();

        CV_Assert(curShape.dims() == refShape.dims() && inputs[i]->type() == inputs[0]->type());
        for (int curAxis = 0; curAxis < refShape.dims(); curAxis++)
        {
            if (curAxis != axisIdx && refShape[curAxis] != curShape[curAxis])
                CV_Error(Error::StsBadSize, "Inconsitent shape for ConcatLayer");
        }

        axisSum += curShape[axisIdx];
        useOpenCL |= inputs[i]->getState() == Blob::HEAD_AT_MAT;
    }

    refShape[axisIdx] = axisSum;
    useOpenCL &= ocl::useOpenCL();
    int allocFlags = (useOpenCL) ? Blob::ALLOC_UMAT : Blob::ALLOC_MAT;

    outputs.resize(1);
    outputs[0].create(refShape, inputs[0]->type(), allocFlags);
}


void ConcatLayerImpl::forward(std::vector<Blob *> &inputs, std::vector<Blob> &outputs)
{
    #ifdef HAVE_OPENCL
    if (useOpenCL)
        forward_<UMat>(inputs, outputs);
    else
    #endif
        forward_<Mat>(inputs, outputs);
}

template<typename XMat>
void ConcatLayerImpl::forward_(std::vector<Blob*> &inputs, std::vector<Blob> &outputs)
{
    XMat& outMat = outputs[0].getRef<XMat>();
    std::vector<Range> ranges(outputs[0].dims(), Range::all());

    ranges[axisIdx].start = 0;
    for (size_t i = 0; i < inputs.size(); i++)
    {
        ranges[axisIdx].end = ranges[axisIdx].start + inputs[i]->size(axisIdx);
        inputs[i]->getRefConst<XMat>().copyTo(outMat(&ranges[0]));
        ranges[axisIdx].start = ranges[axisIdx].end;
    }
}

Ptr<ConcatLayer> ConcatLayer::create(int axis)
{
    return Ptr<ConcatLayer>(new ConcatLayerImpl(axis));
}

}
}
