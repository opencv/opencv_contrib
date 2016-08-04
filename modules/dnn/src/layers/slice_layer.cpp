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
#include "slice_layer.hpp"
#include <opencv2/core/ocl.hpp>
#include <opencv2/dnn/shape_utils.hpp>

namespace cv
{
namespace dnn
{

SliceLayerImpl::SliceLayerImpl(int axis_ /*= 1*/)
{
    axis = axis_;
}

SliceLayerImpl::SliceLayerImpl(int axis_, const std::vector<int> &sliceIndices_)
{
    axis = axis_;
    sliceIndices = sliceIndices_;
}

void SliceLayerImpl::allocate(const std::vector<Blob*> &inputs, std::vector<Blob> &outputs)
{
    CV_Assert(inputs.size() == 1);

    const Blob &inpBlob = *inputs[0];
    useOpenCL = ocl::useOpenCL() && inpBlob.getState() == Blob::HEAD_AT_UMAT;

    axisIdx = inpBlob.canonicalAxis(axis);
    int axisSize = inpBlob.size(axisIdx);
    BlobShape inpShape = inpBlob.shape();
    int allocFlags = useOpenCL ? Blob::ALLOC_UMAT : Blob::ALLOC_MAT;

    if (sliceIndices.size()) //divide blob with respect to passed parameters
    {
        std::vector<int> outAxisSize;
        int prevSlice = 0;

        for (size_t i = 0; i < sliceIndices.size(); i++)
        {
            if (!(prevSlice < sliceIndices[i] && sliceIndices[i] < axisSize))
                CV_Error(Error::StsBadArg, "Slice indices should be positive, increased and don't exceed size of sliced dimension");

            outAxisSize.push_back(sliceIndices[i] - prevSlice);
            prevSlice = sliceIndices[i];
        }
        outAxisSize.push_back(axisSize - prevSlice);

        outputs.resize(outAxisSize.size());
        for (size_t i = 0; i < outAxisSize.size(); i++)
        {
            inpShape[axisIdx] = outAxisSize[i];
            outputs[i].create(inpShape, inpBlob.type(), allocFlags);
        }
    }
    else //divide blob with respect to count of output blobs
    {
        CV_Assert(outputs.size() > 0 && axisSize % outputs.size() == 0);
        int outAxisSize = axisSize / (int)outputs.size();

        for (size_t i = 0; i < outputs.size(); i++)
        {
            inpShape[axisIdx] = outAxisSize;
            outputs[i].create(inpShape, inpBlob.type(), allocFlags);
        }
    }
}

void SliceLayerImpl::forward(std::vector<Blob*> &inputs, std::vector<Blob> &outputs)
{
    #ifdef HAVE_OPENCL
    if (useOpenCL)
        forward_<UMat>(inputs, outputs);
    else
    #endif
        forward_<Mat>(inputs, outputs);
}

template<typename XMat>
void SliceLayerImpl::forward_(std::vector<Blob*> &inputs, std::vector<Blob> &outputs)
{
    const XMat& inpMat = inputs[0]->getRefConst<XMat>();
    std::vector<Range> ranges(inputs[0]->dims(), Range::all());

    ranges[axisIdx].start = 0;
    for (size_t i = 0; i < outputs.size(); i++)
    {
        ranges[axisIdx].end = ranges[axisIdx].start + outputs[i].size(axisIdx);
        inpMat(&ranges[0]).copyTo(outputs[i].getRef<XMat>());
        ranges[axisIdx].start = ranges[axisIdx].end;
    }
}

Ptr<SliceLayer> SliceLayer::create(int axis)
{
    return Ptr<SliceLayer>(new SliceLayerImpl(axis));
}

Ptr<SliceLayer> SliceLayer::create(int axis, const std::vector<int> &sliceIndices)
{
    return Ptr<SliceLayer>(new SliceLayerImpl(axis, sliceIndices));
}

}
}
