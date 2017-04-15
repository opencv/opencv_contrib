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
#include "crop_layer.hpp"

namespace cv
{
namespace dnn
{

CropLayerImpl::CropLayerImpl(int start_axis_, const std::vector<int> &offset_)
{
    startAxis = start_axis_;
    offset = offset_;
}

void CropLayerImpl::allocate(const std::vector<Blob *> &inputs, std::vector<Blob> &outputs)
{
    CV_Assert(2 == inputs.size());

    const Blob &inpBlob = *inputs[0];
    const Blob &inpSzBlob = *inputs[1];

    int start_axis = inpBlob.canonicalAxis(startAxis);
    int dims = inpBlob.dims();

    std::vector<int> offset_final(dims, 0);
    if (offset.size() == 1)
    {
        for (int i = start_axis; i < dims; i++)
            offset_final[i] = offset[0];
    }
    else if (offset.size() > 1)
    {
        if ((int)offset.size() != dims - start_axis)
            CV_Error(Error::StsBadArg, "number of offset values specified must be equal to the number of dimensions following axis.");

        for (int i = start_axis; i < dims; i++)
            offset_final[i] = offset[i - start_axis];
    }

    BlobShape dstShape = inpBlob.shape();
    crop_ranges.resize(dims, Range::all());
    for (int i = start_axis; i < dims; i++)
    {
        dstShape[i] = inpSzBlob.size(i);

        if (!offset.empty()) //normal case
        {
            if (offset_final[i] < 0 || offset_final[i] + inpSzBlob.size(i) > inpBlob.size(i))
                CV_Error(Error::StsBadArg, "invalid crop parameters");

            crop_ranges[i] = Range(offset_final[i], offset_final[i] + inpSzBlob.size(i));
        }
        else //detect offset automatically so that cropped image is center of original one
        {
            if (inpSzBlob.size(i) > inpBlob.size(i))
                CV_Error(Error::StsBadArg, "invalid output blob size");

            int cur_crop = (inpBlob.size(i) - inpSzBlob.size(i)) / 2;
            crop_ranges[i] = Range(cur_crop, cur_crop + inpSzBlob.size(i));
        }
    }

    outputs.resize(1);
    outputs[0].create(dstShape);
}

void CropLayerImpl::forward(std::vector<Blob *> &inputs, std::vector<Blob> &outputs)
{
    Blob &input = *inputs[0];
    Blob &output = outputs[0];

    #ifdef HAVE_OPENCL
    if (input.getState() == Blob::HEAD_AT_UMAT)
        input.umatRefConst()(&crop_ranges[0]).copyTo(output.umatRef());
    else
    #endif
        input.matRefConst()(&crop_ranges[0]).copyTo(output.matRef());
}

Ptr<CropLayer> CropLayer::create(int start_axis, const std::vector<int> &offset)
{
    return Ptr<CropLayer>(new CropLayerImpl(start_axis, offset));
}

}
}
