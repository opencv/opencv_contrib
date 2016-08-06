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
        start_axis = start_axis_;
        offset = offset_;
    }

    void CropLayerImpl::allocate(const std::vector<Blob *> &inputs, std::vector<Blob> &outputs)
    {
        CV_Assert(2 == inputs.size());

        const Blob &inpBlob = *inputs[0];
        CV_Assert(inpBlob.dims() == 4 && inpBlob.type() == CV_32F);

        const Blob &inpSzBlob = *inputs[1];

        outSizes.resize(4, 0);
        for (int i = 0; i < 4; i++)
        {
            if (i < start_axis)
                outSizes[i] = inpBlob.size(i);
            else
                outSizes[i] = inpSzBlob.size(i);
            if (offset[i] + outSizes[i] > inpBlob.size(i))
                CV_Error(Error::StsBadArg, "invalid crop parameters");
        }

        outputs.resize(1);
        outputs[0].create(BlobShape(outSizes));
    }

    void CropLayerImpl::forward(std::vector<Blob *> &inputs, std::vector<Blob> &outputs)
    {
        Blob input = *inputs[0];
        Blob output = outputs[0];
        for (int num = 0; num < outSizes[0]; ++num)
        {
            for (int ch = 0; ch < outSizes[1]; ++ch)
            {
                for (int row = 0; row < outSizes[2]; ++row)
                {
                    float *srcData = input.ptrf(num + offset[0], ch + offset[1], row + offset[2]);
                    float *dstData = output.ptrf(num, ch, row);
                    memcpy(dstData, srcData + offset[3], sizeof(float) * outSizes[3]);
                }
            }
        }
    }

    Ptr<CropLayer> CropLayer::create(int start_axis, const std::vector<int> &offset)
    {
        return Ptr<CropLayer>(new CropLayerImpl(start_axis, offset));
    }
}
}
