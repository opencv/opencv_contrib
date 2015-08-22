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
#include "reshape_layer.hpp"

namespace cv
{
namespace dnn
{

ReshapeLayer::ReshapeLayer(LayerParams &params) : Layer(params)
{
    inAxis = params.get<int>("axis", 0);
    inNumAxes = params.get<int>("num_axes", -1);
    CV_Assert(inNumAxes >= -1);

    autoAxisIdx = -1;

    if (!params.has("dim"))
    {
        shapeDesc = BlobShape(0);
        return;
    }

    DictValue paramShape = params.get("dim");
    shapeDesc = BlobShape(paramShape.size());

    for (int i = 0; i < paramShape.size(); i++)
    {
        int dim = paramShape.get<int>(i);
        CV_Assert(dim >= -1);

        if (dim == -1)
        {
            if (autoAxisIdx != -1)
                CV_Error(Error::StsBadArg, "New shape contains multiple -1 dims");
            autoAxisIdx = i;
        }

        shapeDesc[i] = dim;
    }
}

void ReshapeLayer::allocate(const std::vector<Blob*> &inputs, std::vector<Blob> &outputs)
{
    outputs.resize(inputs.size());

    for (size_t i = 0; i < inputs.size(); i++)
    {
        Blob &inpBlob = *inputs[i];
        Blob &outBlob = outputs[i];
        BlobShape inpShape = inpBlob.shape();

        int startAxis = (inAxis >= 0) ? inAxis : inpShape.dims() + 1 + inAxis;
        int endAxis = (inNumAxes == -1) ? inpShape.dims() : startAxis + inNumAxes;
        CV_Assert(0 <= startAxis && startAxis <= inpShape.dims());
        CV_Assert(0 <= endAxis && endAxis <= inpShape.dims());

        int newDims = inpShape.dims() - (endAxis - startAxis) + shapeDesc.dims();
        BlobShape outShape(newDims);

        computeOutputShape(startAxis, endAxis, inpShape, outShape);

        outBlob.shareFrom(inpBlob);
        outBlob.reshape(outShape);
    }
}

void ReshapeLayer::computeOutputShape(int startAxis, int endAxis, BlobShape &inpShape, BlobShape &outShape)
{
    int idx = 0;
    for (int i = 0; i < startAxis; i++)
        outShape[idx++] = inpShape[i];

    for (int i = 0; i < shapeDesc.dims(); i++)
    {
        if (shapeDesc[i] == 0)
        {
            int inpAxisIdx = startAxis + i;
            if (inpAxisIdx < 0 || inpShape.dims() <= inpAxisIdx)
                CV_Error(Error::StsOutOfRange, "copy dimension (which has zero size) is not presented into reshaped blob");
            outShape[idx++] = inpShape[startAxis + i];
        }
        else
        {
            outShape[idx++] = (shapeDesc[i] > 0) ? shapeDesc[i] : 1;
        }
    }

    for (int i = endAxis; i < inpShape.dims(); i++)
        outShape[idx++] = inpShape[i];

    if (autoAxisIdx >= 0)
    {
        size_t total = inpShape.total();
        size_t curTotal = 1;
        for (int i = 0; i < outShape.dims(); i++)
        {
            if (i != startAxis + autoAxisIdx)
                curTotal *= outShape[i];
        }

        CV_DbgAssert(curTotal <= total && total % curTotal == 0);

        outShape[startAxis + autoAxisIdx] = (int)(total / curTotal);
    }

    if (inpShape.total() != outShape.total())
    {
        CV_Error(Error::StsUnmatchedSizes, "Mismatch between input and output blob elements count");
    }
}


Ptr<Layer> createFlattenLayer(LayerParams&)
{
    LayerParams params;

    int shapeDesc[] = {0, -1};
    params.set("dim", DictValue::arrayInt(shapeDesc, 2));

    return Ptr<Layer>(new ReshapeLayer(params));
}

}
}
