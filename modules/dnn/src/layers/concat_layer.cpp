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

namespace cv
{
namespace dnn
{
    ConcatLayer::ConcatLayer(LayerParams &params) : Layer(params)
    {
        axis = params.get<int>("axis", 1);
        CV_Assert(axis >= 0);
    }

    void ConcatLayer::allocate(const std::vector<Blob *> &inputs, std::vector<Blob> &outputs)
    {
        CV_Assert(inputs.size() > 0);

        int refType = inputs[0]->type();
        BlobShape refShape = inputs[0]->shape();
        CV_Assert(axis < refShape.dims());

        int axisSum = 0;
        for (size_t i = 0; i < inputs.size(); i++)
        {
            BlobShape curShape = inputs[i]->shape();

            CV_Assert(curShape.dims() == refShape.dims() && inputs[i]->type() == refType);
            for (int axisId = 0; axisId < refShape.dims(); axisId++)
            {
                if (axisId != axis && refShape[axisId] != curShape[axisId])
                    CV_Error(Error::StsBadSize, "Inconsitent shape for ConcatLayer");
            }

            axisSum += curShape[axis];
        }

        refShape[axis] = axisSum;
        outputs.resize(1);
        outputs[0].create(refShape);
    }

    void ConcatLayer::forward(std::vector<Blob *> &inputs, std::vector<Blob> &outputs)
    {
        const Mat& outMat = outputs[0].matRef();
        std::vector<Range> ranges(outputs[0].dims(), Range::all());
        int sizeStart = 0;
        for (size_t i = 0; i < inputs.size(); i++)
        {
            int sizeEnd = sizeStart + inputs[i]->size(axis);
            ranges[axis] = Range(sizeStart, sizeEnd);

            Mat outSubMat = outMat(&ranges[0]);
            inputs[i]->matRef().copyTo(outSubMat);

            sizeStart = sizeEnd;
        }
    }
}
}
