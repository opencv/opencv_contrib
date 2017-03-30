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
#include "fully_connected_layer.hpp"
#include "op_blas.hpp"
#include <opencv2/dnn/shape_utils.hpp>
#include <opencv2/core/ocl.hpp>

namespace cv
{
namespace dnn
{

FullyConnectedLayerImpl::FullyConnectedLayerImpl(int axis_)
{
    axis = axis_;
}

void FullyConnectedLayerImpl::allocate(const std::vector<Blob*> &input, std::vector<Blob> &output)
{
    CV_Assert(input.size() > 0);
    CV_Assert(1 <= blobs.size() && blobs.size() <= 2);
    CV_Assert(blobs[0].dims() == 2);

    bias = (blobs.size() >= 1);
    axisCan = input[0]->canonicalAxis(axis);
    dtype = input[0]->type();
    numOutput = blobs[0].size(0);
    innerSize = blobs[0].size(1);
    outerSize = input[0]->total(0, axisCan);

    CV_Assert((size_t)innerSize == input[0]->total(axisCan));
    CV_Assert(!bias || (size_t)numOutput == blobs[1].total());

    biasOnesBlob.create(Shape(outerSize, 1), dtype);
    biasOnesBlob.setTo(1);

    output.resize(input.size());
    for (size_t i = 0; i < input.size(); i++)
    {
        CV_Assert(i == 0 || (input[i]->equalShape(*input[0]) && input[i]->type() == dtype));
        Shape outShape = Shape(outerSize, numOutput);
        output[i].create(outShape, dtype);
    }
}

void FullyConnectedLayerImpl::forward(std::vector<Blob*> &input, std::vector<Blob> &output)
{
    const Mat &weight = blobs[0].matRefConst();
    const Mat *biasMat = NULL, *biasOnesMat = NULL;
    if (bias)
    {
        biasOnesMat = &biasOnesBlob.matRefConst();
        biasMat = &blobs[1].matRefConst();
    }

    for (size_t i = 0; i < input.size(); i++)
    {
        Mat srcMat = reshaped(input[i]->matRefConst(), Shape(outerSize, innerSize));
        Mat dstMat = reshaped(output[i].matRef(), Shape(outerSize, numOutput));
        dnn::gemm(srcMat, weight, 1, dstMat, 0, GEMM_2_T);

        if (bias)
            dnn::gemm(*biasOnesMat, *biasMat, 1, dstMat, 1);
    }
}


Ptr<InnerProductLayer> InnerProductLayer::create(int axis)
{
    return Ptr<InnerProductLayer>(new FullyConnectedLayerImpl(axis));
}

}
}
