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

namespace cv
{
namespace dnn
{
    FullyConnectedLayer::FullyConnectedLayer(LayerParams &params) : Layer(params)
    {
        numOutputs = params.get<int>("num_output");
        bias = params.get<bool>("bias_term", true);
        axis_ = params.get<int>("axis", 1);

        CV_Assert(blobs.size() == (bias ? 2U : 1U));
        CV_Assert(blobs[0].dims() >= 2 && blobs[0].total() >= (size_t)numOutputs);
        CV_Assert(!bias || blobs[1].total() == (size_t)numOutputs);
    }

    void FullyConnectedLayer::allocate(const std::vector<Blob*> &input, std::vector<Blob> &output)
    {
        CV_Assert(input.size() > 0);

        axis = input[0]->canonicalAxis(axis_);
        innerSize = (int)input[0]->total(axis);

        CV_Assert((size_t)innerSize * (size_t)numOutputs == blobs[0].total());
        CV_Assert(blobs[0].size(-2) == numOutputs && blobs[0].size(-1) == innerSize);

        output.resize(input.size());
        for (size_t i = 0; i < input.size(); i++)
        {
            if (i != 0)
                CV_Assert(input[i]->equalShape(*input[0]));

            this->reshape(*input[i], output[i]);
        }
    }

    void FullyConnectedLayer::reshape(const Blob &inp, Blob &out)
    {
        BlobShape inpShape = inp.shape();
        BlobShape outShape(axis+1, inpShape.ptr());
        outShape[axis] = numOutputs;

        out.create(outShape, inp.type());
    }

    void FullyConnectedLayer::forward(std::vector<Blob*> &input, std::vector<Blob> &output)
    {
        for (size_t i = 0; i < input.size(); i++)
        {
            int M = (int)input[i]->total(0, axis);
            int N = numOutputs;
            int K = innerSize;

            Mat srcMat(M, K, input[i]->type(), input[i]->ptrf());
            Mat weight(N, K, blobs[0].type(), blobs[0].ptrf());
            Mat dstMat(M, N, output[i].type(), output[i].ptrf());

            //important: Caffe stores weights as transposed array
            cv::gemm(srcMat, weight, 1, noArray(), 0, dstMat, GEMM_2_T);

            if (bias)
            {
                Mat biasOnesMat = Mat::ones(M, 1, CV_32F);
                Mat biasMat(1, N, CV_32F, blobs[1].ptrf());
                cv::gemm(biasOnesMat, biasMat, 1, dstMat, 1, dstMat);
            }
        }
    }
}
}
