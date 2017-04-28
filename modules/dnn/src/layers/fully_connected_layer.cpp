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
#include "op_blas.hpp"
#include <opencv2/dnn/shape_utils.hpp>

namespace cv
{
namespace dnn
{

class FullyConnectedLayerImpl : public InnerProductLayer
{
public:
    FullyConnectedLayerImpl(const LayerParams& params)
    {
        setParamsFrom(params);
        CV_Assert(1 <= blobs.size() && blobs.size() <= 2);

        int numOutput = params.get<int>("num_output");
        int innerSize = (int)blobs[0].total() / numOutput;
        bias = params.get<bool>("bias_term", true);
        axis = params.get<int>("axis", 1);

        CV_Assert(blobs[0].dims >= 2 && (size_t)(innerSize * numOutput) == blobs[0].total());
        CV_Assert(!bias || (blobs.size() == 2 && (size_t)numOutput == blobs[1].total()));

        blobs[0] = blobs[0].reshape(1, numOutput);
        if (bias)
            blobs[1] = blobs[1].reshape(1, 1);
    }

    bool getMemoryShapes(const std::vector<MatShape> &inputs,
                         const int requiredOutputs,
                         std::vector<MatShape> &outputs,
                         std::vector<MatShape> &internals) const
    {
        CV_Assert(inputs.size() > 0);
        CV_Assert(1 <= blobs.size() && blobs.size() <= 2);
        CV_Assert(blobs[0].dims == 2);

        int cAxis = clamp(axis, inputs[0]);
        int outerSize = total(inputs[0], 0, cAxis);
        int numOutput = blobs[0].size[0];
        outputs.resize(inputs.size(), shape(outerSize, numOutput));

        internals.push_back(shape(outerSize, 1));

        CV_Assert(!bias || (size_t)numOutput == blobs[1].total());

        return false;
    }

    void forward(std::vector<Mat*> &input, std::vector<Mat> &output, std::vector<Mat> &internals)
    {
        internals[0].setTo(1.);
        const Mat &weight = blobs[0];
        const Mat *biasMat = NULL, *biasOnesMat = NULL;

        int axisCan = clamp(axis, input[0]->dims);
        int outerSize = input[0]->total(0, axisCan);

        if (bias)
        {
            biasOnesMat = &internals[0];
            biasMat = &blobs[1];
        }

        for (size_t i = 0; i < input.size(); i++)
        {
            Mat srcMat = input[i]->reshape(1, outerSize);
            Mat dstMat = output[i].reshape(1, outerSize);
            dnn::gemm(srcMat, weight, 1, dstMat, 0, GEMM_2_T);

            if (bias)
                dnn::gemm(*biasOnesMat, *biasMat, 1, dstMat, 1);
        }
    }

    virtual int64 getFLOPS(const std::vector<MatShape> &inputs,
                           const std::vector<MatShape> &outputs) const
    {
        (void)inputs; // suppress unused variable warning
        long flops = 0;

        int innerSize = blobs[0].size[1];
        for(int i = 0; i < outputs.size(); i++)
        {
            flops += 3*innerSize*total(outputs[i]);
        }

        return flops;

    }

    bool bias;
};

Ptr<InnerProductLayer> InnerProductLayer::create(const LayerParams& params)
{
    return Ptr<InnerProductLayer>(new FullyConnectedLayerImpl(params));
}

}
}
