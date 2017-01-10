// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// Copyright (C) 2016, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.

/*
Implementation of Batch Normalization layer.
*/

#include "batch_norm_layer.hpp"

namespace cv
{
namespace dnn
{

BatchNormLayerImpl::BatchNormLayerImpl(float eps_, bool hasWeights_, bool hasBias_):
    eps(eps_),
    hasWeights(hasWeights_),
    hasBias(hasBias_)
{}

void BatchNormLayerImpl::allocate(const std::vector<Blob*> &inputs, std::vector<Blob> &outputs)
{
    CV_Assert(blobs.size() == 4);

    outputs.resize(inputs.size());
    for (size_t i = 0; i < inputs.size(); i++)
    {
        outputs[i].create(inputs[i]->shape());
    }
}

void BatchNormLayerImpl::forward(std::vector<Blob*> &inputs, std::vector<Blob> &outputs)
{
    CV_Assert(inputs.size() == 1);

    Blob &inpBlob = *inputs[0];

    for (size_t ii = 0; ii < outputs.size(); ii++)
    {
      Blob &outBlob = outputs[ii];

      if (hasWeights)
        CV_Assert(inpBlob.channels() == blobs[2].total());

      if (hasBias)
        CV_Assert(inpBlob.channels() == blobs[3].total());

      for (int n = 0; n < inpBlob.channels(); n++)
      {
          float mean = blobs[0].matRefConst().at<float>(n);
          float invstd = 1 / sqrt(blobs[1].matRefConst().at<float>(n) + eps);
          float w = hasWeights ? blobs[2].matRefConst().at<float>(n) : 1;
          float b = hasBias ? blobs[3].matRefConst().at<float>(n) : 0;
          outBlob.getPlane(0, n) = (inpBlob.getPlane(0, n) - mean)*(w*invstd) + b;
      }
    }
}

Ptr<BatchNormLayer> BatchNormLayer::create(float eps, bool has_weights, bool has_bias)
{
    return Ptr<BatchNormLayer>(new BatchNormLayerImpl(eps, has_weights, has_bias));
}

}  // namespace dnn
}  // namespace cv
