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

BatchNormLayerImpl::BatchNormLayerImpl(bool hasWeights_, bool hasBias_, float epsilon_):
    hasWeights(hasWeights_),
    hasBias(hasBias_),
    epsilon(epsilon_)
{}

void BatchNormLayerImpl::allocate(const std::vector<Blob*> &inputs, std::vector<Blob> &outputs)
{
    CV_Assert(blobs.size() >= 2);

    outputs.resize(inputs.size());
    for (size_t i = 0; i < inputs.size(); i++)
    {
        CV_Assert(blobs[0].total() == inputs[i]->channels());
        CV_Assert(blobs[1].total() == inputs[i]->channels());
        outputs[i].create(inputs[i]->shape());
    }
}

void BatchNormLayerImpl::forward(std::vector<Blob*> &inputs, std::vector<Blob> &outputs)
{
    CV_Assert(inputs.size() == 1);

    Blob &inpBlob = *inputs[0];

    int weightsBlobIndex = 2;
    int biasBlobIndex = weightsBlobIndex + hasWeights;

    float varMeanScale = 1;
    if (!hasWeights && !hasBias) {
        varMeanScale = *blobs[2].ptrf();
        if (varMeanScale != 0)
            varMeanScale = 1/varMeanScale;
    }

    Mat invStdMat;
    cv::pow(blobs[1].matRefConst()*varMeanScale + epsilon, -0.5, invStdMat);

    for (size_t ii = 0; ii < outputs.size(); ii++)
    {
      Blob &outBlob = outputs[ii];

      if (hasWeights)
        CV_Assert(inpBlob.channels() == blobs[weightsBlobIndex].total());

      if (hasBias)
        CV_Assert(inpBlob.channels() == blobs[biasBlobIndex].total());

      for(int num = 0; num < outBlob.num(); num++)
      {
          for (int n = 0; n < outBlob.channels(); n++)
          {
              float mean = blobs[0].matRefConst().at<float>(n)*varMeanScale;
              double invstd = invStdMat.at<float>(n);
              float w = hasWeights ? blobs[weightsBlobIndex].matRefConst().at<float>(n) : 1;
              float b = hasBias ? blobs[biasBlobIndex].matRefConst().at<float>(n) : 0;
              outBlob.getPlane(num, n) = (inpBlob.getPlane(num, n) - mean)*w*invstd + b;
          }
      }
    }
}

Ptr<BatchNormLayer> BatchNormLayer::create(bool hasWeights, bool hasBias, float epsilon)
{
    return Ptr<BatchNormLayer>(new BatchNormLayerImpl(hasWeights, hasBias, epsilon));
}

}  // namespace dnn
}  // namespace cv
