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

void BatchNormLayerImpl::allocate(const std::vector<Mat*> &inputs, std::vector<Mat> &outputs)
{
    CV_Assert(blobs.size() >= 2);

    outputs.resize(inputs.size());
    for (size_t i = 0; i < inputs.size(); i++)
    {
        CV_Assert(blobs[0].total() == inputs[i]->size[1]);
        CV_Assert(blobs[1].total() == inputs[i]->size[1]);
        Mat* inp = inputs[i];
        outputs[i].create(inp->dims, &inp->size.p[0], inp->type());
    }
}

void BatchNormLayerImpl::forward(std::vector<Mat*> &inputs, std::vector<Mat> &outputs)
{
    CV_Assert(inputs.size() == 1);

    Mat &inpBlob = *inputs[0];

    int weightsBlobIndex = 2;
    int biasBlobIndex = weightsBlobIndex + hasWeights;

    float varMeanScale = 1;
    if (!hasWeights && !hasBias) {
        varMeanScale = *blobs[2].ptr<float>();
        if (varMeanScale != 0)
            varMeanScale = 1/varMeanScale;
    }

    Mat invStdMat;
    cv::pow(blobs[1]*varMeanScale + epsilon, -0.5, invStdMat);

    int rows = inpBlob.size[2];
    int cols = inpBlob.size[3];

    for (size_t ii = 0; ii < outputs.size(); ii++)
    {
      Mat &outBlob = outputs[ii];

      if (hasWeights)
        CV_Assert(inpBlob.size[1] == blobs[weightsBlobIndex].total());

      if (hasBias)
        CV_Assert(inpBlob.size[1] == blobs[biasBlobIndex].total());

      for(int num = 0; num < outBlob.size[0]; num++)
      {
          for (int n = 0; n < outBlob.size[1]; n++)
          {
              float mean = blobs[0].at<float>(n)*varMeanScale;
              double invstd = invStdMat.at<float>(n);
              float w = hasWeights ? blobs[weightsBlobIndex].at<float>(n) : 1;
              float b = hasBias ? blobs[biasBlobIndex].at<float>(n) : 0;
              Mat inpBlobPlane(rows, cols, CV_32F, inpBlob.ptr<float>(num, n));
              Mat outBlobPlane(rows, cols, CV_32F, outBlob.ptr<float>(num, n));
              inpBlobPlane.convertTo(outBlobPlane, CV_32F, w*invstd, b - mean*w*invstd);
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
