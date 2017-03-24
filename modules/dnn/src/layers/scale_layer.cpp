// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// Copyright (C) 2016, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.

/*
Implementation of Scale layer.
*/

#include "scale_layer.hpp"

namespace cv
{
namespace dnn
{

void ScaleLayerImpl::allocate(const std::vector<Blob*> &inputs, std::vector<Blob> &outputs)
{
    CV_Assert(blobs.size() == 1 + hasBias);

    outputs.resize(inputs.size());
    for (size_t i = 0; i < inputs.size(); i++)
    {
        outputs[i].create(inputs[i]->shape());
    }
}

void ScaleLayerImpl::forward(std::vector<Blob*> &inputs, std::vector<Blob> &outputs)
{
    CV_Assert(inputs.size() == 1);

    Blob &inpBlob = *inputs[0];

    for (size_t ii = 0; ii < outputs.size(); ii++)
    {
      Blob &outBlob = outputs[ii];

      CV_Assert(inpBlob.channels() == blobs[0].total());

      if (hasBias)
        CV_Assert(inpBlob.channels() == blobs[1].total());

      for (int n = 0; n < inpBlob.channels(); n++)
      {
          float w = blobs[0].matRefConst().at<float>(n);
          float b = hasBias ? blobs[1].matRefConst().at<float>(n) : 0;
          outBlob.getPlane(0, n) = w*inpBlob.getPlane(0, n) + b;
      }
    }
}

Ptr<ScaleLayer> ScaleLayer::create(bool hasBias)
{
    return Ptr<ScaleLayer>(new ScaleLayerImpl(hasBias));
}

}  // namespace dnn
}  // namespace cv
