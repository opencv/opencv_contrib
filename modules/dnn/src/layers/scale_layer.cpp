// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// Copyright (C) 2016, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.

/*
Implementation of Scale layer.
*/

#include "../precomp.hpp"
#include "layers_common.hpp"
#include <opencv2/dnn/shape_utils.hpp>

namespace cv
{
namespace dnn
{

class ScaleLayerImpl : public ScaleLayer
{
public:
    ScaleLayerImpl(const LayerParams& params)
    {
        setParamsFrom(params);
        hasBias = params.get<bool>("bias_term", false);
    }

    void allocate(const std::vector<Mat*> &inputs, std::vector<Mat> &outputs)
    {
        CV_Assert(blobs.size() == 1 + hasBias);

        outputs.resize(inputs.size());
        for (size_t i = 0; i < inputs.size(); i++)
        {
            const Mat& inp = *inputs[i];
            outputs[i].create(inp.dims, inp.size.p, inp.type());
        }
    }

    void forward(std::vector<Mat*> &inputs, std::vector<Mat> &outputs)
    {
        for (size_t ii = 0; ii < outputs.size(); ii++)
        {
            Mat &inpBlob = *inputs[ii];
            Mat &outBlob = outputs[ii];

            CV_Assert(inpBlob.size[1] == blobs[0].total());
            if (hasBias)
                CV_Assert(inpBlob.size[1] == blobs[1].total());

            CV_Assert(inpBlob.type() == CV_32F && outBlob.type() == CV_32F);

            for( int cn = 0; cn < inpBlob.size[0]; cn++ )
            {
                for (int n = 0; n < inpBlob.size[1]; n++)
                {
                    float w = blobs[0].at<float>(n);
                    float b = hasBias ? blobs[1].at<float>(n) : 0;
                    Mat outBlobPlane = getPlane(outBlob, cn, n);
                    Mat inpBlobPlane = getPlane(inpBlob, cn, n);
                    inpBlobPlane.convertTo(outBlobPlane, CV_32F, w, b);
                }
            }
        }
    }

    bool hasBias;
};


Ptr<ScaleLayer> ScaleLayer::create(const LayerParams& params)
{
    return Ptr<ScaleLayer>(new ScaleLayerImpl(params));
}

}  // namespace dnn
}  // namespace cv
