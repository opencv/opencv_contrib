// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// Copyright (C) 2016, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.

/*
Implementation of shift layer, which adds up const values to blob.
*/

#include "../precomp.hpp"
#include "op_blas.hpp"

namespace cv
{
namespace dnn
{

class ShiftLayerImpl : public ShiftLayer
{
public:
    ShiftLayerImpl(const LayerParams &params)
    {
        setParamsFrom(params);
        CV_Assert(blobs.size() == 1);

#ifdef HAVE_LAPACK
        {
            if (getBlasThreads() != cv::getThreadNum())
            {
                setBlasThreads(cv::getThreadNum());
            }
        }
#endif
    }

    virtual void allocate(const std::vector<Mat*> &inputs, std::vector<Mat> &outputs)
    {
        CV_Assert(inputs.size() > 0);
        CV_Assert(blobs.size() > 0);
        const Mat &inpBlob = *inputs[0];
        CV_Assert(inpBlob.dims == 4 && inpBlob.type() == CV_32F);
        const Mat &biasBlob = blobs[0];
        outputs.resize(inputs.size());

        if(inpBlob.dims == biasBlob.dims)
        {
            for (size_t i = 0; i < inputs.size(); i++)
            {
                CV_Assert(inputs[i]->type() == inpBlob.type());
                CV_Assert(inputs[i]->dims == inpBlob.dims);

                outputs[i] = *inputs[i];
            }
        }
        else
        {
            CV_Assert(biasBlob.total() == (size_t)inpBlob.size[1]);

            for (size_t i = 0; i < inputs.size(); i++)
            {
                CV_Assert(inputs[i]->type() == inpBlob.type());
                CV_Assert(inputs[i]->dims == 4 && inputs[i]->size[1] == inpBlob.size[1]);

                outputs[i] = *inputs[i];
            }

            biasOnesMat = Mat::ones(1, inpBlob.size[2] * inpBlob.size[3], inpBlob.type());
        }
    }

    virtual void forward(std::vector<Mat*> &inputs, std::vector<Mat> &outputs)
    {
        CV_Assert(inputs.size() > 0);
        CV_Assert(blobs.size() > 0);

        if(inputs[0]->dims == blobs[0].dims)
        {
            for (size_t ii = 0; ii < outputs.size(); ii++)
            {
                Mat &inpBlob = *inputs[ii];
                Mat &outBlob = outputs[ii];

                outBlob = inpBlob + blobs[0];
            }
        }
        else
        {
            for (size_t ii = 0; ii < outputs.size(); ii++)
            {
                Mat &inpBlob = *inputs[ii];
                Mat &outBlob = outputs[ii];

                inpBlob.copyTo(outBlob);

                for (int n = 0; n < inpBlob.size[0]; n++)
                {
                    Mat dstMat(inpBlob.size[1], inpBlob.size[2] * inpBlob.size[3],
                               outBlob.type(), outBlob.ptr(n));
                    dnn::gemm(blobs[0], biasOnesMat, 1, dstMat, 1); //TODO: gemv
                }
            }
        }
    }

    Mat biasOnesMat;
};

Ptr<ShiftLayer> ShiftLayer::create(const LayerParams& params)
{
    return Ptr<ShiftLayer>(new ShiftLayerImpl(params));
}

}
}
