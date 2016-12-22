// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// Copyright (C) 2016, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.

/*
Implementation of shift layer, which adds up const values to blob.
*/

#include "../precomp.hpp"
#include "shift_layer.hpp"
#include "op_blas.hpp"

namespace cv
{
namespace dnn
{

class ShiftLayerImpl {
public:
    static Ptr<ShiftLayerImpl> create(const std::vector<Blob*> &inputs, std::vector<Blob> &outputs,
                                          const std::vector<Blob>& blobs);

    virtual ~ShiftLayerImpl() {}

    virtual void forward(std::vector<Blob*> &inputs, std::vector<Blob> &outputs, const std::vector<Blob>& blobs) = 0;

protected:
    ShiftLayerImpl() {}
    virtual void allocate(const std::vector<Blob*> &inputs, std::vector<Blob> &outputs, const std::vector<Blob>& blobs) = 0;
};

namespace {

class ShiftChannelsLayerImpl : public ShiftLayerImpl {
public:
    virtual void forward(std::vector<Blob*> &inputs, std::vector<Blob> &outputs, const std::vector<Blob>& blobs) {
        for (size_t ii = 0; ii < outputs.size(); ii++)
        {
          Blob &inpBlob = *inputs[ii];
          Blob &outBlob = outputs[ii];

          inpBlob.matRef().copyTo(outBlob.matRef());

          for (int n = 0; n < inpBlob.num(); n++)
          {
            Mat dstMat(inpBlob.channels(), inpBlob.rows() * inpBlob.cols(),
                       outBlob.type(), outBlob.ptr(n));
           dnn::gemm(blobs[0].matRefConst(), biasOnesMat, 1, dstMat, 1); //TODO: gemv
          }
        }
    }

protected:
    virtual void allocate(const std::vector<Blob*> &inputs, std::vector<Blob> &outputs, const std::vector<Blob>& blobs) {
        CV_Assert(inputs.size() > 0);

        const Blob &inpBlob = *inputs[0];
        CV_Assert(inpBlob.dims() == 4 && inpBlob.type() == CV_32F);
        const Blob &biasBlob = blobs[0];
        CV_Assert(biasBlob.total() == (size_t)inpBlob.channels());

        outputs.resize(inputs.size());
        for (size_t i = 0; i < inputs.size(); i++)
        {
            CV_Assert(inputs[i]->type() == inpBlob.type());
            CV_Assert(inputs[i]->dims() == 4 && inputs[i]->channels() == inpBlob.channels());

            outputs[i].shareFrom(*inputs[i]);
        }

        biasOnesMat = Mat::ones(1, inpBlob.rows() * inpBlob.cols(), inpBlob.type());
    }

private:
    Mat biasOnesMat;
};


class ShiftElementsLayerImpl : public ShiftLayerImpl {
public:
    virtual void forward(std::vector<Blob*> &inputs, std::vector<Blob> &outputs, const std::vector<Blob>& blobs) {
        for (size_t ii = 0; ii < outputs.size(); ii++)
        {
          Blob &inpBlob = *inputs[ii];
          Blob &outBlob = outputs[ii];

          outBlob.matRef() = inpBlob.matRef() + blobs[0].matRefConst();
        }
    }

protected:
    virtual void allocate(const std::vector<Blob*> &inputs, std::vector<Blob> &outputs, const std::vector<Blob>& blobs) {
        CV_Assert(inputs.size() > 0);

        const Blob &inpBlob = *inputs[0];
        CV_Assert(inpBlob.type() == CV_32F);
        const Blob &biasBlob = blobs[0];
        CV_Assert(biasBlob.dims() == inpBlob.dims());

        outputs.resize(inputs.size());
        for (size_t i = 0; i < inputs.size(); i++)
        {
            CV_Assert(inputs[i]->type() == inpBlob.type());
            CV_Assert(inputs[i]->dims() == inpBlob.dims());

            outputs[i].shareFrom(*inputs[i]);
        }
    }
};

}

Ptr<ShiftLayerImpl> ShiftLayerImpl::create(const std::vector<Blob*> &inputs, std::vector<Blob> &outputs,
                                      const std::vector<Blob>& blobs) {
    Ptr<ShiftLayerImpl> impl;

    CV_Assert(inputs.size() > 0);
    CV_Assert(blobs.size() > 0);

    if(inputs[0]->dims() == blobs[0].dims())
        impl = Ptr<ShiftLayerImpl>(new ShiftElementsLayerImpl);
    else
        impl = Ptr<ShiftLayerImpl>(new ShiftChannelsLayerImpl);

    impl->allocate(inputs, outputs, blobs);
    return impl;
}

ShiftLayer::ShiftLayer(LayerParams &params) : Layer(params)
{
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

void ShiftLayer::allocate(const std::vector<Blob*> &inputs, std::vector<Blob> &outputs)
{
    impl = ShiftLayerImpl::create(inputs, outputs, blobs);
}

void ShiftLayer::forward(std::vector<Blob*> &inputs, std::vector<Blob> &outputs)
{
    impl->forward(inputs, outputs, blobs);
}

}
}
