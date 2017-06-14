// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// Copyright (C) 2016, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.

/*
Implementation of Batch Normalization layer.
*/

#include "../precomp.hpp"
#include "op_halide.hpp"
#include <opencv2/dnn/shape_utils.hpp>

namespace cv
{
namespace dnn
{

class BatchNormLayerImpl : public BatchNormLayer
{
public:
    BatchNormLayerImpl(const LayerParams& params)
    {
        setParamsFrom(params);
        CV_Assert(blobs.size() >= 3);

        hasWeights = params.get<bool>("has_weight", false);
        hasBias = params.get<bool>("has_bias", false);
        epsilon = params.get<float>("eps", 1E-5);
    }

    bool getMemoryShapes(const std::vector<MatShape> &inputs,
                         const int requiredOutputs,
                         std::vector<MatShape> &outputs,
                         std::vector<MatShape> &internals) const
    {
        Layer::getMemoryShapes(inputs, requiredOutputs, outputs, internals);
        return true;
    }

    virtual bool supportBackend(int backendId)
    {
        return backendId == DNN_BACKEND_DEFAULT ||
               backendId == DNN_BACKEND_HALIDE && haveHalide();
    }

    void forward(std::vector<Mat*> &inputs, std::vector<Mat> &outputs, std::vector<Mat> &internals)
    {
        CV_Assert(blobs.size() >= 2);
        CV_Assert(inputs.size() == 1);

        float varMeanScale = 1.f;
        if (!hasWeights && !hasBias) {
            varMeanScale = *blobs[2].ptr<float>();
            if (varMeanScale != 0)
                varMeanScale = 1/varMeanScale;
        }

        Mat invStdMat;
        cv::pow(blobs[1]*varMeanScale + epsilon, -0.5, invStdMat);

        Mat &inpBlob = *inputs[0];

        int weightsBlobIndex = 2;
        int biasBlobIndex = weightsBlobIndex + hasWeights;

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

    virtual Ptr<BackendNode> tryAttach(const Ptr<BackendNode>& node)
    {
        switch (node->backendId)
        {
            case DNN_BACKEND_HALIDE:
            {
#ifdef HAVE_HALIDE
                auto base = node.dynamicCast<HalideBackendNode>();
                Halide::Func& input = base->funcs.back();
                Halide::Var x("x"), y("y"), c("c"), n("n");
                Halide::Func top = attachHalide(input(x, y, c, n));
                return Ptr<BackendNode>(new HalideBackendNode(base, top));
#endif  // HAVE_HALIDE
                break;
            }
        }
        return Ptr<BackendNode>();
    }

    virtual Ptr<BackendNode> initHalide(const std::vector<Ptr<BackendWrapper> > &inputs)
    {
#ifdef HAVE_HALIDE
        Halide::Buffer<float> input = halideBuffer(inputs[0]);
        Halide::Var x("x"), y("y"), c("c"), n("n");
        Halide::Func top = attachHalide(input(x, y, c, n));
        return Ptr<BackendNode>(new HalideBackendNode(top));
#endif  // HAVE_HALIDE
        return Ptr<BackendNode>();
    }

#ifdef HAVE_HALIDE
    // attachHalide can work both with Halide::Buffer and Halide::Func. In the
    // second case it will be a fusion.
    Halide::Func attachHalide(const Halide::Expr& input)
    {
        Halide::Func top = (name.empty() ? Halide::Func() : Halide::Func(name));
        Halide::Var x("x"), y("y"), c("c"), n("n");

        const int weightsBlobIndex = 2;
        const int biasBlobIndex = weightsBlobIndex + hasWeights;
        const int numChannels = blobs[0].total();
        float* meanData = (float*)blobs[0].data;
        float* stdData = (float*)blobs[1].data;
        float* weightsData = (hasWeights ? (float*)blobs[weightsBlobIndex].data : NULL);
        float* biasData = (hasBias ? (float*)blobs[biasBlobIndex].data : NULL);

        float varMeanScale = 1.f;
        if (!hasWeights && !hasBias) {
            varMeanScale = *blobs[2].ptr<float>();
            if (varMeanScale != 0)
                varMeanScale = 1/varMeanScale;
        }

        Halide::Buffer<float> weights(numChannels);
        Halide::Buffer<float> bias(numChannels);
        for (int i = 0; i < numChannels; ++i)
        {
            weights(i) = (hasWeights ? weightsData[i] : 1.0f) /
                         sqrt(stdData[i] * varMeanScale + epsilon);
            bias(i) = (hasBias ? biasData[i] : 0.0f) -
                      weights(i) * meanData[i] * varMeanScale;
        }
        top(x, y, c, n) = input * weights(c) + bias(c);
        return top;
    }
#endif  // HAVE_HALIDE

    virtual int64 getFLOPS(const std::vector<MatShape> &inputs,
                           const std::vector<MatShape> &outputs) const
    {
        (void)outputs; // suppress unused variable warning

        int64 flops = 0;
        for(int i = 0; i < inputs.size(); i++)
        {
            flops += 3*total(inputs[i]);
        }
        return flops;
    }
};

Ptr<BatchNormLayer> BatchNormLayer::create(const LayerParams& params)
{
    return Ptr<BatchNormLayer>(new BatchNormLayerImpl(params));
}

}  // namespace dnn
}  // namespace cv
