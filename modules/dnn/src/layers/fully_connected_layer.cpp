#include "../precomp.hpp"
#include "layers_common.hpp"
#include <iostream>

namespace cv
{
namespace dnn
{
    class FullyConnectedLayer : public Layer
    {
        bool bias;
        int numOutputs;
        int axis;

        int innerSize;

        void reshape(const Blob &inp, Blob &out);

    public:
        FullyConnectedLayer(LayerParams &params);
        void allocate(const std::vector<Blob*> &inputs, std::vector<Blob> &outputs);
        void forward(std::vector<Blob*> &inputs, std::vector<Blob> &outputs);
    };


    REGISTER_LAYER_CLASS(InnerProduct, FullyConnectedLayer)


    FullyConnectedLayer::FullyConnectedLayer(LayerParams &params)
    {
        numOutputs = params.get<int>("num_output");
        bias = params.get<bool>("bias_term", true);
        axis = params.get<int>("axis", 1);

        CV_Assert(0 <= axis && axis < 4);
        CV_Assert(params.learnedBlobs.size() >= 1);
        CV_Assert(!bias || (params.learnedBlobs.size() >= 2 && (int)params.learnedBlobs[1].total() == numOutputs));

        learnedParams.resize(bias ? 2 : 1);
        learnedParams[0] = params.learnedBlobs[0];
        if (bias)
        {
            learnedParams[1] = params.learnedBlobs[1];
        }
    }

    void FullyConnectedLayer::allocate(const std::vector<Blob*> &inputs, std::vector<Blob> &outputs)
    {
        CV_Assert(inputs.size() > 0);

        innerSize = (int)inputs[0]->total(axis);
        CV_Assert((size_t)innerSize * (size_t)numOutputs == learnedParams[0].total());
        CV_Assert(learnedParams[0].rows() == numOutputs && learnedParams[0].cols() == innerSize);

        outputs.resize(inputs.size());
        for (size_t i = 0; i < inputs.size(); i++)
        {
            if (i != 0)
                CV_Assert(inputs[i]->total(axis) == (size_t)innerSize);

            this->reshape(*inputs[i], outputs[i]);
        }
    }

    void FullyConnectedLayer::reshape(const Blob &inp, Blob &out)
    {
        Vec4i inpShape = inp.shape4();
        Vec4i outShape = Vec4i::all(1);

        for (int a = 0; a < axis; a++)
            outShape[a] = inpShape[a];
        outShape[3] = numOutputs;

        out.create(outShape, inp.type());
    }

    void FullyConnectedLayer::forward(std::vector<Blob*> &inputs, std::vector<Blob> &outputs)
    {
        for (size_t i = 0; i < inputs.size(); i++)
        {
            int M = inputs[i]->total(0, axis);
            int N = numOutputs;
            int K = innerSize;

            Mat srcMat(M, K, CV_32F, inputs[i]->ptrf());
            Mat weights(N, K, CV_32F, learnedParams[0].ptrf());
            Mat dstMat(M, N, CV_32F, outputs[i].ptrf());

            //important: Caffe stores weights as transposed array
            cv::gemm(srcMat, weights, 1, noArray(), 0, dstMat, GEMM_2_T);

            if (bias)
            {
                Mat biasOnesMat = Mat::ones(M, 1, CV_32F);
                Mat biasMat(1, N, CV_32F, learnedParams[1].ptrf());
                cv::gemm(biasOnesMat, biasMat, 1, dstMat, 1, dstMat);
            }
        }
    }
}
}
