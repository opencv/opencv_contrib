#include "../precomp.hpp"
#include "layers_common.hpp"

namespace cv
{
namespace dnn
{
    //TODO: implement axis number parameter
    class FullyConnectedLayer : public Layer
    {
        bool bias;
        int numOutputs;

        int inC, inH, inW;
        int inSize;

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

        inC = inputs[0]->channels();
        inH = inputs[0]->rows();
        inW = inputs[0]->cols();
        inSize = inC * inH * inW;

        CV_Assert((size_t)inSize * (size_t)numOutputs == learnedParams[0].total());

        outputs.resize(inputs.size());
        for (size_t i = 0; i < inputs.size(); i++)
        {
            if (i != 0)
                CV_Assert(inputs[i]->channels() == inC && inputs[i]->rows() == inH && inputs[i]->cols() == inW);

            outputs[i].create(inputs[i]->num(), numOutputs, 1, 1);
        }
    }

    void FullyConnectedLayer::forward(std::vector<Blob*> &inputs, std::vector<Blob> &outputs)
    {
        for (size_t i = 0; i < inputs.size(); i++)
        {
            int M = inputs[i]->num();
            int N = numOutputs;
            int K = inSize;

            Mat srcMat(M, K, CV_32F, inputs[i]->ptr<float>());
            Mat weights(K, N, CV_32F, learnedParams[0].ptr<float>());
            Mat dstMat(M, N, CV_32F, outputs[i].ptr<float>());

            cv::gemm(srcMat, weights, 1, noArray(), 0, dstMat);

            if (bias)
            {
                Mat biasOnesMat = Mat::ones(M, 1, CV_32F);
                Mat biasMat(1, N, CV_32F, learnedParams[1].ptr<float>());
                cv::gemm(biasOnesMat, biasMat, 1, dstMat, 1, dstMat);
            }
        }
    }
}
}
