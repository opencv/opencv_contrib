#include "../precomp.hpp"
#include "layers_common.hpp"
#include "fully_connected_layer.hpp"

namespace cv
{
namespace dnn
{
    FullyConnectedLayer::FullyConnectedLayer(LayerParams &params) : Layer(params)
    {
        numOutputs = params.get<int>("num_output");
        bias = params.get<bool>("bias_term", true);
        axis_ = params.get<int>("axis", 1);

        CV_Assert(blobs.size() == (bias ? 2 : 1));
        CV_Assert(blobs[0].dims() >= 2 && blobs[0].total() >= (size_t)numOutputs);
        CV_Assert(!bias || blobs[1].total() == (size_t)numOutputs);
    }

    void FullyConnectedLayer::allocate(const std::vector<Blob*> &inputs, std::vector<Blob> &outputs)
    {
        CV_Assert(inputs.size() > 0);

        axis = inputs[0]->canonicalAxis(axis_);
        innerSize = (int)inputs[0]->total(axis);

        CV_Assert((size_t)innerSize * (size_t)numOutputs == blobs[0].total());
        CV_Assert(blobs[0].size(-2) == numOutputs && blobs[0].size(-1) == innerSize);

        outputs.resize(inputs.size());
        for (size_t i = 0; i < inputs.size(); i++)
        {
            if (i != 0)
                CV_Assert(inputs[i]->equalShape(*inputs[0]));

            this->reshape(*inputs[i], outputs[i]);
        }
    }

    void FullyConnectedLayer::reshape(const Blob &inp, Blob &out)
    {
        BlobShape inpShape = inp.shape();
        BlobShape outShape(axis+1, inpShape.ptr());
        outShape[axis] = numOutputs;

        out.create(outShape, inp.type());
    }

    void FullyConnectedLayer::forward(std::vector<Blob*> &inputs, std::vector<Blob> &outputs)
    {
        for (size_t i = 0; i < inputs.size(); i++)
        {
            int M = (int)inputs[i]->total(0, axis);
            int N = numOutputs;
            int K = innerSize;

            Mat srcMat(M, K, inputs[i]->type(), inputs[i]->ptrf());
            Mat weight(N, K, blobs[0].type(), blobs[0].ptrf());
            Mat dstMat(M, N, outputs[i].type(), outputs[i].ptrf());

            //important: Caffe stores weights as transposed array
            cv::gemm(srcMat, weight, 1, noArray(), 0, dstMat, GEMM_2_T);

            if (bias)
            {
                Mat biasOnesMat = Mat::ones(M, 1, CV_32F);
                Mat biasMat(1, N, CV_32F, blobs[1].ptrf());
                cv::gemm(biasOnesMat, biasMat, 1, dstMat, 1, dstMat);
            }
        }
    }
}
}
