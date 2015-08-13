#ifndef __OPENCV_DNN_LAYERS_POOLING_LAYER_HPP__
#define __OPENCV_DNN_LAYERS_POOLING_LAYER_HPP__
#include "../precomp.hpp"

namespace cv
{
namespace dnn
{
    class PoolingLayer : public Layer
    {
        enum
        {
            MAX,
            AVE,
            STOCHASTIC
        };

        int type;
        int padH, padW;
        int strideH, strideW;
        int kernelH, kernelW;

        int inpH, inpW;
        int outH, outW;

        void computeOutputShape(int inpH, int inpW);
        void maxPooling(Blob &input, Blob &output);
        void avePooling(Blob &input, Blob &output);

    public:
        PoolingLayer(LayerParams &params);
        void allocate(const std::vector<Blob*> &inputs, std::vector<Blob> &outputs);
        void forward(std::vector<Blob*> &inputs, std::vector<Blob> &outputs);
    };
}
}
#endif