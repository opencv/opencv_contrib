#ifndef __OPENCV_DNN_LAYERS_HPP__
#define __OPENCV_DNN_LAYERS_HPP__
#include <opencv2/dnn.hpp>

namespace cv
{
namespace dnn
{

    template<typename Func>
    class ElementWiseLayer : public Layer
    {
        Func func;
    public:

        ElementWiseLayer(LayerParams &_params) : func(_params) {}

        void allocate(const std::vector<Blob*> &inputs, std::vector<Blob> &outputs)
        {
            outputs.resize(inputs.size());
            for (size_t i = 0; i < inputs.size(); i++)
                outputs[i] = *inputs[i];
        }

        void forward(std::vector<Blob*> &inputs, std::vector<Blob> &outputs)
        {
            CV_Assert(inputs.size() == outputs.size());

            for (size_t i = 0; i < inputs.size(); i++)
            {
                CV_Assert(inputs[i]->ptr<float>() == outputs[i].ptr<float>());
                float *data = outputs[i].ptr<float>();
                size_t size = outputs[i].total();

                //Vec4i shape = outputs[0].shape();
                //CV_Assert(pitch[i] == shape[i] * sizeof(float) );

                for (size_t j = 0; j < size; j++)
                    data[j] = func(data[j]);
            }
        }
    };

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

        int inH, inW;
        int pooledH, pooledW;

        void computeOutputShape(int inH, int inW);
        void maxPooling(Blob &input, Blob &output);

    public:
        PoolingLayer(LayerParams &params);
        void allocate(const std::vector<Blob*> &inputs, std::vector<Blob> &outputs);
        void forward(std::vector<Blob*> &inputs, std::vector<Blob> &outputs);
    };

    class ConvolutionLayer : public Layer
    {
        bool bias;
        int numOutput, group;
        int padH, padW;
        int strideH, strideW;
        int kernelH, kernelW;

        int inH, inW, inCn, colCn;
        int outH, outW;

        Mat imColsMat, biasOnesMat;

        void computeOutputShape(int inH, int inW);

    public:
        ConvolutionLayer(LayerParams &params);
        void allocate(const std::vector<Blob*> &inputs, std::vector<Blob> &outputs);
        void forward(std::vector<Blob*> &inputs, std::vector<Blob> &outputs);
    };

    class FullyConnectedLayer : public Layer
    {
        bool bias;
        int numOutputs;

        int inC, inH, inW;
        size_t inSize;

    public:
        FullyConnectedLayer(LayerParams &params);
        void allocate(const std::vector<Blob*> &inputs, std::vector<Blob> &outputs);
        void forward(std::vector<Blob*> &inputs, std::vector<Blob> &outputs);
    };
}
}


#endif