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
            CV_Assert(inputs.size() == 1);
            outputs[0] = *inputs[0];
        }

        void forward(std::vector<Blob*> &inputs, std::vector<Blob> &outputs)
        {
            CV_Assert(inputs.size() == 1 && outputs.size() == 1);
            CV_Assert(inputs[0]->getMatRef().data == outputs[0].getMatRef().data);

            float *data = outputs[0].getMatRef().ptr<float>();

            //Vec4i shape = outputs[0].shape();
            //CV_Assert(pitch[i] == shape[i] * sizeof(float) );

            for (size_t i = 0; i < outputs[0].total(); i++)
                data[i] = func(data[i]);
        }
    };

    class PoolingLayer : public Layer
    {
        int type;
        int strideH, strideW;
        int sizeH, sizeW;

    public:
        PoolingLayer(LayerParams &params);
        void allocate(const std::vector<Blob*> &inputs, std::vector<Blob> &outputs);
        void forward(std::vector<Blob*> &inputs, std::vector<Blob> &outputs);
    };

    class ConvolutionLayer : public Layer
    {
        int groups;
        int strideH, strideW;
        int sizeH, sizeW;

    public:
        ConvolutionLayer(LayerParams &params);
        void allocate(const std::vector<Blob*> &inputs, std::vector<Blob> &outputs);
        void forward(std::vector<Blob*> &inputs, std::vector<Blob> &outputs);
    };

    class FullyConnectedLayer : public Layer
    {
        int numOutputs;

    public:
        FullyConnectedLayer(LayerParams &params);
        void allocate(const std::vector<Blob*> &inputs, std::vector<Blob> &outputs);
        void forward(std::vector<Blob*> &inputs, std::vector<Blob> &outputs);
    };
}
}


#endif