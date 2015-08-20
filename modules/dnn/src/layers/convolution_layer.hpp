#ifndef __OPENCV_DNN_LAYERS_CONVOLUTION_LAYER_HPP__
#define __OPENCV_DNN_LAYERS_CONVOLUTION_LAYER_HPP__
#include "../precomp.hpp"

namespace cv
{
namespace dnn
{
    //TODO: simultaneously convolution and bias addition for cache optimization
    class ConvolutionLayer : public Layer
    {
    protected:
        bool bias;
        int numOutput, group;
        int padH, padW;
        int kerH, kerW;
        int strideH, strideW;

        int inpH, inpW, inpCn;
        int outH, outW, outCn;
        int topH, topW, topCn; //switched between inp/out on deconv/conv
        int inpGroupCn, outGroupCn;
        int ksize;

        bool useOpenCL;
        Mat colMat, biasOnesMat;

        inline bool is1x1() const;
        virtual void computeInpOutShape(const Blob &inpBlob);
        void im2col(Blob &inpBlob, int imNum, int cnGroup);

    public:
        ConvolutionLayer() {}
        ConvolutionLayer(LayerParams &params);
        void allocate(const std::vector<Blob*> &inputs, std::vector<Blob> &outputs);
        void forward(std::vector<Blob*> &inputs, std::vector<Blob> &outputs);
    };

    class DeConvolutionLayer : public ConvolutionLayer
    {
    protected:
        void computeInpOutShape(const Blob &inpBlob);
        void col2im(Mat &dstMat);

    public:
        DeConvolutionLayer(LayerParams &params);
        void forward(std::vector<Blob*> &inputs, std::vector<Blob> &outputs);
    };
}
}
#endif
