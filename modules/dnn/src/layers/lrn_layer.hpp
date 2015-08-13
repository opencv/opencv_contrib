#ifndef __OPENCV_DNN_LAYERS_LRN_LAYER_HPP__
#define __OPENCV_DNN_LAYERS_LRN_LAYER_HPP__
#include "../precomp.hpp"

namespace cv
{
namespace dnn
{
    class LRNLayer : public Layer
    {
        enum
        {
            CHANNEL_NRM,
            SPATIAL_NRM,
            SPATIAL_CONTRAST_NRM //cuda-convnet feature
        } type;

        int size;
        double alpha, beta;

        Blob bufBlob;

        void channelNoramlization(Blob &src, Blob &dst);
        void spatialNormalization(Blob &src, Blob &dst);

    public:

        LRNLayer(LayerParams &params);
        void allocate(const std::vector<Blob*> &inputs, std::vector<Blob> &outputs);
        void forward(std::vector<Blob*> &inputs, std::vector<Blob> &outputs);
    };
}
}
#endif