#ifndef __OPENCV_DNN_LAYERS_SPLIT_LAYER_HPP__
#define __OPENCV_DNN_LAYERS_SPLIT_LAYER_HPP__
#include "../precomp.hpp"

namespace cv
{
namespace dnn
{

class SplitLayer : public Layer
{
public:
    SplitLayer(LayerParams &params);

    void allocate(const std::vector<Blob*> &inputs, std::vector<Blob> &outputs);

    void forward(std::vector<Blob*> &inputs, std::vector<Blob> &outputs);

private:
    int outputsNum;
};

}
}
#endif