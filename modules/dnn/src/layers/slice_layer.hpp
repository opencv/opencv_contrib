#ifndef __OPENCV_DNN_LAYERS_SLICE_LAYER_HPP__
#define __OPENCV_DNN_LAYERS_SLICE_LAYER_HPP__
#include "../precomp.hpp"

namespace cv
{
namespace dnn
{

class SliceLayer : public Layer
{
public:
    SliceLayer(LayerParams &params);

    void allocate(const std::vector<Blob*> &inputs, std::vector<Blob> &outputs);

    void forward(std::vector<Blob*> &inputs, std::vector<Blob> &outputs);

private:
    int inAxis;
    std::vector<int> slicePoints;
};

}
}
#endif