#ifndef __OPENCV_DNN_LAYERS_RESHAPE_LAYER_HPP__
#define __OPENCV_DNN_LAYERS_RESHAPE_LAYER_HPP__
#include "../precomp.hpp"

namespace cv
{
namespace dnn
{

class ReshapeLayer : public Layer
{
public:
    ReshapeLayer(LayerParams &params);

    void allocate(const std::vector<Blob*> &inputs, std::vector<Blob> &outputs);

    void forward(std::vector<Blob*>&, std::vector<Blob>&) {}

protected:
    BlobShape shapeDesc;
    int inAxis, inNumAxes, autoAxisIdx;

    void computeOutputShape(int startAxis, int endAxis, BlobShape &inpShape, BlobShape &outShape);
};

Ptr<Layer> createFlattenLayer(LayerParams&);

}
}
#endif
