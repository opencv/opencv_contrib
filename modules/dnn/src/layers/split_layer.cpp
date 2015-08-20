#include "../precomp.hpp"
#include "layers_common.hpp"
#include "split_layer.hpp"

namespace cv
{
namespace dnn
{

//TODO: maybe "top_count" param is useless because it can be determined by output connections number?
SplitLayer::SplitLayer(LayerParams &params) : Layer(params)
{
    if (params.has("top_count"))
    {
        outputsNum = params.get<int>("top_count");
        CV_Assert(outputsNum >= 0);
    }
    else
    {
        outputsNum = -1;
    }
}

void SplitLayer::allocate(const std::vector<Blob*> &inputs, std::vector<Blob> &outputs)
{
    CV_Assert(inputs.size() == 1);

    if (outputsNum >= 0)
        outputs.resize(outputsNum);

    for (size_t i = 0; i < outputs.size(); i++)
        outputs[i].create(inputs[0]->shape(), inputs[0]->type());
}

void SplitLayer::forward(std::vector<Blob*> &inputs, std::vector<Blob> &outputs)
{
    for (size_t i = 0; i < outputs.size(); i++)
        inputs[0]->getMatRef().copyTo(outputs[i].getMatRef());
}

}
}
