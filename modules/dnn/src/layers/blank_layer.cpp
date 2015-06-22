#include "../precomp.hpp"
#include "layers_common.hpp"

namespace cv
{
namespace dnn
{
    class BlankLayer : public Layer
    {
    public:

        BlankLayer(LayerParams &params)
        {

        }

        void allocate(const std::vector<Blob*> &inputs, std::vector<Blob> &outputs)
        {
            outputs.resize(inputs.size());
            for (size_t i = 0; i < inputs.size(); i++)
                outputs[i] = *inputs[i];
        }

        void forward(std::vector<Blob*> &inputs, std::vector<Blob> &outputs)
        {
            for (size_t i = 0; i < inputs.size(); i++)
                outputs[i] = *inputs[i];
        }
    };

    static Ptr<Layer> blankLayerRegisterer(LayerParams &params)
    {
        return Ptr<Layer>(new BlankLayer(params));
    }


    REGISTER_LAYER_FUNC(Dropout, blankLayerRegisterer)
}
}