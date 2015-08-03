#include "../precomp.hpp"
#include "layers_common.hpp"

namespace cv
{
namespace dnn
{
    class BlankLayer : public Layer
    {
    public:

        BlankLayer(LayerParams&)
        {

        }

        void allocate(const std::vector<Blob*> &inputs, std::vector<Blob> &outputs)
        {
            outputs.resize(inputs.size());
            for (size_t i = 0; i < inputs.size(); i++)
                outputs[i].shareFrom(*inputs[i]);
        }

        void forward(std::vector<Blob*> &inputs, std::vector<Blob> &outputs)
        {
            for (size_t i = 0; i < inputs.size(); i++)
                outputs[i] = *inputs[i];
        }
    };

    REGISTER_LAYER_CLASS(Dropout, BlankLayer)
}
}