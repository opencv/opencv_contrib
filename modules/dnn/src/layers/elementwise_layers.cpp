#include "../precomp.hpp"
#include "layers_common.hpp"
#include <math.h>

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
                outputs[i] = *inputs[i]; //no data copy
        }

        void forward(std::vector<Blob*> &inputs, std::vector<Blob> &outputs)
        {
            CV_Assert(inputs.size() == outputs.size());

            for (size_t i = 0; i < inputs.size(); i++)
            {
                CV_Assert(inputs[i]->ptr<float>() == outputs[i].ptr<float>());
                float *data = outputs[i].ptr<float>();
                size_t size = outputs[i].total();

                for (size_t j = 0; j < size; j++)
                    data[j] = func(data[j]);
            }
        }
    };


    struct ReLUFunctor
    {
        float negative_slope;

        ReLUFunctor(LayerParams &params)
        {
            if (params.has("negative_slope"))
                negative_slope = params.get<float>("negative_slope");
            else
                negative_slope = 0.f;
        }

        inline float operator()(float x)
        {
            return (x >= 0) ? x : negative_slope * x;
        }
    };

    struct TanHFunctor
    {
        TanHFunctor(LayerParams&) {}

        inline float operator()(float x)
        {
            return tanh(x);
        }
    };

    REGISTER_LAYER_CLASS(ReLU, ElementWiseLayer<ReLUFunctor>)
    REGISTER_LAYER_CLASS(TanH, ElementWiseLayer<TanHFunctor>)

}
}