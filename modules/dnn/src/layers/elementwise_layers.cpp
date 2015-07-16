#include "../precomp.hpp"
#include "layers_common.hpp"
#include <cmath>
using std::abs;
using std::exp;
using std::tanh;
using std::pow;

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
                outputs[i].shareFrom(*inputs[i]); //no data copy
        }

        void forward(std::vector<Blob*> &inputs, std::vector<Blob> &outputs)
        {
            for (size_t i = 0; i < inputs.size(); i++)
            {
                CV_Assert(inputs[i]->ptrRaw() == outputs[i].ptrRaw() && inputs[i]->type() == outputs[i].type());

                size_t size = outputs[i].total();

                if (outputs[i].isFloat())
                {
                    float *data = outputs[i].ptrf();
                    for (size_t j = 0; j < size; j++)
                        data[j] = func(data[j]);
                }
                else if (outputs[i].isDouble())
                {
                    double *data = outputs[i].ptr<double>();
                    for (size_t j = 0; j < size; j++)
                        data[j] = func(data[j]);
                }
                else
                {
                    CV_Error(Error::StsNotImplemented, "Only CV_32F and CV_64F blobs are supported");
                }
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

        template<typename TFloat>
        inline TFloat operator()(TFloat x)
        {
            return (x >= (TFloat)0) ? x : negative_slope * x;
        }
    };

    struct TanHFunctor
    {
        TanHFunctor(LayerParams&) {}

        template<typename TFloat>
        inline TFloat operator()(TFloat x)
        {
            return tanh(x);
        }
    };
    
    struct SigmoidFunctor
    {
        SigmoidFunctor(LayerParams&) {}

        template<typename TFloat>
        inline TFloat operator()(TFloat x)
        {
            return (TFloat)1 / ((TFloat)1 + exp(-x));
        }
    };

    struct AbsValFunctor
    {
        AbsValFunctor(LayerParams&) {}

        template<typename TFloat>
        inline TFloat operator()(TFloat x)
        {
            return abs(x);
        }
    };

    struct PowerFunctor
    {
        float power, scale, shift;

        PowerFunctor(LayerParams &params)
        {
            power = params.get<float>("power", 1.0f);
            scale = params.get<float>("scale", 1.0f);
            shift = params.get<float>("shift", 0.0f);
        }

        template<typename TFloat>
        inline TFloat operator()(TFloat x)
        {
            return pow((TFloat)shift + (TFloat)scale * x, (TFloat)power);
        }
    };

    struct BNLLFunctor
    {
        BNLLFunctor(LayerParams&) {}

        template<typename TFloat>
        inline TFloat operator()(TFloat x)
        {
            return log((TFloat)1 + exp(x));
        }
    };

    REGISTER_LAYER_CLASS(ReLU, ElementWiseLayer<ReLUFunctor>)
    REGISTER_LAYER_CLASS(TanH, ElementWiseLayer<TanHFunctor>)
    REGISTER_LAYER_CLASS(BNLL, ElementWiseLayer<BNLLFunctor>)
    REGISTER_LAYER_CLASS(Power, ElementWiseLayer<PowerFunctor>)
    REGISTER_LAYER_CLASS(AbsVal, ElementWiseLayer<AbsValFunctor>)
    REGISTER_LAYER_CLASS(Sigmoid, ElementWiseLayer<SigmoidFunctor>)

}
}