#include "../precomp.hpp"
#include "elementwise_layers.hpp"

namespace cv
{
namespace dnn
{

#define ACTIVATION_CREATOR_FOR(_Layer, _Functor, ...) \
Ptr<_Layer> _Layer::create() { \
    return return Ptr<_Layer>( new ElementWiseLayer<_Functor>(_Functor()) ); }


Ptr<ReLULayer> ReLULayer::create(double negativeSlope)
{
    return Ptr<ReLULayer>(new ElementWiseLayer<ReLUFunctor>(ReLUFunctor(negativeSlope)));
}

Ptr<TanHLayer> TanHLayer::create()
{
    return Ptr<TanHLayer>(new ElementWiseLayer<TanHFunctor>());
}

Ptr<SigmoidLayer> SigmoidLayer::create()
{
    return Ptr<SigmoidLayer>(new ElementWiseLayer<SigmoidFunctor>());
}

Ptr<AbsLayer> AbsLayer::create()
{
    return Ptr<AbsLayer>(new ElementWiseLayer<AbsValFunctor>());
}

Ptr<BNLLLayer> BNLLLayer::create()
{
    return Ptr<BNLLLayer>(new ElementWiseLayer<BNLLFunctor>());
}

Ptr<PowerLayer> PowerLayer::create(double power /*= 1*/, double scale /*= 1*/, double shift /*= 0*/)
{
    const PowerFunctor f(power, scale, shift);
    return Ptr<PowerLayer>(new ElementWiseLayer<PowerFunctor>(f));
}


Ptr<Layer> createReLULayerFromCaffe(LayerParams &params)
{
    float negative_slope;

    if (params.has("negative_slope"))
        negative_slope = params.get<float>("negative_slope");
    else
        negative_slope = 0.f;

    return Ptr<Layer>(ReLULayer::create(negative_slope));
}


Ptr<Layer> createSigmoidLayerFromCaffe(LayerParams&)
{
    return Ptr<Layer>(SigmoidLayer::create());
}

Ptr<Layer> createTanHLayerFromCaffe(LayerParams&)
{
    return Ptr<Layer>(TanHLayer::create());
}

Ptr<Layer> createAbsLayerFromCaffe(LayerParams&)
{
    return Ptr<Layer>(AbsLayer::create());
}

Ptr<Layer> createBNLLLayerFromCaffe(LayerParams&)
{
    return Ptr<Layer>(BNLLLayer::create());
}

Ptr<Layer> createPowerLayerFromCaffe(LayerParams &params)
{
    float power = params.get<float>("power", 1.0f);
    float scale = params.get<float>("scale", 1.0f);
    float shift = params.get<float>("shift", 0.0f);

    return Ptr<Layer>(PowerLayer::create(power, scale, shift));
}

}
}