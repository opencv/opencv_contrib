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

}
}