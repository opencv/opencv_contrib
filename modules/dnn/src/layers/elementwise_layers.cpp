#include "../precomp.hpp"
#include "elementwise_layers.hpp"
#include "opencv2/imgproc.hpp"

namespace cv
{
namespace dnn
{

#define ACTIVATION_CREATOR_FOR(_Layer, _Functor, ...) \
Ptr<_Layer> _Layer::create() { \
    return return Ptr<_Layer>( new ElementWiseLayer<_Functor>(_Functor()) ); }


Ptr<ReLULayer> ReLULayer::create(double negativeSlope)
{
    Ptr<ReLULayer> layer(new ElementWiseLayer<ReLUFunctor>(ReLUFunctor(negativeSlope)));
    layer->negativeSlope = negativeSlope;
    return layer;
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
    Ptr<PowerLayer> layer(new ElementWiseLayer<PowerFunctor>(f));
    layer->power = power;
    layer->scale = scale;
    layer->shift = shift;
    return layer;
}

////////////////////////////////////////////////////////////////////////////

void ChannelsPReLULayerImpl::allocate(const std::vector<Blob*> &inputs, std::vector<Blob> &outputs)
{
    CV_Assert(blobs.size() == 1);

    outputs.resize(inputs.size());
    for (size_t i = 0; i < inputs.size(); i++)
    {
        outputs[i].create(inputs[i]->shape());
    }
}

void ChannelsPReLULayerImpl::forward(std::vector<Blob*> &inputs, std::vector<Blob> &outputs)
{
    CV_Assert(inputs.size() == 1);

    Blob &inpBlob = *inputs[0];

    for (size_t ii = 0; ii < outputs.size(); ii++)
    {
        Blob &outBlob = outputs[ii];

        CV_Assert(blobs[0].total() == inpBlob.channels());

        for (int n = 0; n < inpBlob.channels(); n++)
        {
            float slopeWeight = blobs[0].matRefConst().at<float>(n);

            cv::threshold(inpBlob.getPlane(0, n), outBlob.getPlane(0, n), 0, 0, cv::THRESH_TOZERO_INV);
            outBlob.getPlane(0, n) = inpBlob.getPlane(0, n) + (slopeWeight - 1)*outBlob.getPlane(0, n);
        }
    }
}

Ptr<ChannelsPReLULayer> ChannelsPReLULayer::create()
{
    return Ptr<ChannelsPReLULayer>(new ChannelsPReLULayerImpl());
}

}
}
