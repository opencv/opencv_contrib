#include "../precomp.hpp"
#include "layer_loaders.hpp"
#include <opencv2/dnn/shape_utils.hpp>
#include "../layers/layers_common.hpp"

namespace cv
{
namespace dnn
{

template<>
Ptr<Layer> createLayerFromCaffe<PoolingLayer>(LayerParams &params)
{
    int type;
    Size kernel, stride, pad;

    if (params.has("pool"))
    {
        String pool = params.get<String>("pool").toLowerCase();
        if (pool == "max")
            type = PoolingLayer::MAX;
        else if (pool == "ave")
            type = PoolingLayer::AVE;
        else if (pool == "stochastic")
            type = PoolingLayer::STOCHASTIC;
        else
            CV_Error(Error::StsBadArg, "Unknown pooling type \"" + pool + "\"");
    }
    else
    {
        type = PoolingLayer::MAX;
    }

    getCaffeConvParams(params, kernel, pad, stride);

    return Ptr<Layer>(PoolingLayer::create(type, kernel, stride, pad));
}

template<>
Ptr<Layer> createLayerFromCaffe<SoftmaxLayer>(LayerParams &params)
{
    int axis = params.get<int>("axis", 1);
    return Ptr<Layer>(SoftmaxLayer::create(axis));
}

template<> //InnerProduct specialization
Ptr<Layer> createLayerFromCaffe<InnerProductLayer>(LayerParams &params)
{
    const std::vector<Blob> &blobs = params.blobs;
    CV_Assert(1 <= blobs.size() && blobs.size() <= 2);

    int numOutputs = params.get<int>("num_output");
    int innerSize = (int)blobs[0].total() / numOutputs;
    bool bias = params.get<bool>("bias_term", true);
    int axis = params.get<int>("axis", 1);

    CV_Assert(blobs[0].dims() >= 2 && (size_t)(innerSize * numOutputs) == blobs[0].total());
    CV_Assert(!bias || (blobs.size() == 2 && (size_t)numOutputs == blobs[1].total()));

    Ptr<InnerProductLayer> l = InnerProductLayer::create(axis);
    l->setParamsFrom(params);
    l->blobs[0].reshape(Shape(numOutputs, innerSize));
    if (bias)
        l->blobs[1].reshape(Shape(1, numOutputs));

    return Ptr<Layer>(l);
}

template<> //LRNLayer specialization
Ptr<Layer> createLayerFromCaffe<LRNLayer>(LayerParams& params)
{
    int type;
    String nrmType = params.get<String>("norm_region", "ACROSS_CHANNELS");
    if (nrmType == "ACROSS_CHANNELS")
        type = LRNLayer::CHANNEL_NRM;
    else if (nrmType == "WITHIN_CHANNEL")
        type = LRNLayer::SPATIAL_NRM;
    else
        CV_Error(Error::StsBadArg, "Unknown region type \"" + nrmType + "\"");

    int size = params.get<int>("local_size", 5);
    if (size % 2 != 1 || size <= 0)
        CV_Error(Error::StsBadArg, "LRN layer supports only positive odd values for local_size");

    double alpha = params.get<double>("alpha", 1);
    double beta = params.get<double>("beta", 0.75);

    return Ptr<Layer>(LRNLayer::create(type, size, alpha, beta));
}

//Activation layers

template <typename ActivationLayer> //Intended for parameters-free activations
Ptr<Layer> createLayerFromCaffe(LayerParams&)
{
    return Ptr<Layer>(ActivationLayer::create());
}

template<> //ReLU specialization
Ptr<Layer> createLayerFromCaffe<ReLULayer>(LayerParams& params)
{
    float negative_slope = params.get<float>("negative_slope", 0.f);
    return Ptr<Layer>(ReLULayer::create(negative_slope));
}

template<> //Power specialization
Ptr<Layer> createLayerFromCaffe<PowerLayer>(LayerParams& params)
{
    float power = params.get<float>("power", 1.0f);
    float scale = params.get<float>("scale", 1.0f);
    float shift = params.get<float>("shift", 0.0f);
    return Ptr<Layer>(PowerLayer::create(power, scale, shift));
}

//Explicit instantiation
template Ptr<Layer> createLayerFromCaffe<SoftmaxLayer>(LayerParams&);
template Ptr<Layer> createLayerFromCaffe<InnerProductLayer>(LayerParams&);
template Ptr<Layer> createLayerFromCaffe<LRNLayer>(LayerParams&);

template Ptr<Layer> createLayerFromCaffe<ReLULayer>(LayerParams&);
template Ptr<Layer> createLayerFromCaffe<SigmoidLayer>(LayerParams&);
template Ptr<Layer> createLayerFromCaffe<TanHLayer>(LayerParams&);
template Ptr<Layer> createLayerFromCaffe<AbsLayer>(LayerParams&);
template Ptr<Layer> createLayerFromCaffe<BNLLLayer>(LayerParams&);
template Ptr<Layer> createLayerFromCaffe<PowerLayer>(LayerParams&);

}
}