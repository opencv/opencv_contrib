#include "../precomp.hpp"
#include "layer_loaders.hpp"
#include <opencv2/dnn/shape_utils.hpp>
#include <climits>

namespace cv
{
namespace dnn
{

//Utils

//Extracts params used into Conv, Deconv and Pooling layers
static void getCaffeConvParams(LayerParams &params, Size &kernel, Size &pad, Size &stride)
{
    if (params.has("kernel_h") && params.has("kernel_w"))
    {
        kernel.height = params.get<int>("kernel_h");
        kernel.width = params.get<int>("kernel_w");
    }
    else if (params.has("kernel_size"))
    {
        kernel.height = kernel.width = params.get<int>("kernel_size");
    }
    else
    {
        CV_Error(Error::StsBadArg, "kernel_size (or kernel_h and kernel_w) not specified");
    }
    CV_Assert(kernel.height > 0 && kernel.width > 0);

    if (params.has("pad_h") && params.has("pad_w"))
    {
        pad.height = params.get<int>("pad_h");
        pad.width = params.get<int>("pad_w");
    }
    else
    {
        pad.height = pad.width = params.get<int>("pad", 0);
    }
    CV_Assert(pad.height >= 0 && pad.width >= 0);

    if (params.has("stride_h") && params.has("stride_w"))
    {
        stride.height = params.get<int>("stride_h");
        stride.width = params.get<int>("stride_w");
    }
    else
    {
        stride.height = stride.width = params.get<int>("stride", 1);
    }
    CV_Assert(stride.height > 0 && stride.width > 0);
}

static void getCaffeConvDilation(LayerParams &params, Size &dilation)
{
    if (params.has("dilation_h") && params.has("dilation_w"))
    {
        dilation.height = params.get<int>("dilation_h");
        dilation.width = params.get<int>("dilation_w");
    }
    else
    {
        dilation.height = dilation.width = params.get<int>("dilation", 1);
    }
    CV_Assert(dilation.height > 0 && dilation.width > 0);
}

//Layers

//Convolution and Deconvolution
static void initConvDeconvLayerFromCaffe(Ptr<BaseConvolutionLayer> l, LayerParams &params)
{
    l->setParamsFrom(params);
    getCaffeConvParams(params, l->kernel, l->pad, l->stride);

    bool bias = params.get<bool>("bias_term", true);
    int numOutput = params.get<int>("num_output");
    int group = params.get<int>("group", 1);

    CV_Assert(numOutput % group == 0);
    CV_Assert((bias && l->blobs.size() == 2) || (!bias && l->blobs.size() == 1));
}

template<>
Ptr<Layer> createLayerFromCaffe<ConvolutionLayer>(LayerParams &params)
{
    Size kernel, stride, pad, dilation;
    getCaffeConvParams(params, kernel, pad, stride);
    getCaffeConvDilation(params, dilation);

    Ptr<BaseConvolutionLayer> l = ConvolutionLayer::create(kernel, stride, pad, dilation);
    initConvDeconvLayerFromCaffe(l, params);
    return Ptr<Layer>(l);
}

template<>
Ptr<Layer> createLayerFromCaffe<DeconvolutionLayer>(LayerParams &params)
{
    Ptr<BaseConvolutionLayer> l = DeconvolutionLayer::create();
    initConvDeconvLayerFromCaffe(l, params);
    return Ptr<Layer>(l);
}

template<>
Ptr<Layer> createLayerFromCaffe<PoolingLayer>(LayerParams &params)
{
    int type;
    Size kernel, stride, pad;
    bool globalPooling;

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

    globalPooling = params.has("global_pooling");

    if (globalPooling)
    {
        if(params.has("kernel_h") || params.has("kernel_w") || params.has("kernel_size"))
        {
            CV_Error(cv::Error::StsBadArg, "In global_pooling mode, kernel_size (or kernel_h and kernel_w) cannot be specified");
        }
        if(pad.height != 0 || pad.width != 0 || stride.height != 1 || stride.width != 1)
        {
            CV_Error(cv::Error::StsBadArg, "In global_pooling mode, pad_h and pad_w must be = 0, and stride_h and stride_w must be = 1");
        }
    }

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

template<>
Ptr<Layer> createLayerFromCaffe<MVNLayer>(LayerParams &params)
{
    return Ptr<Layer>(MVNLayer::create(
                          params.get<bool>("normalize_variance", true),
                          params.get<bool>("across_channels", false),
                          params.get<double>("eps", 1e-9)
                      ));
}

/* Reshape layers */

template<>
Ptr<Layer> createLayerFromCaffe<ReshapeLayer>(LayerParams &params)
{
    int axis = params.get<int>("axis", 0);
    int numAxes = params.get<int>("num_axes", -1);
    CV_Assert(numAxes >= -1);
    Range applyingRange = (numAxes == -1) ? Range(axis, INT_MAX) : Range(axis, axis + numAxes);

    Shape newShape;
    if (params.has("dim"))
    {
        const DictValue &paramShape = params.get("dim");
        newShape = Shape::all(paramShape.size());
        for (int i = 0; i < paramShape.size(); i++)
            newShape[i] = paramShape.get<int>(i);
    }
    else
        newShape = Shape::all(0);

    return Ptr<Layer>(ReshapeLayer::create(newShape, applyingRange));
}

Ptr<Layer> createFlattenLayerFromCaffe(LayerParams&)
{
    return Ptr<Layer>(ReshapeLayer::create(Shape(0, -1)));
}

template<>
Ptr<Layer> createLayerFromCaffe<ConcatLayer>(LayerParams& params)
{
    return Ptr<Layer>(ConcatLayer::create(params.get<int>("axis", 1)));
}

template<>
Ptr<Layer> createLayerFromCaffe<SplitLayer>(LayerParams &params)
{
    int outputsCount;

    //TODO: maybe "top_count" param is useless because it can be determined by output connections number
    if (params.has("top_count"))
    {
        outputsCount = params.get<int>("top_count");
        CV_Assert(outputsCount >= 0);
    }
    else
    {
        outputsCount = -1;
    }

    return Ptr<Layer>(SplitLayer::create(outputsCount));
}

template<>
Ptr<Layer> createLayerFromCaffe<SliceLayer>(LayerParams& params)
{
    int axis = params.get<int>("axis", 1);

    if (!params.has("slice_point"))
    {
        return Ptr<Layer>(SliceLayer::create(axis));
    }
    else
    {
        const DictValue &indicesValue = params.get("slice_point");
        std::vector<int> sliceIndices(indicesValue.size());
        for (int i = 0; i < indicesValue.size(); i++)
            sliceIndices[i] = indicesValue.get<int>(i);

        return Ptr<Layer>(SliceLayer::create(axis, sliceIndices));
    }
}

/* Activation layers */

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

template<> //Flatten specialization
Ptr<Layer> createLayerFromCaffe<FlattenLayer>(LayerParams& params)
{
    float startAxis = params.get<float>("axis", 0);
    float endAxis = params.get<float>("end_axis", -1);
    return Ptr<Layer>(FlattenLayer::create(startAxis, endAxis));
}

static void checkCurrentOrder(const std::vector<size_t>& order, int currentOrder)
{
    if(currentOrder < 0 || currentOrder > 3)
    {
        CV_Error(
            Error::StsBadArg,
            "Orders of dimensions in Permute layer parameter"
            "must be in [0...3] interval");
    }

    if(std::find(order.begin(), order.end(), currentOrder) != order.end())
    {
        CV_Error(Error::StsBadArg,
                 "Permute layer parameter contains duplicated orders.");
    }
}

template<> //Permute specialization
Ptr<Layer> createLayerFromCaffe<PermuteLayer>(LayerParams& params)
{
    std::vector<size_t> order;
    bool needsPermute = true;

    if (!params.has("order"))
    {
        needsPermute = false;
    }
    else
    {
        DictValue paramOrder = params.get("order");
        if(paramOrder.size() > 4)
        {
            CV_Error(
                Error::StsBadArg,
                "Too many (> 4) orders of dimensions in Permute layer");
        }

        for (int i = 0; i < paramOrder.size(); i++)
        {
            int currentOrder = paramOrder.get<int>(i);
            checkCurrentOrder(order, currentOrder);
            order.push_back(currentOrder);
        }
    }
    return Ptr<Layer>(PermuteLayer::create(order, needsPermute));
}

static void getAspectRatios(const LayerParams &params, const bool flip, std::vector<float> aspectRatios)
{
    DictValue aspectRatioParameter = params.get("aspect_ratio");

    for (int i = 0; i < aspectRatioParameter.size(); ++i)
    {
        float aspectRatio = aspectRatioParameter.get<float>(i);
        bool alreadyExists = false;

        for (size_t j = 0; j < aspectRatios.size(); ++j)
        {
            if (fabs(aspectRatio - aspectRatios[j]) < 1e-6)
            {
                alreadyExists = true;
                break;
            }
        }
        if (!alreadyExists)
        {
            aspectRatios.push_back(aspectRatio);
            if (flip)
            {
                aspectRatios.push_back(1./aspectRatio);
            }
        }
    }
}

static void getVariance(const LayerParams &params, std::vector<float> &variance)
{
    DictValue varianceParameter = params.get("variance");

    int varianceSize = varianceParameter.size();
    if (varianceSize > 1)
    {
        // Must and only provide 4 variance.
        CV_Assert(varianceSize == 4);

        for (int i = 0; i < varianceSize; ++i)
        {
            float var = varianceParameter.get<float>(i);
            CV_Assert(var > 0);
            variance.push_back(var);
        }
    }
    else
    {
        if (varianceSize == 1)
        {
            float var = varianceParameter.get<float>(0);
            CV_Assert(var > 0);
            variance.push_back(var);
        }
        else
        {
            // Set default to 0.1.
            variance.push_back(0.1f);
        }
    }
}


template<> //PriorBox specialization
Ptr<Layer> createLayerFromCaffe<PriorBoxLayer>(LayerParams& params)
{
    float minSize = params.get("min_size", 0.0f);
    CV_Assert(minSize > 0);

    bool flip = params.get("flip", false);
    bool clip = params.get("clip", false);

    std::vector<float> variance;
    std::vector<float> aspectRatios;

    aspectRatios.clear();
    aspectRatios.push_back(1.);

    getAspectRatios(params, flip, aspectRatios);
    getVariance(params, variance);

    size_t numPriors = aspectRatios.size();

    float maxSize = -1;
    if (params.has("max_size"))
    {
        maxSize = params.get("max_size", 0.0f);
        CV_Assert(maxSize > minSize);

        numPriors += 1;
    }

    return Ptr<Layer>(PriorBoxLayer::create(minSize, maxSize,
                                            aspectRatios, variance,
                                            flip, clip, numPriors));
}

template<> //NormalizeBBox specialization
Ptr<Layer> createLayerFromCaffe<NormalizeBBoxLayer>(LayerParams& params)
{
    float eps = params.get("eps", 1e-10f);
    bool acrossSpatial = params.get("across_spatial", false);
    bool channelShared = params.get("channel_shared", false);
    return Ptr<Layer>(NormalizeBBoxLayer::create(eps, acrossSpatial, channelShared));
}

static DetectionOutputLayer::CodeType getCodeType(LayerParams &params)
{
    String codeTypeString = params.get<String>("code_type").toLowerCase();
    if (codeTypeString == "corner")
        return caffe::PriorBoxParameter_CodeType_CORNER;
    else if (codeTypeString == "center_size")
        return caffe::PriorBoxParameter_CodeType_CENTER_SIZE;
    else
        return caffe::PriorBoxParameter_CodeType_CORNER;
}

template<> //DetectionOutput specialization
Ptr<Layer> createLayerFromCaffe<DetectionOutputLayer>(LayerParams& params)
{
    unsigned numClasses = params.get("num_classes", 5);
    bool shareLocation = params.get("share_location", false);
    int numLocClasses = shareLocation ? 1 : numClasses;
    int backgroundLabelId = params.get("background_label_id", 0);
    bool varianceEncodedInTarget = params.get("variance_encoded_in_target", false);
    int keepTopK = params.get("keep_top_k", 1);
    float confidenceThreshold = params.get("confidence_threshold", -FLT_MAX);
    int topK = params.get("top_k", -1);

    DetectionOutputLayer::CodeType codeType = getCodeType(params);

    float nmsThreshold = params.get("nms_threshold", 0.01);
    CV_Assert(nmsThreshold > 0.);

    return Ptr<Layer>(DetectionOutputLayer::create(numClasses,
                                                   shareLocation,
                                                   numLocClasses,
                                                   backgroundLabelId,
                                                   codeType,
                                                   varianceEncodedInTarget,
                                                   keepTopK,
                                                   confidenceThreshold,
                                                   nmsThreshold,
                                                   topK));
}

//Explicit instantiation
template Ptr<Layer> createLayerFromCaffe<ConvolutionLayer>(LayerParams&);
template Ptr<Layer> createLayerFromCaffe<DeconvolutionLayer>(LayerParams&);
template Ptr<Layer> createLayerFromCaffe<SoftmaxLayer>(LayerParams&);
template Ptr<Layer> createLayerFromCaffe<InnerProductLayer>(LayerParams&);
template Ptr<Layer> createLayerFromCaffe<LRNLayer>(LayerParams&);
template Ptr<Layer> createLayerFromCaffe<MVNLayer>(LayerParams&);

template Ptr<Layer> createLayerFromCaffe<ConcatLayer>(LayerParams&);
template Ptr<Layer> createLayerFromCaffe<SliceLayer>(LayerParams&);
template Ptr<Layer> createLayerFromCaffe<SplitLayer>(LayerParams&);

template Ptr<Layer> createLayerFromCaffe<ReLULayer>(LayerParams&);
template Ptr<Layer> createLayerFromCaffe<SigmoidLayer>(LayerParams&);
template Ptr<Layer> createLayerFromCaffe<TanHLayer>(LayerParams&);
template Ptr<Layer> createLayerFromCaffe<AbsLayer>(LayerParams&);
template Ptr<Layer> createLayerFromCaffe<BNLLLayer>(LayerParams&);
template Ptr<Layer> createLayerFromCaffe<PowerLayer>(LayerParams&);

template Ptr<Layer> createLayerFromCaffe<FlattenLayer>(LayerParams&);
template Ptr<Layer> createLayerFromCaffe<PermuteLayer>(LayerParams&);
template Ptr<Layer> createLayerFromCaffe<PriorBoxLayer>(LayerParams&);
template Ptr<Layer> createLayerFromCaffe<NormalizeBBoxLayer>(LayerParams&);
template Ptr<Layer> createLayerFromCaffe<DetectionOutputLayer>(LayerParams&);
}
}
