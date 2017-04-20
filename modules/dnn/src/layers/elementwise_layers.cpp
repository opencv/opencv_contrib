#include "../precomp.hpp"
#include "opencv2/imgproc.hpp"
#include <opencv2/dnn/shape_utils.hpp>

namespace cv
{
namespace dnn
{
    
using std::abs;
using std::exp;
using std::tanh;
using std::pow;

template<typename Func>
class ElementWiseLayer : public Func::Layer
{
    Func func;

    template<typename Dtype>
    class PBody : public cv::ParallelLoopBody
    {
        Func &func;
        Dtype *data;
    public:

        PBody(Mat &mat, Func &func_) :
            func(func_), data(mat.ptr<Dtype>())
        {}

        void operator()(const Range &r) const
        {
            for (int i = r.start; i < r.end; i++)
                data[i] = func(data[i]);
        }
    };

public:

    ElementWiseLayer() {}
    ElementWiseLayer(const Func &f) : func(f) {}

    void allocate(const std::vector<Mat*> &inputs, std::vector<Mat> &outputs)
    {
        outputs.resize(inputs.size());
        for (size_t i = 0; i < inputs.size(); i++)
        {
            outputs[i] = *inputs[i];
        }
    }

    void forward(std::vector<Mat*> &inputs, std::vector<Mat> &outputs)
    {
        for (size_t i = 0; i < inputs.size(); i++)
        {
            const Mat &src = *inputs[i];
            Mat &dst = outputs[i];
            CV_Assert(src.ptr() == dst.ptr() && src.isContinuous());

            Range sizeRange = Range(0, dst.total());
            if (dst.type() == CV_32F)
            {
                cv::parallel_for_(sizeRange, PBody<float>(dst, func));
            }
            else if (dst.type() == CV_64F)
            {
                cv::parallel_for_(sizeRange, PBody<double>(dst, func));
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
    typedef ReLULayer Layer;
    double slope;

    ReLUFunctor(double slope_)
        : slope(slope_) {}

    template<typename TFloat>
    inline TFloat operator()(TFloat x) const
    {
        return (x >= (TFloat)0) ? x : (TFloat)slope * x;
    }
};

struct TanHFunctor
{
    typedef TanHLayer Layer;

    template<typename TFloat>
    inline TFloat operator()(TFloat x) const
    {
        return tanh(x);
    }
};

struct SigmoidFunctor
{
    typedef SigmoidLayer Layer;

    template<typename TFloat>
    inline TFloat operator()(TFloat x) const
    {
        return (TFloat)1 / ((TFloat)1 + exp(-x));
    }
};

struct AbsValFunctor
{
    typedef AbsLayer Layer;

    template<typename TFloat>
    inline TFloat operator()(TFloat x) const
    {
        return abs(x);
    }
};

struct BNLLFunctor
{
    typedef BNLLLayer Layer;

    template<typename TFloat>
    inline TFloat operator()(TFloat x) const
    {
        return log((TFloat)1 + exp(-abs(x)));
    }
};

struct PowerFunctor
{
    typedef PowerLayer Layer;

    const double power;
    const double scale;
    const double shift;

    PowerFunctor(double power_, double scale_ = 1, double shift_ = 0)
        : power(power_), scale(scale_), shift(shift_) {}

    template<typename TFloat>
    inline TFloat operator()(TFloat x) const
    {
        return power == 1.0 ? (TFloat)shift + (TFloat)scale * x : pow((TFloat)shift + (TFloat)scale * x, (TFloat)power);
    }
};

class ChannelsPReLULayerImpl : public ChannelsPReLULayer
{
public:
    ChannelsPReLULayerImpl() {}

    void allocate(const std::vector<Mat*> &inputs, std::vector<Mat> &outputs);

    void forward(std::vector<Mat*> &inputs, std::vector<Mat> &outputs);
};    

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

void ChannelsPReLULayerImpl::allocate(const std::vector<Mat*> &inputs, std::vector<Mat> &outputs)
{
    CV_Assert(blobs.size() == 1);

    outputs.resize(inputs.size());
    for (size_t i = 0; i < inputs.size(); i++)
    {
        outputs[i].create(inputs[i]->dims, inputs[i]->size.p, inputs[i]->type());
    }
}

void ChannelsPReLULayerImpl::forward(std::vector<Mat*> &inputs, std::vector<Mat> &outputs)
{
    CV_Assert(inputs.size() == 1);

    Mat &inpBlob = *inputs[0];

    for (size_t ii = 0; ii < outputs.size(); ii++)
    {
        Mat &outBlob = outputs[ii];

        CV_Assert(blobs[0].total() == inpBlob.size[1]);

        for (int n = 0; n < inpBlob.size[1]; n++)
        {
            float slopeWeight = blobs[0].at<float>(n);

            Mat inpBlobPlane = getPlane(inpBlob, 0, n);
            Mat outBlobPlane = getPlane(outBlob, 0, n);

            threshold(inpBlobPlane, outBlobPlane, 0, 0, cv::THRESH_TOZERO_INV);
            scaleAdd(outBlobPlane, slopeWeight-1, inpBlobPlane, outBlobPlane);
        }
    }
}

Ptr<ChannelsPReLULayer> ChannelsPReLULayer::create()
{
    return Ptr<ChannelsPReLULayer>(new ChannelsPReLULayerImpl());
}

}
}
