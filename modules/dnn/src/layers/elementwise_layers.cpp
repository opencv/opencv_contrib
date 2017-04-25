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
public:
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

    ElementWiseLayer(bool run_parallel_=false, const Func &f=Func()) : func(f), run_parallel(run_parallel_) {}

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
            CV_Assert(src.type() == CV_32F);
            PBody<float> body(dst, func);
            if( run_parallel )
                cv::parallel_for_(sizeRange, body);
            else
                body(sizeRange);
        }
    }

    Func func;
    bool run_parallel;
};

struct ReLUFunctor
{
    typedef ReLULayer Layer;
    float slope;

    ReLUFunctor(float slope_) : slope(slope_) {}

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

    const float power;
    const float scale;
    const float shift;

    PowerFunctor(float power_, float scale_ = 1.f, float shift_ = 0)
        : power(power_), scale(scale_), shift(shift_) {}

    template<typename TFloat>
    inline TFloat operator()(TFloat x) const
    {
        return pow((TFloat)shift + (TFloat)scale * x, (TFloat)power);
    }
};

struct PowerFunctor1
{
    typedef PowerLayer Layer;

    const float scale;
    const float shift;

    PowerFunctor1(float scale_ = 1.f, float shift_ = 0)
    : scale(scale_), shift(shift_) {}

    template<typename TFloat>
    inline TFloat operator()(TFloat x) const
    {
        return (TFloat)shift + (TFloat)scale * x;
    }
};

class ChannelsPReLULayerImpl : public ChannelsPReLULayer
{
public:
    ChannelsPReLULayerImpl(const LayerParams& params)
    {
        CV_Assert(params.blobs.size() == 1);
        setParamsFrom(params);
    }

    ////////////////////////////////////////////////////////////////////////////

    void allocate(const std::vector<Mat*> &inputs, std::vector<Mat> &outputs)
    {
        CV_Assert(blobs.size() == 1);

        outputs.resize(inputs.size());
        for (size_t i = 0; i < inputs.size(); i++)
        {
            outputs[i].create(inputs[i]->dims, inputs[i]->size.p, inputs[i]->type());
        }
    }

    void forward(std::vector<Mat*> &inputs, std::vector<Mat> &outputs)
    {
        CV_Assert(inputs.size() == 1);
        Mat &inpBlob = *inputs[0];

        for (size_t ii = 0; ii < outputs.size(); ii++)
        {
            Mat &outBlob = outputs[ii];
            CV_Assert(inpBlob.isContinuous() && outBlob.isContinuous());

            CV_Assert(blobs[0].total() == inpBlob.size[1]);

            for (int n = 0; n < inpBlob.size[1]; n++)
            {
                float slopeWeight = blobs[0].at<float>(n);

                Mat inpBlobPlane = getPlane(inpBlob, 0, n);
                Mat outBlobPlane = getPlane(outBlob, 0, n);

                size_t i, planeTotal = inpBlobPlane.total();
                const float* inptr = inpBlobPlane.ptr<float>();
                float* outptr = outBlobPlane.ptr<float>();
                for( i = 0; i < planeTotal; i++ )
                {
                    float val = inptr[i];
                    outptr[i] = val*(val >= 0.f ? 1.f : slopeWeight);
                }
                //threshold(inpBlobPlane, outBlobPlane, 0, 0, cv::THRESH_TOZERO_INV);
                //scaleAdd(outBlobPlane, slopeWeight-1, inpBlobPlane, outBlobPlane);
            }
        }
    }
};

#define ACTIVATION_CREATOR_FOR(_Layer, _Functor, ...) \
Ptr<_Layer> _Layer::create() { \
    return return Ptr<_Layer>( new ElementWiseLayer<_Functor>(_Functor()) ); }


Ptr<ReLULayer> ReLULayer::create(const LayerParams& params)
{
    float negativeSlope = params.get<float>("negative_slope", 0.f);
    Ptr<ReLULayer> l(new ElementWiseLayer<ReLUFunctor>(false, ReLUFunctor(negativeSlope)));
    l->setParamsFrom(params);

    return l;
}

Ptr<TanHLayer> TanHLayer::create(const LayerParams& params)
{
    Ptr<TanHLayer> l(new ElementWiseLayer<TanHFunctor>(true));
    l->setParamsFrom(params);

    return l;
}

Ptr<SigmoidLayer> SigmoidLayer::create(const LayerParams& params)
{
    Ptr<SigmoidLayer> l(new ElementWiseLayer<SigmoidFunctor>(true));
    l->setParamsFrom(params);

    return l;
}

Ptr<AbsLayer> AbsLayer::create(const LayerParams& params)
{
    Ptr<AbsLayer> l(new ElementWiseLayer<AbsValFunctor>());
    l->setParamsFrom(params);

    return l;
}

Ptr<BNLLLayer> BNLLLayer::create(const LayerParams& params)
{
    Ptr<BNLLLayer> l(new ElementWiseLayer<BNLLFunctor>(true));
    l->setParamsFrom(params);

    return l;
}

Ptr<PowerLayer> PowerLayer::create(const LayerParams& params)
{
    float power = params.get<float>("power", 1.0f);
    float scale = params.get<float>("scale", 1.0f);
    float shift = params.get<float>("shift", 0.0f);
    Ptr<PowerLayer> l(power == 1.f ?
                      (PowerLayer*)(new ElementWiseLayer<PowerFunctor1>(false, PowerFunctor1(scale, shift))) :
                      (PowerLayer*)(new ElementWiseLayer<PowerFunctor>(true, PowerFunctor(power, scale, shift))));
    l->setParamsFrom(params);

    return l;
}


Ptr<ChannelsPReLULayer> ChannelsPReLULayer::create(const LayerParams& params)
{
    return Ptr<ChannelsPReLULayer>(new ChannelsPReLULayerImpl(params));
}

}
}
