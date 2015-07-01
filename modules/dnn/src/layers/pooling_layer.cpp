#include "../precomp.hpp"
#include "layers_common.hpp"
#include <float.h>
#include <algorithm>
using std::max;

namespace cv
{
namespace dnn
{
    class PoolingLayer : public Layer
    {
        enum
        {
            MAX,
            AVE,
            STOCHASTIC
        };

        int type;
        int padH, padW;
        int strideH, strideW;
        int kernelH, kernelW;

        int inH, inW;
        int pooledH, pooledW;

        void computeOutputShape(int inH, int inW);
        void maxPooling(Blob &input, Blob &output);

    public:
        PoolingLayer(LayerParams &params);
        void allocate(const std::vector<Blob*> &inputs, std::vector<Blob> &outputs);
        void forward(std::vector<Blob*> &inputs, std::vector<Blob> &outputs);
    };


    REGISTER_LAYER_CLASS(Pooling, PoolingLayer)


    PoolingLayer::PoolingLayer(LayerParams &params)
    {
        if (params.has("pool"))
        {
            String pool = params.get<String>("pool").toLowerCase();
            if (pool == "max")
                type = MAX;
            else if (pool == "ave")
                type = AVE;
            else if (pool == "stochastic")
                type = STOCHASTIC;
            else
                CV_Error(cv::Error::StsBadArg, "Unknown pooling type \"" + pool + "\"");
        }
        else
        {
            type = MAX;
        }

        getKernelParams(params, kernelH, kernelW, padH, padW, strideH, strideW);
    }

    void PoolingLayer::allocate(const std::vector<Blob*> &inputs, std::vector<Blob> &outputs)
    {
        CV_Assert(inputs.size() > 0);

        inW = inputs[0]->cols();
        inH = inputs[0]->rows();
        computeOutputShape(inH, inW);

        outputs.resize(inputs.size());
        for (size_t i = 0; i < inputs.size(); i++)
        {
            CV_Assert(inputs[i]->rows() == inH && inputs[i]->cols() == inW);
            outputs[i].create(inputs[i]->num(), inputs[i]->channels(), pooledH, pooledW);
        }
    }

    void PoolingLayer::forward(std::vector<Blob*> &inputs, std::vector<Blob> &outputs)
    {
        for (size_t ii = 0; ii < inputs.size(); ii++)
        {
            switch (type)
            {
            case MAX:
                maxPooling(*inputs[ii], outputs[ii]);
                break;
            default:
                CV_Error(cv::Error::StsNotImplemented, "Not implemented");
                break;
            }
        }
    }

    void PoolingLayer::maxPooling(Blob &input, Blob &output)
    {
        CV_DbgAssert(output.rows() == pooledH && output.cols() == pooledW);

        for (int n = 0; n < input.num(); ++n)
        {
            for (int c = 0; c < input.channels(); ++c)
            {
                float *srcData = input.ptr<float>(n, c);
                float *dstData = output.ptr<float>(n, c);

                for (int ph = 0; ph < pooledH; ++ph)
                {
                    for (int pw = 0; pw < pooledW; ++pw)
                    {
                        int hstart = ph * strideH - padH;
                        int wstart = pw * strideW - padW;
                        int hend = min(hstart + kernelH, inH);
                        int wend = min(wstart + kernelW, inW);
                        hstart = max(hstart, 0);
                        wstart = max(wstart, 0);
                        const int pool_index = ph * pooledW + pw;
                        float max_val = -FLT_MAX;

                        for (int h = hstart; h < hend; ++h)
                            for (int w = wstart; w < wend; ++w)
                            {
                                const int index = h * inW + w;
                                if (srcData[index] > max_val)
                                    max_val = srcData[index];
                            }

                        dstData[pool_index] = max_val;
                    }
                }
            }
        }
    }

    void PoolingLayer::computeOutputShape(int inH, int inW)
    {
        //Yeah, something strange Caffe scheme-)
        pooledH = static_cast<int>(ceil(static_cast<float>(inH + 2 * padH - kernelH) / strideH)) + 1;
        pooledW = static_cast<int>(ceil(static_cast<float>(inW + 2 * padW - kernelW) / strideW)) + 1;

        if (padH || padW)
        {
            // If we have padding, ensure that the last pooling starts strictly
            // inside the image (instead of at the padding); otherwise clip the last.
            if ((pooledH - 1) * strideH >= inH + padH)
                --pooledH;
            if ((pooledW - 1) * strideW >= inW + padW)
                --pooledW;
            CV_Assert((pooledH - 1) * strideH < inH + padH);
            CV_Assert((pooledW - 1) * strideW < inW + padW);
        }
    }
}
}
