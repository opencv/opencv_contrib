#include "../precomp.hpp"
#include "layers_common.hpp"
#include <float.h>
#include <algorithm>
using std::max;
using std::min;

namespace cv
{
namespace dnn
{
//TODO: add ceil_mode param
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

        int inpH, inpW;
        int outH, outW;

        void computeOutputShape(int inpH, int inpW);
        void maxPooling(Blob &input, Blob &output);
        void avePooling(Blob &input, Blob &output);

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

        inpW = inputs[0]->cols();
        inpH = inputs[0]->rows();
        computeOutputShape(inpH, inpW);

        outputs.resize(inputs.size());
        for (size_t i = 0; i < inputs.size(); i++)
        {
            CV_Assert(inputs[i]->rows() == inpH && inputs[i]->cols() == inpW);
            outputs[i].create(BlobShape(inputs[i]->num(), inputs[i]->channels(), outH, outW));
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
            case AVE:
                avePooling(*inputs[ii], outputs[ii]);
                break;
            default:
                CV_Error(cv::Error::StsNotImplemented, "Not implemented");
                break;
            }
        }
    }

    void PoolingLayer::maxPooling(Blob &input, Blob &output)
    {
        CV_DbgAssert(output.rows() == outH && output.cols() == outW);

        for (int n = 0; n < input.num(); ++n)
        {
            for (int c = 0; c < input.channels(); ++c)
            {
                float *srcData = input.ptrf(n, c);
                float *dstData = output.ptrf(n, c);

                for (int ph = 0; ph < outH; ++ph)
                {
                    for (int pw = 0; pw < outW; ++pw)
                    {
                        int hstart = ph * strideH - padH;
                        int wstart = pw * strideW - padW;
                        int hend = min(hstart + kernelH, inpH);
                        int wend = min(wstart + kernelW, inpW);
                        hstart = max(hstart, 0);
                        wstart = max(wstart, 0);
                        const int poolIndex = ph * outW + pw;
                        float max_val = -FLT_MAX;

                        for (int h = hstart; h < hend; ++h)
                            for (int w = wstart; w < wend; ++w)
                            {
                                const int index = h * inpW + w;
                                if (srcData[index] > max_val)
                                    max_val = srcData[index];
                            }

                        dstData[poolIndex] = max_val;
                    }
                }
            }
        }
    }

    void PoolingLayer::avePooling(Blob &input, Blob &output)
    {
        for (int n = 0; n < input.num(); ++n)
        {
            for (int c = 0; c < input.channels(); ++c)
            {
                float *srcData = input.ptrf(n, c);
                float *dstData = output.ptrf(n, c);

                for (int ph = 0; ph < outH; ++ph)
                {
                    for (int pw = 0; pw < outW; ++pw)
                    {
                        int hstart = ph * strideH - padH;
                        int wstart = pw * strideW - padW;
                        int hend = min(hstart + kernelH, inpH + padH);
                        int wend = min(wstart + kernelW, inpW + padW);
                        int poolSize = (hend - hstart) * (wend - wstart);
                        hstart = max(hstart, 0);
                        wstart = max(wstart, 0);
                        hend = min(hend, inpH);
                        wend = min(wend, inpW);

                        dstData[ph * outW + pw] = 0.f;

                        for (int h = hstart; h < hend; ++h)
                            for (int w = wstart; w < wend; ++w)
                                dstData[ph * outW + pw] += srcData[h * inpW + w];

                        dstData[ph * outW + pw] /= poolSize;
                    }
                }
          }
        }
    }

    void PoolingLayer::computeOutputShape(int inH, int inW)
    {
        //Yeah, something strange Caffe scheme-)
        outH = static_cast<int>(ceil(static_cast<float>(inH + 2 * padH - kernelH) / strideH)) + 1;
        outW = static_cast<int>(ceil(static_cast<float>(inW + 2 * padW - kernelW) / strideW)) + 1;

        if (padH || padW)
        {
            // If we have padding, ensure that the last pooling starts strictly
            // inside the image (instead of at the padding); otherwise clip the last.
            if ((outH - 1) * strideH >= inH + padH)
                --outH;
            if ((outW - 1) * strideW >= inW + padW)
                --outW;
            CV_Assert((outH - 1) * strideH < inH + padH);
            CV_Assert((outW - 1) * strideW < inW + padW);
        }
    }
}
}
