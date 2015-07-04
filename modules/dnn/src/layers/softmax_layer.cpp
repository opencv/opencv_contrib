#include "../precomp.hpp"
#include "layers_common.hpp"
#include <algorithm>
#include <stdlib.h>
using std::max;

namespace cv
{
namespace dnn
{
    //TODO: set default axis number to 1, and add custom shape length in FullyConnected
    class SoftMaxLayer : public Layer
    {
        int axis;
        Blob maxAggregator;

    public:
        SoftMaxLayer(LayerParams &params);
        void allocate(const std::vector<Blob*> &inputs, std::vector<Blob> &outputs);
        void forward(std::vector<Blob*> &inputs, std::vector<Blob> &outputs);
    };


    REGISTER_LAYER_CLASS(Softmax, SoftMaxLayer);


    SoftMaxLayer::SoftMaxLayer(LayerParams &params)
    {
        //hotfix!!!
        axis = params.get<int>("axis", 3);
        CV_Assert(0 <= axis && axis < 4);
    }

    void SoftMaxLayer::allocate(const std::vector<Blob*> &inputs, std::vector<Blob> &outputs)
    {
        CV_Assert(inputs.size() == 1);

        Vec4i shape = inputs[0]->shape4();
        outputs.resize(1);
        outputs[0].create(shape);

        shape[axis] = 1;
        maxAggregator.create(shape);
    }

    void SoftMaxLayer::forward(std::vector<Blob*> &inputs, std::vector<Blob> &outputs)
    {
        Blob &src = *inputs[0];
        Blob &dst = outputs[0];

        float *srcPtr = src.ptrf();
        float *dstPtr = dst.ptrf();
        float *bufPtr = maxAggregator.ptrf();

        size_t outerSize = src.total(0, axis);
        size_t channels = src.size(axis);
        size_t innerSize = src.total(axis + 1, -1);

        size_t outerStep = src.total(axis);
        size_t cnStep = src.total(axis + 1);

        //compute max along axis
        for (size_t outerDim = 0; outerDim < outerSize; outerDim++)
        {
            size_t srcOffset = outerDim * outerStep;
            size_t bufOffset = outerDim * cnStep;

            memcpy(bufPtr + bufOffset, srcPtr + srcOffset, innerSize * sizeof(float));

            for (size_t cnDim = 1; cnDim < channels; cnDim++)
            {
                for (size_t i = 0; i < innerSize; i++)
                    bufPtr[bufOffset + i] = std::max(bufPtr[bufOffset + i], srcPtr[srcOffset + cnDim * cnStep + i]);
            }
        }

        //subtract max
        for (size_t outerDim = 0; outerDim < outerSize; outerDim++)
        {
            size_t srcOffset = outerDim * outerStep;
            size_t bufOffset = outerDim * cnStep;

            for (size_t cnDim = 0; cnDim < channels; cnDim++)
            {
                for (size_t i = 0; i < innerSize; i++)
                    dstPtr[srcOffset + cnDim * cnStep + i] = srcPtr[srcOffset + cnDim * cnStep + i] - bufPtr[bufOffset + i];
            }
        }

        cv::exp(dst.getMat(), dst.getMat());

        for (size_t outerDim = 0; outerDim < outerSize; outerDim++)
        {
            size_t srcOffset = outerDim * outerStep;
            size_t bufOffset = outerDim * cnStep;

            //sum exp along axis
            for (size_t i = 0; i < innerSize; i++)
                bufPtr[bufOffset + i] = 0.f;

            for (size_t cnDim = 0; cnDim < channels; cnDim++)
            {
                for (size_t i = 0; i < innerSize; i++)
                    bufPtr[bufOffset + i] += dstPtr[srcOffset + cnDim * cnStep + i];
            }

            //divide by computed sum
            for (size_t cnDim = 0; cnDim < channels; cnDim++)
            {
                for (size_t i = 0; i < innerSize; i++)
                    dstPtr[srcOffset + cnDim * cnStep + i] /= bufPtr[bufOffset + i];
            }
        }
    }

}
}
