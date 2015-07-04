#include "../precomp.hpp"
#include "layers_common.hpp"
#include <iostream>
#include <stdlib.h>

namespace cv
{
namespace dnn
{
    class ConcatLayer : public Layer
    {
        int axis;

    public:
        ConcatLayer(LayerParams& params);
        void allocate(const std::vector<Blob*> &inputs, std::vector<Blob> &outputs);
        void forward(std::vector<Blob*> &inputs, std::vector<Blob> &outputs);
    };


    REGISTER_LAYER_CLASS(Concat, ConcatLayer)


    ConcatLayer::ConcatLayer(LayerParams &params)
    {
        axis = params.get<int>("axis", 1);
        CV_Assert(axis == 0 || axis == 1);
    }

    void ConcatLayer::allocate(const std::vector<Blob *> &inputs, std::vector<Blob> &outputs)
    {
        CV_Assert(inputs.size() > 0);

        int axisSum = 0;
        for (size_t i = 0; i < inputs.size(); i++)
        {
            Vec4i refShape = inputs[0]->shape4();
            Vec4i curShape = inputs[i]->shape4();

            for (int axisId = 0; axisId < 4; axisId++)
            {
                if (axisId != axis && refShape[axisId] != curShape[axisId])
                    CV_Error(cv::Error::StsBadArg, "Inconsitent shape for ConcatLayer");
            }

            axisSum += curShape[axis];
        }

        Vec4i shape = inputs[0]->shape4();
        shape[axis] = axisSum;
        outputs.resize(1);
        outputs[0].create(shape);
    }

    void ConcatLayer::forward(std::vector<Blob *> &inputs, std::vector<Blob> &outputs)
    {
        float *dstPtr = outputs[0].ptrf();

        if (axis == 0)
        {
            for (size_t i = 0; i < inputs.size(); i++)
            {
                const float *srcPtr = inputs[i]->ptrf();
                memcpy(dstPtr, srcPtr, inputs[i]->total() * sizeof(float));
                dstPtr += inputs[i]->total();
            }
        }
        else
        {
            for (int n = 0; n < outputs[0].num(); n++)
            {
                for (size_t i = 0; i < inputs.size(); i++)
                {
                    Blob &inp = *inputs[i];
                    memcpy(dstPtr, inp.ptrf(n), inp.total(1) * sizeof(float));
                    dstPtr += inp.total(1);
                }
            }
        }
    }
}
}
