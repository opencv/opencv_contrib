#include "../precomp.hpp"
#include "layers_common.hpp"

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
        CV_Assert(axis >= 0);
    }

    void ConcatLayer::allocate(const std::vector<Blob *> &inputs, std::vector<Blob> &outputs)
    {
        CV_Assert(inputs.size() > 0);
        
        int refType = inputs[0]->type();
        BlobShape refShape = inputs[0]->shape();
        CV_Assert(axis < refShape.dims());

        int axisSum = 0;
        for (size_t i = 0; i < inputs.size(); i++)
        {
            BlobShape curShape = inputs[i]->shape();

            CV_Assert(curShape.dims() == refShape.dims() && inputs[i]->type() == refType);
            for (int axisId = 0; axisId < refShape.dims(); axisId++)
            {
                if (axisId != axis && refShape[axisId] != curShape[axisId])
                    CV_Error(Error::StsBadSize, "Inconsitent shape for ConcatLayer");
            }

            axisSum += curShape[axis];
        }

        refShape[axis] = axisSum;
        outputs.resize(1);
        outputs[0].create(refShape);
    }

    void ConcatLayer::forward(std::vector<Blob *> &inputs, std::vector<Blob> &outputs)
    {
        const Mat& outMat = outputs[0].getMatRef();
        std::vector<Range> ranges(outputs[0].dims(), Range::all());
        int sizeStart = 0;
        for (size_t i = 0; i < inputs.size(); i++)
        {
            int sizeEnd = sizeStart + inputs[i]->size(axis);
            ranges[axis] = Range(sizeStart, sizeEnd);

            Mat outSubMat = outMat(&ranges[0]);
            inputs[i]->getMatRef().copyTo(outSubMat);

            sizeStart = sizeEnd;
        }
    }
}
}
