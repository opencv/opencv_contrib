#include "../precomp.hpp"
#include "layers_common.hpp"
#include "reshape_layer.hpp"

namespace cv
{
namespace dnn
{

ReshapeLayer::ReshapeLayer(LayerParams &params) : Layer(params)
{
    inAxis = params.get<int>("axis", 0);
    inNumAxes = params.get<int>("num_axes", -1);
    CV_Assert(inNumAxes >= -1);

    autoAxisIdx = -1;

    if (!params.has("dim"))
    {
        shapeDesc = BlobShape(0);
        return;
    }

    DictValue paramShape = params.get("dim");
    shapeDesc = BlobShape(paramShape.size());

    for (int i = 0; i < paramShape.size(); i++)
    {
        int dim = paramShape.get<int>(i);
        CV_Assert(dim >= -1);

        if (dim == -1)
        {
            if (autoAxisIdx != -1)
                CV_Error(Error::StsBadArg, "New shape contains multiple -1 dims");
            autoAxisIdx = i;
        }

        shapeDesc[i] = dim;
    }
}

void ReshapeLayer::allocate(const std::vector<Blob*> &inputs, std::vector<Blob> &outputs)
{
    outputs.resize(inputs.size());

    for (size_t i = 0; i < inputs.size(); i++)
    {
        Blob &inpBlob = *inputs[i];
        Blob &outBlob = outputs[i];
        BlobShape inpShape = inpBlob.shape();

        int startAxis = (inAxis >= 0) ? inAxis : inpShape.dims() + 1 + inAxis;
        int endAxis = (inNumAxes == -1) ? inpShape.dims() : startAxis + inNumAxes;
        CV_Assert(0 <= startAxis && startAxis <= inpShape.dims());
        CV_Assert(0 <= endAxis && endAxis <= inpShape.dims());

        int newDims = inpShape.dims() - (endAxis - startAxis) + shapeDesc.dims();
        BlobShape outShape(newDims);

        computeOutputShape(startAxis, endAxis, inpShape, outShape);

        outBlob.shareFrom(inpBlob);
        outBlob.reshape(outShape);
    }
}

void ReshapeLayer::computeOutputShape(int startAxis, int endAxis, BlobShape &inpShape, BlobShape &outShape)
{
    int idx = 0;
    for (int i = 0; i < startAxis; i++)
        outShape[idx++] = inpShape[i];

    for (int i = 0; i < shapeDesc.dims(); i++)
    {
        if (shapeDesc[i] == 0)
        {
            int inpAxisIdx = startAxis + i;
            if (inpAxisIdx < 0 || inpShape.dims() <= inpAxisIdx)
                CV_Error(Error::StsOutOfRange, "copy dimension (which has zero size) is not presented into reshaped blob");
            outShape[idx++] = inpShape[startAxis + i];
        }
        else
        {
            outShape[idx++] = (shapeDesc[i] > 0) ? shapeDesc[i] : 1;
        }
    }

    for (int i = endAxis; i < inpShape.dims(); i++)
        outShape[idx++] = inpShape[i];

    if (autoAxisIdx >= 0)
    {
        size_t total = inpShape.total();
        size_t curTotal = 1;
        for (int i = 0; i < outShape.dims(); i++)
        {
            if (i != startAxis + autoAxisIdx)
                curTotal *= outShape[i];
        }

        CV_DbgAssert(curTotal <= total && total % curTotal == 0);

        outShape[startAxis + autoAxisIdx] = (int)(total / curTotal);
    }

    if (inpShape.total() != outShape.total())
    {
        CV_Error(Error::StsUnmatchedSizes, "Mismatch between input and output blob elements count");
    }
}


Ptr<Layer> createFlattenLayer(LayerParams&)
{
    LayerParams params;

    int shapeDesc[] = {0, -1};
    params.set("dim", DictValue::arrayInt(shapeDesc, 2));

    return Ptr<Layer>(new ReshapeLayer(params));
}

}
}
