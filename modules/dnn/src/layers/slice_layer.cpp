#include "../precomp.hpp"
#include "layers_common.hpp"
#include "slice_layer.hpp"

namespace cv
{
namespace dnn
{

SliceLayer::SliceLayer(LayerParams &params) : Layer(params)
{
    inAxis = params.get<int>("axis", 1);

    if (!params.has("slice_point"))
        return;

    const DictValue &_slicePoints = params.get("slice_point");
    slicePoints.resize(_slicePoints.size());
    for (int i = 0; i < _slicePoints.size(); i++)
    {
        slicePoints[i] = _slicePoints.get<int>(i);
        CV_Assert(slicePoints[i] > 0 && (i == 0 || slicePoints[i-1] < slicePoints[i]));
    }
}

void SliceLayer::allocate(const std::vector<Blob*> &inputs, std::vector<Blob> &outputs)
{
    CV_Assert(inputs.size() == 1);

    const Blob inpBlob = *inputs[0];
    int axis = inpBlob.canonicalAxis(inAxis);
    int axisSize = inpBlob.size(axis);
    BlobShape inpShape = inpBlob.shape();

    if (slicePoints.size()) //divide blob with respect to passed parameters
    {
        std::vector<int> outAxisSize;
        int prevSlice = 0;

        for (size_t i = 0; i < slicePoints.size(); i++)
        {
            CV_Assert(prevSlice < slicePoints[i] && slicePoints[i] < axisSize);
            outAxisSize.push_back(slicePoints[i] - prevSlice);
            prevSlice = slicePoints[i];
        }
        outAxisSize.push_back(axisSize - prevSlice);

        outputs.resize(outAxisSize.size());
        for (size_t i = 0; i < outAxisSize.size(); i++)
        {
            inpShape[axis] = outAxisSize[i];
            outputs[i].create(inpShape, inpBlob.type());
        }
    }
    else //divide blob with respect to count of output blobs
    {
        CV_Assert(outputs.size() > 0 && axisSize % outputs.size() == 0);
        int outAxisSize = axisSize / (int)outputs.size();

        for (size_t i = 0; i < outputs.size(); i++)
        {
            inpShape[axis] = outAxisSize;
            outputs[i].create(inpShape, inpBlob.type());
        }
    }
}

void SliceLayer::forward(std::vector<Blob*> &inputs, std::vector<Blob> &outputs)
{
    Blob &inpBlob = *inputs[0];
    const int axis = inpBlob.canonicalAxis(inAxis);
    const Mat& inpMat = inpBlob.getMatRef();

    std::vector<Range> ranges(inpBlob.dims(), Range::all());
    int sizeStart = 0;
    for (size_t i = 0; i < outputs.size(); i++)
    {
        int sizeEnd = sizeStart + outputs[i].size(axis);
        ranges[axis] = Range(sizeStart, sizeEnd);

        Mat inpSubMat = inpMat(&ranges[0]);
        inpSubMat.copyTo(outputs[i].getMatRef());

        sizeStart = sizeEnd;
    }
}

}
}
