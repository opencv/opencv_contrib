#include "../precomp.hpp"
#include "layers_common.hpp"
#include "mvn_layer.hpp"

namespace cv
{
namespace dnn
{

MVNLayer::MVNLayer(LayerParams &params)
{
    eps = params.get<double>("eps", 1e-9);
    acrossChannels = params.get<bool>("across_channels", false);
    normalizeVariance = params.get<bool>("normalize_variance", true);
}

void MVNLayer::allocate(const std::vector<Blob *> &inputs, std::vector<Blob> &outputs)
{
    outputs.resize(inputs.size());
    for (size_t i = 0; i < inputs.size(); i++)
    {
        CV_Assert(!acrossChannels || inputs[i]->dims() >= 2);
        outputs[i].create(inputs[i]->shape(), inputs[i]->type());
    }
}

void MVNLayer::forward(std::vector<Blob *> &inputs, std::vector<Blob> &outputs)
{
    for (size_t inpIdx = 0; inpIdx < inputs.size(); inpIdx++)
    {
        Blob &inpBlob = *inputs[inpIdx];
        Blob &outBlob = outputs[inpIdx];

        int workSize[2];
        int splitDim = (acrossChannels) ? 1 : 2;
        workSize[0] = (int)inpBlob.total(0, splitDim);
        workSize[1] = (int)inpBlob.total(splitDim);

        Mat inpMat = inpBlob.getMatRef().reshape(1, 2, workSize);
        Mat outMat = outBlob.getMatRef().reshape(1, 2, workSize);

        Scalar mean, dev;
        for (int i = 0; i < workSize[0]; i++)
        {
            Mat inpRow = inpMat.row(i);
            Mat outRow = outMat.row(i);

            cv::meanStdDev(inpRow, mean, (normalizeVariance) ? dev : noArray());
            double alpha = (normalizeVariance) ? 1/(eps + dev[0]) : 1;
            inpRow.convertTo(outRow, outRow.type(), alpha, -mean[0] * alpha);
        }
    }
}

}
}
