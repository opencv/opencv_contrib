#include "../precomp.hpp"
#include "layers_common.hpp"
#include <opencv2/imgproc.hpp>

namespace cv
{
namespace dnn
{
    class LRNLayer : public Layer
    {
        enum
        {
            CHANNEL_NRM,
            SPATIAL_NRM,
            SPATIAL_CONTRAST_NRM //cuda-convnet feature
        } type;

        int size;
        double alpha, beta;

        Blob bufBlob;

        void channelNoramlization(Blob &src, Blob &dst);
        void spatialNormalization(Blob &src, Blob &dst);

    public:
        
        LRNLayer(LayerParams &params);
        void allocate(const std::vector<Blob*> &inputs, std::vector<Blob> &outputs);
        void forward(std::vector<Blob*> &inputs, std::vector<Blob> &outputs);
    };


    REGISTER_LAYER_CLASS(LRN, LRNLayer)


    LRNLayer::LRNLayer(LayerParams &params)
    {
        String nrmType = params.get<String>("norm_region", "ACROSS_CHANNELS");
        if (nrmType == "ACROSS_CHANNELS")
            type = CHANNEL_NRM;
        else if (nrmType == "WITHIN_CHANNEL")
            type = SPATIAL_NRM;
        else
            CV_Error(cv::Error::StsBadArg, "Unknown region type \"" + nrmType + "\"");

        size = params.get<int>("local_size", 5);
        if (size % 2 != 1)
            CV_Error(cv::Error::StsBadArg, "LRN layer only supports odd values for local_size");

        alpha = params.get<double>("alpha", 1);
        beta = params.get<double>("beta", 0.75);
    }

    void LRNLayer::allocate(const std::vector<Blob*> &inputs, std::vector<Blob> &outputs)
    {
        CV_Assert(inputs.size() == 1);
        outputs.resize(1);

        Vec4i shape = inputs[0]->shape();
        outputs[0].create(shape);

        shape[1] = 1; //maybe make shape[0] = 1 too
        bufBlob.create(shape);
    }

    void LRNLayer::forward(std::vector<Blob*> &inputs, std::vector<Blob> &outputs)
    {
        Blob &src = *inputs[0];
        Blob &dst = outputs[0];

        switch (type)
        {
        case CHANNEL_NRM:
            channelNoramlization(src, dst);
            break;
        case SPATIAL_NRM:
            spatialNormalization(src, dst);
            break;
        default:
            CV_Error(cv::Error::StsNotImplemented, "Unimplemented mode of LRN layer");
            break;
        }
    }

    void LRNLayer::channelNoramlization(Blob &srcBlob, Blob &dstBlob)
    {
        int num = srcBlob.num();
        int channels = srcBlob.channels();

        for (int n = 0; n < num; n++)
        {
            Mat buf = bufBlob.getMat(n, 0);
            Mat accum = dstBlob.getMat(n, 0); //memory saving
            accum.setTo(0);

            for (int cn = 0; cn < channels; cn++)
            {
                cv::accumulateSquare(srcBlob.getMat(n, cn), accum);
            }

            accum.convertTo(accum, accum.type(), alpha/channels, 1);
            cv::pow(accum, beta, accum);
            
            for (int cn = channels - 1; cn >= 0; cn--)
            {
                cv::divide(srcBlob.getMat(n, cn), accum, dstBlob.getMat(n, cn));
            }
        }
    }

    void LRNLayer::spatialNormalization(Blob &srcBlob, Blob &dstBlob)
    {
        int num = srcBlob.num();
        int channels = srcBlob.channels();

        for (int n = 0; n < num; n++)
        {
            for (int cn = 0; cn < channels; cn++)
            {
                Mat src = srcBlob.getMat(n, cn);
                Mat dst = dstBlob.getMat(n, cn);
                uchar *dataDst0 = dst.data;

                cv::pow(srcBlob.getMat(n, cn), 2, dst);
                //TODO: check border type
                cv::boxFilter(dst, dst, dst.depth(), cv::Size(size, size), cv::Point(-1, -1), false, cv::BORDER_CONSTANT);
                dst.convertTo(dst, dst.type(), alpha/(size*size), 1);
                cv::pow(dst, beta, dst);
                cv::divide(src, dst, dst);

                CV_Assert(dataDst0 == dst.data); //debug
            }
        }
    }

}
}