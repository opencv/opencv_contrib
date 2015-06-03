#include "opencv2/dnn.hpp"
#include "caffe.pb.h"

#include <iostream>
#include <fstream>
#include <google/protobuf/message.h>
#include <google/protobuf/text_format.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>

using namespace cv;
using namespace cv::dnn;

namespace
{

    class CaffeImporter : public Importer
    {

    public:

        CaffeImporter(const char *pototxt, const char *caffeModel)
        {
            std::ifstream proto_ifs(pototxt, std::ifstream::in);
            std::ifstream model_ifs(caffeModel, std::ifstream::in);

            CV_Assert(proto_ifs.is_open() && model_ifs.is_open());

            google::protobuf::io::IstreamInputStream proto_zcs(&proto_ifs);
            google::protobuf::io::IstreamInputStream model_zcs(&model_ifs);

            //google::protobuf::Message msg_weights;
            caffe::NetParameter msg_arch;

            CV_Assert(google::protobuf::TextFormat::Parse(&proto_zcs, &msg_arch));
            //CV_Assert( msg_weights.ParseFromZeroCopyStream(model_zcs) );

            const google::protobuf::Descriptor *desc_arch = msg_arch.GetDescriptor();
            CV_Assert(desc_arch);
        }

        void populateNetConfiguration(Ptr<NetConfiguration> config)
        {

        }

        ~CaffeImporter()
        {

        }
    };

}

Ptr<Importer> cv::dnn::createCaffeImporter(const String &prototxt, const String &caffeModel)
{
    return Ptr<Importer>(new CaffeImporter(prototxt.c_str(), caffeModel.c_str()));
}