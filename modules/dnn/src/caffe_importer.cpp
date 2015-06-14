#include "opencv2/dnn.hpp"
#include "caffe.pb.h"

#include <iostream>
#include <fstream>
#include <algorithm>
#include <google/protobuf/message.h>
#include <google/protobuf/text_format.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include "caffe/util/upgrade_proto.hpp"

using namespace cv;
using namespace cv::dnn;

using ::google::protobuf::RepeatedField;
using ::google::protobuf::RepeatedPtrField;
using ::google::protobuf::Message;
using ::google::protobuf::Descriptor;
using ::google::protobuf::FieldDescriptor;
using ::google::protobuf::Reflection;

namespace
{
    class CaffeImporter : public Importer
    {
        caffe::NetParameter net;
        caffe::NetParameter netBinary;
 
    public:

        CaffeImporter(const char *pototxt, const char *caffeModel)
        {
            ReadNetParamsFromTextFileOrDie(std::string(pototxt), &net);

            if (caffeModel && caffeModel[0])
                ReadNetParamsFromBinaryFileOrDie(caffeModel, &netBinary);
        }

        inline bool skipCaffeLayerParam(const FieldDescriptor *fd)
        {
            const std::string &name = fd->name();

            if (fd->cpp_type() != FieldDescriptor::CPPTYPE_MESSAGE)
            {
                static const char *SKIP_FIELDS[] = { "type", "name", "top", "bottom", NULL };

                for (int i = 0; SKIP_FIELDS[i]; i++)
                {
                    if (name == SKIP_FIELDS[i])
                        return true;
                }

                return false;
            }
            else
            {
                static const std::string _param("_param");
                bool endsWith_param = (name.size() >= _param.size()) && name.compare(name.size() - _param.size(), _param.size(), _param) == 0;
                return !endsWith_param;
            }
        }

        void addParam(const Message &msg, const FieldDescriptor *field, cv::dnn::LayerParams &params)
        {
            const Reflection *msgRefl = msg.GetReflection();
            int type = field->cpp_type();
            bool isRepeated = field->is_repeated();
            const std::string &name = field->name();

            std::cout << field->type_name() << " " << name << ":";
            
            #define GET_FIRST(Type) (isRepeated ? msgRefl->GetRepeated##Type(msg, field, 0) : msgRefl->Get##Type(msg, field))

            switch (type)
            {
            case FieldDescriptor::CPPTYPE_INT32:
                std::cout << params.set(name, GET_FIRST(Int32));
                break;
            case FieldDescriptor::CPPTYPE_UINT32:
                std::cout << params.set(name, GET_FIRST(UInt32));
                break;
            case FieldDescriptor::CPPTYPE_DOUBLE:
                std::cout << params.set(name, GET_FIRST(Double));
                break;
            case FieldDescriptor::CPPTYPE_FLOAT:
                std::cout << params.set(name, GET_FIRST(Float));
                break;
            case FieldDescriptor::CPPTYPE_ENUM:
                std::cout << params.set(name, GET_FIRST(Enum)->name());
                break;
            case FieldDescriptor::CPPTYPE_BOOL:
                std::cout << params.set(name, GET_FIRST(Bool));
                break;
            default:
                std::cout << "unknown";
                break;
            }

            std::cout << std::endl; 
        }

        void extractLayerParams(const Message &msg, cv::dnn::LayerParams &params)
        {
            const Descriptor *msgDesc = msg.GetDescriptor();
            const Reflection *msgRefl = msg.GetReflection();

            for (int fieldId = 0; fieldId < msgDesc->field_count(); fieldId++)
            {
                const FieldDescriptor *fd = msgDesc->field(fieldId);

                bool hasData =  fd->is_required() || 
                               (fd->is_optional() && (msgRefl->HasField(msg, fd) || fd->has_default_value())) ||
                               (fd->is_repeated() && msgRefl->FieldSize(msg, fd) > 0);

                if ( !hasData || skipCaffeLayerParam(fd) )
                    continue;

                if (fd->cpp_type() == FieldDescriptor::CPPTYPE_MESSAGE)
                {
                    if (fd->is_repeated()) //Extract only first item!
                        extractLayerParams(msgRefl->GetRepeatedMessage(msg, fd, 0), params); 
                    else
                        extractLayerParams(msgRefl->GetMessage(msg, fd), params);
                }
                else
                {
                    addParam(msg, fd, params);
                }
            }
        }

        void blobFromProto(const caffe::BlobProto &protoBlob, cv::dnn::Blob &dstBlob)
        {
            AutoBuffer<int, 4> shape;

            if (protoBlob.has_num() || protoBlob.has_channels() || protoBlob.has_height() || protoBlob.has_width())
            {
                shape.resize(4);
                shape[0] = protoBlob.num();
                shape[1] = protoBlob.channels();
                shape[2] = protoBlob.height();
                shape[3] = protoBlob.width();
            }
            else if (protoBlob.has_shape())
            {
                const caffe::BlobShape &_shape = protoBlob.shape();
                shape.resize(_shape.dim_size());

                for (int i = 0; i < _shape.dim_size(); i++)
                    shape[i] = _shape.dim(i);
            }
            else
            {
                CV_Error(cv::Error::StsAssert, "Unknown shape of input blob");
            }

            dstBlob.create(shape.size(), shape, CV_32F);
            CV_Assert(protoBlob.data_size() == dstBlob.getMatRef().total());

            CV_DbgAssert(protoBlob.GetDescriptor()->FindFieldByLowercaseName("data")->cpp_type() == FieldDescriptor::CPPTYPE_FLOAT);
            float *dstData = dstBlob.getMatRef().ptr<float>();

            for (size_t i = 0; i < protoBlob.data_size(); i++)
                dstData[i] = protoBlob.data(i);
        }

        void extractBinaryLayerParms(const caffe::LayerParameter& layer, LayerParams& layerParams)
        {
            const std::string &name = layer.name();

            int li;
            for (li = 0; li != netBinary.layer_size(); li++)
            {
                if (netBinary.layer(li).name() == name)
                    break;
            }

            if (li == netBinary.layer_size() || netBinary.layer(li).blobs_size() == 0)
                return;

            const caffe::LayerParameter &binLayer = netBinary.layer(li);
            layerParams.learnedBlobs.resize(binLayer.blobs_size());
            for (int bi = 0; bi < binLayer.blobs_size(); bi++)
            {
                blobFromProto(binLayer.blobs(bi), layerParams.learnedBlobs[bi]);
            }
        }

        void populateNet(Net dstNet)
        {
            int layersSize = net.layer_size();

            std::vector<String> layersName(layersSize);
            std::vector<int> layersId(layersSize);
            std::vector<std::vector<String>> bottomsVec(layersSize);

            for (int li = 0; li < layersSize; li++)
            {
                const caffe::LayerParameter layer = net.layer(li);
                String name = layer.name();
                String type = layer.type();
                LayerParams layerParams;

                std::vector<String> tops;
                tops.assign(layer.top().begin(), layer.top().end());
                bottomsVec[li].assign(layer.bottom().begin(), layer.bottom().end());

                std::cout << std::endl << "LAYER: " << name << std::endl;

                extractLayerParams(layer, layerParams);
                extractBinaryLayerParms(layer, layerParams);
                
                int id = dstNet.addLayer(name, type);
                dstNet.setOutputNames(id, tops);

                layersName[li] = name;
                layersId[li] = id;
            }

            for (int li = 0; li < layersSize; li++)
            {
                dstNet.setLayerInputs(bottomsVec[li], layersId[li]);
            }
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