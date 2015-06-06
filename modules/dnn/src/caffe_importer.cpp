#include "opencv2/dnn.hpp"
#include "caffe.pb.h"

#include <iostream>
#include <fstream>
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

    void walk(const Descriptor *desc)
    {
        if (desc == NULL)
            return;

        std::cout << "* " << desc->full_name() << std::endl;

        for (int i = 0; i < desc->field_count(); i++)
        {
            const FieldDescriptor *fdesc = desc->field(i);

            if (fdesc->message_type())
                walk(fdesc->message_type());
            else;
                //std::cout << "f " << desc->field(i)->full_name() << std::endl;
        }
    }

    class CaffeImporter : public Importer
    {
        caffe::NetParameter net;
        cv::dnn::LayerParams params;

    public:

        CaffeImporter(const char *pototxt, const char *caffeModel)
        {
            ReadNetParamsFromTextFileOrDie(std::string(pototxt), &net);
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

        void populateNetConfiguration(Ptr<NetConfiguration> config)
        {
            const Descriptor *layerDescriptor = caffe::LayerParameter::descriptor();

            for (int li = 0; li < net.layer_size(); li++)
            {
                const caffe::LayerParameter layer = net.layer(li);
                String name = layer.name();
                String type = layer.type();

                std::vector<String> bottoms, tops;
                bottoms.assign(layer.bottom().begin(), layer.bottom().end());
                tops.assign(layer.top().begin(), layer.top().end());

                std::cout << std::endl << "LAYER: " << name << std::endl;

                cv::dnn::LayerParams params;
                extractLayerParams(layer, params);

                //SetUp
                //int id = config->addLayer(name, type);
                //config->setLayerOutputLabels(id, bottoms);
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