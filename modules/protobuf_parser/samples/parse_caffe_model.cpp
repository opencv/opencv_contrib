// Sample with parsing deep learning model from Caffe framework using OpenCV's
// protobuf parser module.

#include <fstream>
#include <iostream>

#include <opencv2/protobuf_parser.hpp>

using namespace cv;
using namespace pb;

const char* keys =
    "{ help h     |  | print this message }"
    "{ proto      |  | path to compiled .proto file. "
                      "Use protobuf compiler with --descriptor_set_out }"
    "{ caffemodel |  | path to caffemodel }";

void printLayers(const ProtobufNode& layers, bool deprecatedLayers = false);

int main(int argc, char** argv)
{
    //! [Parse arguments]
    CommandLineParser parser(argc, argv, keys);
    if (parser.has("help") || argc == 1)
    {
        parser.printMessage();
        return 0;
    }
    CV_Assert(parser.has("proto"));
    CV_Assert(parser.has("caffemodel"));

    std::string proto = parser.get<std::string>("proto");
    std::string caffemodel = parser.get<std::string>("caffemodel");
    //! [Parse arguments]

    //! [Initialize parser]
    ProtobufParser pp(proto, ".caffe.NetParameter");
    //! [Initialize parser]

    //! [Parse model]
    pp.parse(caffemodel);
    //! [Parse model]

    //! [Print name of network]
    if (pp.has("name"))
    {
        std::cout << "Network: " << (std::string)pp["name"] << std::endl;
    }
    //! [Print name of network]

    //! [Print layers data]
    if (pp.has("layer"))
    {
        printLayers(pp["layer"]);
    }
    else if (pp.has("layers"))
    {
        // For deprecated caffe layers.
        printLayers(pp["layers"], true);
    }
    //! [Print layers data]
}

void printLayers(const ProtobufNode& layers, bool deprecatedLayers)
{
    //! [Get number of layers]
    const int numLayers = (int)layers.size();
    //! [Get number of layers]
    for (int i = 0; i < numLayers; ++i)
    {
        std::string layerName = "no_name", layerType = "no_type";
        //! [Get name of layer]
        if (layers[i].has("name"))
        {
            layers[i]["name"] >> layerName;
        }
        //! [Get name of layer]

        //! [Get type name of layer]
        if (layers[i].has("type"))
        {
            layers[i]["type"] >> layerType;
        }
        //! [Get type name of layer]

        //! [Compute number of parameters]
        int64_t numParameters = 0;
        int num, channels, height, width;
        int numBlobs = (int)layers[i]["blobs"].size();
        for (int j = 0; j < numBlobs; ++j)
        {
            if (deprecatedLayers)
            {
                layers[i]["blobs"][j]["num"] >> num;
                layers[i]["blobs"][j]["channels"] >> channels;
                layers[i]["blobs"][j]["height"] >> height;
                layers[i]["blobs"][j]["width"] >> width;
                numParameters += num * channels * height * width;
            }
            else
            {
                int dims = (int)layers[i]["blobs"][j]["shape"]["dim"].size();
                int64_t total = 1;
                for (int k = 0; k < dims; ++k)
                {
                    total *= (int64_t)layers[i]["blobs"][j]["shape"]["dim"][k];
                }
                numParameters += total;
            }
        }
        //! [Compute number of parameters]

        std::cout << "Layer " << layerName
                  << " of type " << layerType
                  << " with " << numParameters << " parameters" << std::endl;
    }
}
