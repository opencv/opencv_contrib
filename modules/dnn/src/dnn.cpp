#include "precomp.hpp"
#include <set>
#include <algorithm>
#include <iostream>

using namespace cv;
using namespace cv::dnn;

using std::vector;
using std::map;
using std::make_pair;
using std::set;

namespace cv
{
namespace dnn
{

Blob::Blob()
{
    int zeros[4] = {0, 0, 0, 0};
    m = Mat(4, zeros, CV_32F, NULL);
}

Blob::Blob(InputArray in)
{
    CV_Assert(in.isMat());
    m = in.getMat();
}

static Vec4i blobNormalizeShape(int ndims, const int *sizes)
{
    Vec4i shape = Vec4i::all(1);

    for (int i = 0; i < std::min(3, ndims); i++)
        shape[3 - i] = sizes[ndims-1 - i];

    for (int i = 3; i < ndims; i++)
        shape[0] *= sizes[ndims-1 - i];

    return shape;
}

void Blob::fill(int ndims, const int *sizes, int type, void *data, bool deepCopy)
{
    CV_Assert(type == CV_32F || type == CV_64F);

    Vec4i shape = blobNormalizeShape(ndims, sizes);

    if (deepCopy)
    {
        m.create(3, &shape[0], type);
        size_t dataSize = m.total() * m.elemSize();
        memcpy(m.data, data, dataSize);
    }
    else
    {
        m = Mat(shape.channels, &shape[0], type, data);
    }
}

void Blob::fill(InputArray in)
{
    CV_Assert(in.isMat() || in.isMatVector());
}

void Blob::create(int ndims, const int *sizes, int type /*= CV_32F*/)
{
    Vec4i shape = blobNormalizeShape(ndims, sizes);
    m.create(shape.channels, &shape[0], type);
}

//////////////////////////////////////////////////////////////////////////

struct LayerOutId
{
    int lid;
    int oid;
    String name;

    LayerOutId() {}
    LayerOutId(int layerId, int outputId, const String &outputName = String()) 
        : lid(layerId), oid(outputId), name(outputName) {}

    struct UnaryMatchName
    {   
        const String &name;
        UnaryMatchName(const String &_name) : name(_name) {}
        bool operator()(const String &other) { return name == other; }
    };
};

struct LayerData
{
    LayerData() {}
    LayerData(const String &_name, const String &_type, LayerParams &_params = LayerParams())
        : name(_name), type(_type), params(_params)
    {}

    String name;
    String type;
    LayerParams params;

    std::vector<String> outputNames;
    std::vector<String> inputNames;
    bool hasNamedOutput(const String &name)
    {
        return std::find(outputNames.begin(), outputNames.end(), name) != outputNames.end();
    }
    bool hasNemedInput(const String &name)
    {
        return std::find(inputNames.begin(), inputNames.end(), name) != inputNames.end();
    }

    std::vector<LayerOutId> inputBlobsId;
    std::set<int> inputLayersId;
    std::set<int> requiredOutputs;

    Ptr<Layer> layerInstance;
    std::vector<Blob> outputBlobs;
    std::vector<Blob*> inputBlobs;

    int flag;
};

struct Net::Impl
{
    Impl()
    {
        layers.insert(make_pair(0, LayerData("_input", "_input")));
        lastLayerId = 1;

        netInputAliases.push_back("input");
        netInputAliases.push_back("data");
    }

    std::vector<String> netInputAliases;

    std::vector<int> netOutputs;

    typedef std::map<int, LayerData> MapIdToLayerData;
    std::map<int, LayerData> layers;

    std::map<String, int> layerNameToId;

    int lastLayerId;

    int getLayerId(const String &layerName)
    {
        std::map<String, int>::iterator it = layerNameToId.find(layerName);
        return (it != layerNameToId.end()) ? it->second : -1;
    }

    int getLayerId(const DictValue &v)
    {
        if (v.isString())
            return getLayerId(v.get<String>());
        else if (v.isInt())
            int id = v.get<int>();
        else
            CV_Assert(v.isString() || v.isInt());
    }

    LayerData& getLayerData(const DictValue &v)
    {
        int id = getLayerId(v);
        std::map<int, LayerData>::iterator it = layers.find(id);
        CV_Assert(id >= 0 && it != layers.end());
        return it->second;
    }

    int findOutputsByName(const String &name, LayerOutId *found, int maxCount)
    {
        int count = 0;

        MapIdToLayerData::iterator it;
        for (it = layers.begin(); it != layers.end() && count < maxCount; it++)
        {
            int lid = it->first;
            LayerData &ld = it->second;

            for (size_t oi = 0; oi < ld.outputNames.size() && count < maxCount; oi++)
            {
                if (ld.outputNames[oi] == name)
                    found[count++] = LayerOutId(lid, oi);
            }
        }

        return count;
    }

    void connectInputs()
    {
        LayerOutId foundOutputs[3], out;

        MapIdToLayerData::iterator it;
        for (it = layers.begin(); it != layers.end(); it++)
        {
            int lid = it->first;
            LayerData &ld = it->second;

            ld.inputBlobs.resize(ld.inputNames.size());
            ld.inputBlobsId.resize(ld.inputNames.size());
            ld.inputLayersId.clear();

            for (size_t ii = 0; ii < ld.inputNames.size(); ii++)
            {
                const String &tgtName = ld.inputNames[ii];

                int foundCount = findOutputsByName(tgtName, foundOutputs, 3);

                if (foundCount > 2)
                {
                    CV_Error(cv::Error::StsNotImplemented, "Two or more non-inplace blobs have the same name \"" + tgtName + "\"");
                }
                else if (foundCount == 2)
                {
                    bool inPlace[2];
                    inPlace[0] = layers[ foundOutputs[0].lid ].hasNemedInput(tgtName);
                    inPlace[1] = layers[ foundOutputs[1].lid ].hasNemedInput(tgtName);

                    if (!inPlace[0] && !inPlace[1])
                    {
                        CV_Error(cv::Error::StsNotImplemented, "Two or more non-inplace blobs have the same name \"" + tgtName + "\"");
                    }
                    else if (inPlace[0] && inPlace[1])
                    {
                        CV_Error(cv::Error::StsNotImplemented, "Two or more blobs has same in-place blob \"" + tgtName + "\"");
                    }
                    else
                    {
                        if (ld.hasNamedOutput(tgtName))
                            out = (inPlace[0]) ? foundOutputs[1] : foundOutputs[0];
                        else
                            out = (inPlace[0]) ? foundOutputs[0] : foundOutputs[1];
                    }
                }
                else if (foundCount == 0)
                {
                    if (std::find(netInputAliases.begin(), netInputAliases.end(), tgtName) == netInputAliases.end())
                    {
                        CV_Error(cv::Error::StsBadArg, "Can't find specified input blob \"" + tgtName + "\" for layer \"" + ld.name + "\"");
                        continue;
                    }

                    LayerData &inputLayer = layers[0];
                    int outIndex = std::find(inputLayer.outputNames.begin(), inputLayer.outputNames.end(), tgtName) - inputLayer.outputNames.begin();

                    if (outIndex < inputLayer.outputNames.size())
                    {
                        out = LayerOutId(0, outIndex);
                    }
                    else
                    {
                        inputLayer.outputNames.push_back(tgtName);
                        inputLayer.outputBlobs.resize(inputLayer.outputNames.size());
                        out = LayerOutId(0, (int)inputLayer.outputNames.size() - 1);
                    }
                }
                else
                {
                    out = foundOutputs[0];
                }

                ld.inputBlobs[ii] = &layers[out.lid].outputBlobs[out.oid];
                ld.inputBlobsId[ii] = out;
                ld.inputLayersId.insert(out.lid);
                layers[out.lid].requiredOutputs.insert(out.oid);
            }
        }

        for (it = layers.begin(); it != layers.end(); it++)
        {
            LayerData& ld = it->second;

            std::cout << ld.name << std::endl;
            std::cout << "Connected:" << std::endl;
            for (std::set<int>::iterator j = ld.inputLayersId.begin(); j != ld.inputLayersId.end(); j++)
                std::cout << layers[*j].name << std::endl;
            std::cout << std::endl;
        }
    }

    void computeNetOutputs()
    {
        netOutputs.clear();
        
        MapIdToLayerData::iterator it;
        for (it = layers.begin(); it != layers.end(); it++)
        {
            int lid = it->first;
            LayerData &ld = it->second;

            if (ld.requiredOutputs.size() == 0)
                netOutputs.push_back(lid);
        }

        std::cout << "\nNet Outputs(" << netOutputs.size() << "):\n";
        for (int i = 0; i < netOutputs.size(); i++)
            std::cout << layers[netOutputs[i]].name << std::endl;
    }

    void allocateOutputBlobs()
    {
        MapIdToLayerData::iterator it;
        for (it = layers.begin(); it != layers.end(); it++)
        {
            LayerData &ld = it->second;
            ld.outputBlobs.resize(ld.outputNames.size());
        }
    }

    void allocateLayers()
    {
        MapIdToLayerData::iterator it;
        for (it = layers.begin(); it != layers.end(); it++)
        {
            int lid = it->first;
            LayerData &ld = it->second;

            if (ld.layerInstance == NULL && lid != 0)
            {
                ld.layerInstance = LayerRegister::createLayerInstance(ld.type, ld.params);
                if (ld.layerInstance == NULL)
                {
                    std::cerr << "Can't create layer \"" << ld.name << "\" of type \"" << ld.type << "\"" << std::endl;
                }
            }
        }
    }

    void forwardLayer(int layerId, bool clearFlags = true)
    {
        if (clearFlags)
        {
            MapIdToLayerData::iterator it;
            for (it = layers.begin(); it != layers.end(); it++)
                it->second.flag = 0;
        }

        LayerData &ld = layers[layerId];
        for (set<int>::iterator i = ld.inputLayersId.begin(); i != ld.inputLayersId.end(); i++)
        {
            LayerData &ild = layers[*i];

            if (!ild.flag)
            {
                if (ild.layerInstance)
                    ild.layerInstance->forward(ild.inputBlobs, ild.outputBlobs);
                ild.flag = true;
            }
        }
    }

    void forwardAll()
    {
        MapIdToLayerData::iterator it;
        for (it = layers.begin(); it != layers.end(); it++)
            it->second.flag = 0;

        for (it = layers.begin(); it != layers.end(); it++)
            forwardLayer(it->first, false);
    }
};

Net::Net() : impl(new Net::Impl)
{

}

Net::~Net()
{

}

int Net::addLayer(const String &name, const String &type, LayerParams &params)
{
    if (impl->getLayerId(name) >= 0)
    {
        CV_Error(cv::Error::StsBadArg, "Layer \"" + name + "\" already into net");
        return -1;
    }

    int id = ++impl->lastLayerId;
    impl->layerNameToId.insert(std::make_pair(name, id));
    impl->layers.insert(std::make_pair(id, LayerData(name, type, params)));

    return id;
}

void Net::connect(BlobId input, BlobId output)
{
    
}

void Net::setOutputNames(LayerId layer, const std::vector<String> &outputNames)
{
    LayerData &ld = impl->getLayerData(layer);
    CV_Assert(ld.outputNames.size() == 0);
    ld.outputNames.assign(outputNames.begin(), outputNames.end());
}

void Net::setLayerInputs(const std::vector<String> &outputs, LayerId layer)
{
    LayerData &ld = impl->getLayerData(layer);
    ld.inputNames.assign(outputs.begin(), outputs.end());
}

void Net::forward()
{
    impl->allocateOutputBlobs();
    impl->connectInputs();
    impl->computeNetOutputs();
    impl->allocateLayers();

    impl->forwardAll();
}

void Net::forward(LayerId toLayer)
{
    impl->forwardLayer(impl->getLayerId(toLayer));
}

Importer::~Importer()
{

}

//////////////////////////////////////////////////////////////////////////

#include <sstream>
template<typename T>
String toString(const T &v)
{
    std::stringstream ss;
    ss << v;
    return ss.str();
}

int Layer::getNumInputs()
{
    return 1;
}

int Layer::getNumOutputs()
{
    return 1;
}

cv::String Layer::getInputName(int inputNum)
{
    return "input" + toString(inputNum);
}


cv::String Layer::getOutputName(int outputNum)
{
    return "output" + toString(outputNum);
}

Layer::~Layer()
{

}

//////////////////////////////////////////////////////////////////////////

struct LayerRegister::Impl : public std::map<String, LayerRegister::Constuctor>
{
};

//allocates on load and cleans on exit
Ptr<LayerRegister::Impl> LayerRegister::impl(new LayerRegister::Impl());

void LayerRegister::registerLayer(const String &_type, Constuctor constructor)
{
    String type = _type.toLowerCase();
    Impl::iterator it = impl->find(type);

    if (it != impl->end() && it->second != constructor)
    {
        CV_Error(cv::Error::StsBadArg, "Layer \"" + type + "\" already was registered");
    }

    impl->insert(std::make_pair(type, constructor));
}

void LayerRegister::unregisterLayer(const String &_type)
{
    String type = _type.toLowerCase();
    impl->erase(type);
}

Ptr<Layer> LayerRegister::createLayerInstance(const String &_type, LayerParams& params)
{
    String type = _type.toLowerCase();
    Impl::const_iterator it = LayerRegister::impl->find(type);

    if (it != impl->end())
    {
        return it->second(params);
    }
    else
    {
        return Ptr<Layer>(); //NULL
    }
}

}
}