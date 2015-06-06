#ifndef __OPENCV_DNN_HPP__
#define __OPENCV_DNN_HPP__

#include <opencv2/core.hpp>
#include <map>
#include <vector>
#include "dict.hpp"

namespace cv
{
namespace dnn
{
    class Layer;
    class NetConfiguration;
    class Net;
    class Blob;
    class LayerParams;

    //wrapper over cv::Mat and cv::UMat
    CV_EXPORTS class Blob
    {
        Mat m;

    public:
        Blob();
        Blob(InputArray in);

        bool empty() const;

        int width() const; //cols
        int height() const; //rows
        int channels() const;
        int num() const;

        Vec4i size() const;
    };

    CV_EXPORTS class LayerParams : public Dict
    {

    public:

        std::vector<Blob> learnedWeights;
    };

    //this class allows to build new Layers
    CV_EXPORTS class Layer
    {
    public:
        //Layer registration routines

        typedef Layer* (*Constuctor)();

        static void registerLayer(const String &type, Constuctor constructor);

        static void unregisterLayer(const String &type);

        static Ptr<Layer> createLayerInstance(const String &type);

    public:

        //TODO: this field must be declared as public if we want support possibility to change these params in runtime
        std::vector<Blob> learnedParams;

        virtual ~Layer();

        //type of Layer
        virtual String type() const;

        //setUp calls once (think that it's constructor)
        virtual void setUp(LayerParams &params);

        //after setUp the following two function must be able to return values
        virtual int getNumInputs();
        virtual int getNumOutputs();

        //maybe useless function
        //shape of output blobs must be adjusted with respect to shape of input blobs
        virtual void adjustShape(const std::vector<Blob> &inputs, std::vector<Blob> &outputs);

        virtual void forward(std::vector<Blob> &inputs, std::vector<Blob> &outputs);

    private:

        static std::map<String, Constuctor> registeredLayers;
    };

    //TODO: divide NetConfiguration interface and implementation, hide internal data
    //TODO: maybe eliminate all int ids and replace them by string names
    //Proxy class for different formats
    //Each format importer must populate it
    CV_EXPORTS class NetConfiguration
    {
    public:

        CV_EXPORTS static Ptr<NetConfiguration> create();

        int addLayer(const String &name, const String &type);

        void deleteLayer(int layerId);

        void setLayerParams(int layerId, LayerParams &params);

        //each output of each layer can be labeled by unique string label (as in Caffe)
        //if label not specified then %layer_name%:c_%N% will be used
        void setLayerOutputLabels(int layerId, const std::vector<String> &outputNames);

        //version #1
        void addConnection(int fromLayer, int fromLayerOutput, int toLayer, int toLayerInput);

        //or maybe version #2
        inline int getBlobId(int layerId, int inputOutputNumber)
        {
            return (layerId << 16) + inputOutputNumber;
        }

        void addConnection(int outputId, int inputId);

        void addConnections(const std::vector<int> &outputIds, const std::vector<int> &inputIds);

    private:

        int lastLayerId;
        std::map< int, Ptr<Layer> > layers;
        std::map< int, std::vector<String> > layerOutputLabels;
    };


    CV_EXPORTS class Net
    {
    public:

        CV_EXPORTS static Ptr<Net> create(Ptr<NetConfiguration> config);

        virtual ~Net() = 0;

        virtual int getBlobId(int layerId, int outputId) = 0;

        virtual int getBlobId(const String &blobName) = 0;

        virtual void forward(std::vector< int, Ptr<Blob> > &inputBlobs, std::vector<int, Ptr<Blob> > &outputBlobs) = 0;

        virtual void forward(int layer, std::vector<Ptr<Blob> > &layerOutputs) = 0;
    };

    CV_EXPORTS class Importer
    {
    public:

        virtual void populateNetConfiguration(Ptr<NetConfiguration> config) = 0;

        virtual ~Importer();
    };

    CV_EXPORTS Ptr<Importer> createCaffeImporter(const String &prototxt, const String &caffeModel);

}
}

#endif
