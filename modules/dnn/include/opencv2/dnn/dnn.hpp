#ifndef __OPENCV_DNN_DNN_HPP__
#define __OPENCV_DNN_DNN_HPP__

#include <opencv2/core.hpp>
#include <map>
#include <vector>
#include <opencv2/dnn/dict.hpp>

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
    class CV_EXPORTS Blob
    {
    public:
        Blob();
        Blob(InputArray in);

        void fill(InputArray in);
        void fill(int ndims, const int *sizes, int type, void *data, bool deepCopy = true);
        void create(int ndims, const int *sizes, int type = CV_32F);

        bool empty() const;

        Mat& getMatRef();
        const Mat& getMatRef() const;
        Mat getMat();
        Mat getMat(int num, int channel);

        int cols() const;
        int rows() const;
        Size size() const;

        int channels() const;
        int num() const;

        Vec4i shape() const;

    private:
        Mat m;
    };

    class CV_EXPORTS LayerParams : public Dict
    {
    public:

        std::vector<Blob> learnedBlobs;
    };

    class CV_EXPORTS LayerRegister
    {
    public:

        typedef Layer* (*Constuctor)();

        static void registerLayer(const String &type, Constuctor constructor);

        static void unregisterLayer(const String &type);

        static Ptr<Layer> createLayerInstance(const String &type);

    private:
        LayerRegister();
        LayerRegister(const LayerRegister &lr);

        static std::map<String, Constuctor> registeredLayers;
    };

    //this class allows to build new Layers
    class CV_EXPORTS Layer
    {
    public:
        //TODO: this field must be declared as public if we want support possibility to change these params in runtime
        std::vector<Blob> learnedParams;

        virtual ~Layer();

        //type of Layer
        virtual String type() const = 0;

        //setUp calls once (think that it's constructor)
        virtual void setUp(LayerParams &params) = 0;

        //maybe useless function
        //shape of output blobs must be adjusted with respect to shape of input blobs
        virtual void adjustShape(const std::vector<Blob> &inputs, std::vector<Blob> &outputs) = 0;

        virtual void forward(std::vector<Blob> &inputs, std::vector<Blob> &outputs) = 0;

        virtual int getNumInputs();
        virtual int getNumOutputs();
        //each input/output can be labeled to easily identify their using "layer_name.output_name"
        virtual String getInputName(int inputNum);
        virtual String getOutputName(int outputNum);
    };
    
    //containers for String and int
    typedef DictValue LayerId;
    typedef DictValue BlobId;

    class CV_EXPORTS Net
    {
    public:

        Net();
        ~Net();

        int addLayer(const String &name, const String &type, LayerParams &params = LayerParams());
        void deleteLayer(LayerId layer);
        
        //each output of each layer can be labeled by unique string label (as in Caffe)
        //if label not specified then %layer_name%.%layer_output_id% can be used
        void setOutputNames(LayerId layer, const std::vector<String> &outputNames);

        void connect(BlobId input, BlobId output);
        void connect(const std::vector<BlobId> &outputs, const std::vector<BlobId> &inputs);
        void connect(const std::vector<BlobId> &outputs, LayerId layer);

        int getOutputId(LayerId layer, int outputNum);
        int getInputId(LayerId layer, int inputNum);
        int getLayerId(LayerId layer);

        void forward();
        void forward(LayerId toLayer);
        void forward(LayerId startLayer, LayerId toLayer);
        void forward(const std::vector<LayerId> &startLayers, const std::vector<LayerId> &toLayers);

        //[Wished feature] Optimized smart forward(). Makes forward only for layers which wasn't changed after previous forward().
        void forwardOpt(LayerId toLayer);
        void forwardOpt(const std::vector<LayerId> &toLayers);

        void setBlob(BlobId outputName, const Blob &blob);
        Blob getBlob(BlobId outputName);
        
        void setParam(LayerId layer, int numParam, const Blob &blob);
        void getParam(LayerId layer, int numParam);

    private:

        struct Impl;
        Ptr<Impl> impl;
    };

    class CV_EXPORTS Importer
    {
    public:

        virtual void populateNet(Net net) = 0;

        virtual ~Importer();
    };

    CV_EXPORTS Ptr<Importer> createCaffeImporter(const String &prototxt, const String &caffeModel);

}
}

#include <opencv2/dnn/dnn.inl.hpp>

#endif  /* __OPENCV_DNN_DNN_HPP__ */
