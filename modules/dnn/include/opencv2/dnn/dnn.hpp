#ifndef __OPENCV_DNN_DNN_HPP__
#define __OPENCV_DNN_DNN_HPP__

#include <vector>
#include <opencv2/core.hpp>
#include <opencv2/dnn/dict.hpp>
#include <opencv2/dnn/blob.hpp>

namespace cv
{
namespace dnn
{
    CV_EXPORTS void initModule();

    class CV_EXPORTS LayerParams : public Dict
    {
    public:

        std::vector<Blob> learnedBlobs;
    };

    //Interface class allows to build new Layers
    class CV_EXPORTS Layer
    {
    public:
        //learned params of layer must be stored here to allow externally read them
        std::vector<Blob> learnedParams;

        virtual ~Layer();

        //shape of output blobs must be adjusted with respect to shape of input blobs
        virtual void allocate(const std::vector<Blob*> &inputs, std::vector<Blob> &outputs) = 0;

        virtual void forward(std::vector<Blob*> &inputs, std::vector<Blob> &outputs) = 0;

        //each input/output can be labeled to easily identify their using "layer_name.output_name"
        virtual int inputNameToIndex(String inputName);
        virtual int outputNameToIndex(String outputName);
    };

    //containers for String and int
    typedef DictValue LayerId;

    class CV_EXPORTS Net
    {
    public:

        Net();
        ~Net();

        int addLayer(const String &name, const String &type, LayerParams &params);
        int addLayerToPrev(const String &name, const String &type, LayerParams &params);

        int getLayerId(const String &layer);
        void deleteLayer(LayerId layer);

        void connect(String outPin, String inpPin);
        void connect(int outLayerId, int outNum, int inLayerId, int inNum);
        void setNetInputs(const std::vector<String> &inputBlobNames);

        void forward();
        void forward(LayerId toLayer);
        void forward(LayerId startLayer, LayerId toLayer);
        void forward(const std::vector<LayerId> &startLayers, const std::vector<LayerId> &toLayers);

        //[Wished feature] Optimized smart forward(). Makes forward only for layers which wasn't changed after previous forward().
        void forwardOpt(LayerId toLayer);
        void forwardOpt(const std::vector<LayerId> &toLayers);

        void setBlob(String outputName, const Blob &blob);
        Blob getBlob(String outputName);

        void setParam(LayerId layer, int numParam, const Blob &blob);
        Blob getParam(LayerId layer, int numParam = 0);

    private:

        struct Impl;
        Ptr<Impl> impl;
    };

    class Importer
    {
    public:

        virtual void populateNet(Net net) = 0;

        virtual ~Importer();
    };

    CV_EXPORTS Ptr<Importer> createCaffeImporter(const String &prototxt, const String &caffeModel = String());

    CV_EXPORTS Ptr<Importer> createTorchImporter(const String &filename, bool isBinary = true);

    CV_EXPORTS Blob readTorchMat(const String &filename, bool isBinary = true);
}
}

#include <opencv2/dnn/layer.hpp>
#include <opencv2/dnn/dnn.inl.hpp>

#endif  /* __OPENCV_DNN_DNN_HPP__ */
