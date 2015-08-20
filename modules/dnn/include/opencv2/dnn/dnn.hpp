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
    /** @brief initialize dnn module and built-in layers
     * This function automatically called on most of OpenCV builds,
     * but you need to call it manually on some specific configurations.
     */
    CV_EXPORTS void initModule();

    struct CV_EXPORTS LayerParams : public Dict
    {
        ///list of learned parameters stored as blobs
        std::vector<Blob> blobs;

        ///optional, name of the layer instance (can be used internal purposes)
        String name;
        ///optional, type name which was used for creating layer by layer factory
        String type;
    };

    ///Interface class allows to build new Layers
    struct CV_EXPORTS Layer
    {
        ///list of learned parameters must be stored here to allow read them using Net::getParam()
        std::vector<Blob> blobs;

        //shape of output blobs must be adjusted with respect to shape of input blobs
        virtual void allocate(const std::vector<Blob*> &inputs, std::vector<Blob> &outputs) = 0;

        virtual void forward(std::vector<Blob*> &inputs, std::vector<Blob> &outputs) = 0;

        //each input and output can be labeled to easily identify them using "<layer_name>[.output_name]" notation
        virtual int inputNameToIndex(String inputName);
        virtual int outputNameToIndex(String outputName);

        String name; //!< name of the layer instance, can be used for logging or other internal purposes
        String type; //!< type name which was used for creating layer by layer factory

        Layer();
        explicit Layer(const LayerParams &params); //!< intialize only #name, #type and #blobs fields
        virtual ~Layer();
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

        //[Wished feature] Optimized forward: makes forward only for layers which wasn't changed after previous forward().
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
