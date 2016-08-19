/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2013, OpenCV Foundation, all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#ifndef __OPENCV_DNN_DNN_HPP__
#define __OPENCV_DNN_DNN_HPP__

#include <vector>
#include <opencv2/core.hpp>
#include <opencv2/dnn/dict.hpp>
#include <opencv2/dnn/blob.hpp>

namespace cv
{
namespace dnn //! This namespace is used for dnn module functionlaity.
{
//! @addtogroup dnn
//! @{

    /** @brief Initialize dnn module and built-in layers.
     *
     * This function automatically called on most of OpenCV builds,
     * but you need to call it manually on some specific configurations (iOS for example).
     */
    CV_EXPORTS_W void initModule();

    /** @brief This class provides all data needed to initialize layer.
     *
     * It includes dictionary with scalar params (which can be readed by using Dict interface),
     * blob params #blobs and optional meta information: #name and #type of layer instance.
    */
    class CV_EXPORTS LayerParams : public Dict
    {
    public:
        //TODO: Add ability to name blob params
        std::vector<Blob> blobs; //!< List of learned parameters stored as blobs.

        String name; //!< Name of the layer instance (optional, can be used internal purposes).
        String type; //!< Type name which was used for creating layer by layer factory (optional).
    };

    /** @brief This interface class allows to build new Layers - are building blocks of networks.
     *
     * Each class, derived from Layer, must implement allocate() methods to declare own outputs and forward() to compute outputs.
     * Also before using the new layer into networks you must register your layer by using one of @ref dnnLayerFactory "LayerFactory" macros.
     */
    class CV_EXPORTS_W Layer
    {
    public:

        //! List of learned parameters must be stored here to allow read them by using Net::getParam().
        CV_PROP_RW std::vector<Blob> blobs;

        /** @brief Allocates internal buffers and output blobs with respect to the shape of inputs.
         *  @param[in]  input  vector of already allocated input blobs
         *  @param[out] output vector of output blobs, which must be allocated
         *
         * This method must create each produced blob according to shape of @p input blobs and internal layer params.
         * If this method is called first time then @p output vector consists from empty blobs and its size determined by number of output connections.
         * This method can be called multiple times if size of any @p input blob was changed.
         */
        virtual void allocate(const std::vector<Blob*> &input, std::vector<Blob> &output) = 0;

        /** @brief Given the @p input blobs, computes the output @p blobs.
         *  @param[in]  input  the input blobs.
         *  @param[out] output allocated output blobs, which will store results of the computation.
         */
        virtual void forward(std::vector<Blob*> &input, std::vector<Blob> &output) = 0;

        /** @brief @overload */
        CV_WRAP void allocate(const std::vector<Blob> &inputs, CV_OUT std::vector<Blob> &outputs);

        /** @brief @overload */
        CV_WRAP std::vector<Blob> allocate(const std::vector<Blob> &inputs);

        /** @brief @overload */
        CV_WRAP void forward(const std::vector<Blob> &inputs, CV_IN_OUT std::vector<Blob> &outputs);

        /** @brief Allocates layer and computes output. */
        CV_WRAP void run(const std::vector<Blob> &inputs, CV_OUT std::vector<Blob> &outputs);

        /** @brief Returns index of input blob into the input array.
         *  @param inputName label of input blob
         *
         * Each layer input and output can be labeled to easily identify them using "%<layer_name%>[.output_name]" notation.
         * This method maps label of input blob to its index into input vector.
         */
        virtual int inputNameToIndex(String inputName);
        /** @brief Returns index of output blob in output array.
         *  @see inputNameToIndex()
         */
        virtual int outputNameToIndex(String outputName);

        CV_PROP String name; //!< Name of the layer instance, can be used for logging or other internal purposes.
        CV_PROP String type; //!< Type name which was used for creating layer by layer factory.

        Layer();
        explicit Layer(const LayerParams &params);      //!< Initializes only #name, #type and #blobs fields.
        void setParamsFrom(const LayerParams &params);  //!< Initializes only #name, #type and #blobs fields.
        virtual ~Layer();
    };

    /** @brief This class allows to create and manipulate comprehensive artificial neural networks.
     *
     * Neural network is presented as directed acyclic graph (DAG), where vertices are Layer instances,
     * and edges specify relationships between layers inputs and outputs.
     *
     * Each network layer has unique integer id and unique string name inside its network.
     * LayerId can store either layer name or layer id.
     *
     * This class supports reference counting of its instances, i. e. copies point to the same instance.
     */
    class CV_EXPORTS_W_SIMPLE Net
    {
    public:

        CV_WRAP Net();  //!< Default constructor.
        CV_WRAP ~Net(); //!< Destructor frees the net only if there aren't references to the net anymore.

        /** Returns true if there are no layers in the network. */
        CV_WRAP bool empty() const;

        /** @brief Adds new layer to the net.
         *  @param name   unique name of the adding layer.
         *  @param type   typename of the adding layer (type must be registered in LayerRegister).
         *  @param params parameters which will be used to initialize the creating layer.
         *  @returns unique identifier of created layer, or -1 if a failure will happen.
         */
        int addLayer(const String &name, const String &type, LayerParams &params);
        /** @brief Adds new layer and connects its first input to the first output of previously added layer.
         *  @see addLayer()
         */
        int addLayerToPrev(const String &name, const String &type, LayerParams &params);

        /** @brief Converts string name of the layer to the integer identifier.
         *  @returns id of the layer, or -1 if the layer wasn't found.
         */
        CV_WRAP int getLayerId(const String &layer);

        CV_WRAP std::vector<String> getLayerNames() const;

        /** @brief Container for strings and integers. */
        typedef DictValue LayerId;

        /** @brief Returns pointer to layer with specified name which the network use. */
        CV_WRAP Ptr<Layer> getLayer(LayerId layerId);

        /** @brief Delete layer for the network (not implemented yet) */
        CV_WRAP void deleteLayer(LayerId layer);

        /** @brief Connects output of the first layer to input of the second layer.
         *  @param outPin descriptor of the first layer output.
         *  @param inpPin descriptor of the second layer input.
         *
         * Descriptors have the following template <DFN>&lt;layer_name&gt;[.input_number]</DFN>:
         * - the first part of the template <DFN>layer_name</DFN> is sting name of the added layer.
         *   If this part is empty then the network input pseudo layer will be used;
         * - the second optional part of the template <DFN>input_number</DFN>
         *   is either number of the layer input, either label one.
         *   If this part is omitted then the first layer input will be used.
         *
         *  @see setNetInputs(), Layer::inputNameToIndex(), Layer::outputNameToIndex()
         */
        CV_WRAP void connect(String outPin, String inpPin);

        /** @brief Connects #@p outNum output of the first layer to #@p inNum input of the second layer.
         *  @param outLayerId identifier of the first layer
         *  @param inpLayerId identifier of the second layer
         *  @param outNum number of the first layer output
         *  @param inpNum number of the second layer input
         */
        void connect(int outLayerId, int outNum, int inpLayerId, int inpNum);

        /** @brief Sets outputs names of the network input pseudo layer.
         *
         * Each net always has special own the network input pseudo layer with id=0.
         * This layer stores the user blobs only and don't make any computations.
         * In fact, this layer provides the only way to pass user data into the network.
         * As any other layer, this layer can label its outputs and this function provides an easy way to do this.
         */
        CV_WRAP void setNetInputs(const std::vector<String> &inputBlobNames);

        /** @brief Initializes and allocates all layers. */
        CV_WRAP void allocate();

        /** @brief Runs forward pass to compute output of layer @p toLayer.
          * @details By default runs forward pass for the whole network.
          */
        CV_WRAP void forward(LayerId toLayer = String());
        /** @brief Runs forward pass to compute output of layer @p toLayer, but computations start from @p startLayer */
        void forward(LayerId startLayer, LayerId toLayer);
        /** @overload */
        void forward(const std::vector<LayerId> &startLayers, const std::vector<LayerId> &toLayers);

        //TODO:
        /** @brief Optimized forward.
         *  @warning Not implemented yet.
         *  @details Makes forward only those layers which weren't changed after previous forward().
         */
        void forwardOpt(LayerId toLayer);
        /** @overload */
        void forwardOpt(const std::vector<LayerId> &toLayers);

        /** @brief Sets the new value for the layer output blob
         *  @param outputName descriptor of the updating layer output blob.
         *  @param blob new blob.
         *  @see connect(String, String) to know format of the descriptor.
         *  @note If updating blob is not empty then @p blob must have the same shape,
         *  because network reshaping is not implemented yet.
         */
        CV_WRAP void setBlob(String outputName, const Blob &blob);

        /** @brief Returns the layer output blob.
         *  @param outputName the descriptor of the returning layer output blob.
         *  @see connect(String, String)
         */
        CV_WRAP Blob getBlob(String outputName);

        /** @brief Sets the new value for the learned param of the layer.
         *  @param layer name or id of the layer.
         *  @param numParam index of the layer parameter in the Layer::blobs array.
         *  @param blob the new value.
         *  @see Layer::blobs
         *  @note If shape of the new blob differs from the previous shape,
         *  then the following forward pass may fail.
        */
        CV_WRAP void setParam(LayerId layer, int numParam, const Blob &blob);

        /** @brief Returns parameter blob of the layer.
         *  @param layer name or id of the layer.
         *  @param numParam index of the layer parameter in the Layer::blobs array.
         *  @see Layer::blobs
         */
        CV_WRAP Blob getParam(LayerId layer, int numParam = 0);

    private:

        struct Impl;
        Ptr<Impl> impl;
    };

    /** @brief Small interface class for loading trained serialized models of different dnn-frameworks. */
    class CV_EXPORTS_W Importer
    {
    public:

        /** @brief Adds loaded layers into the @p net and sets connections between them. */
        CV_WRAP virtual void populateNet(Net net) = 0;

        virtual ~Importer();
    };

    /** @brief Creates the importer of <a href="http://caffe.berkeleyvision.org">Caffe</a> framework network.
     *  @param prototxt   path to the .prototxt file with text description of the network architecture.
     *  @param caffeModel path to the .caffemodel file with learned network.
     *  @returns Pointer to the created importer, NULL in failure cases.
     */
    CV_EXPORTS_W Ptr<Importer> createCaffeImporter(const String &prototxt, const String &caffeModel = String());

    /** @brief Reads a network model stored in Caffe model files.
      * @details This is shortcut consisting from createCaffeImporter and Net::populateNet calls.
      */
    CV_EXPORTS_W Net readNetFromCaffe(const String &prototxt, const String &caffeModel = String());

    /** @brief Creates the importer of <a href="http://torch.ch">Torch7</a> framework network.
     *  @param filename path to the file, dumped from Torch by using torch.save() function.
     *  @param isBinary specifies whether the network was serialized in ascii mode or binary.
     *  @returns Pointer to the created importer, NULL in failure cases.
     *
     *  @warning Torch7 importer is experimental now, you need explicitly set CMake `opencv_dnn_BUILD_TORCH_IMPORTER` flag to compile its.
     *
     *  @note Ascii mode of Torch serializer is more preferable, because binary mode extensively use `long` type of C language,
     *  which has various bit-length on different systems.
     *
     * The loading file must contain serialized <a href="https://github.com/torch/nn/blob/master/doc/module.md">nn.Module</a> object
     * with importing network. Try to eliminate a custom objects from serialazing data to avoid importing errors.
     *
     * List of supported layers (i.e. object instances derived from Torch nn.Module class):
     * - nn.Sequential
     * - nn.Parallel
     * - nn.Concat
     * - nn.Linear
     * - nn.SpatialConvolution
     * - nn.SpatialMaxPooling, nn.SpatialAveragePooling
     * - nn.ReLU, nn.TanH, nn.Sigmoid
     * - nn.Reshape
     *
     * Also some equivalents of these classes from cunn, cudnn, and fbcunn may be successfully imported.
     */
    CV_EXPORTS_W Ptr<Importer> createTorchImporter(const String &filename, bool isBinary = true);

    /** @brief Loads blob which was serialized as torch.Tensor object of Torch7 framework.
     *  @warning This function has the same limitations as createTorchImporter().
     */
    CV_EXPORTS_W Blob readTorchBlob(const String &filename, bool isBinary = true);

//! @}
}
}

#include <opencv2/dnn/layer.hpp>
#include <opencv2/dnn/dnn.inl.hpp>

#endif  /* __OPENCV_DNN_DNN_HPP__ */
