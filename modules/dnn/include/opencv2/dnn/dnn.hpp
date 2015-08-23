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
namespace dnn
{
    /** @brief Initialize dnn module and built-in layers.
     *
     * This function automatically called on most of OpenCV builds,
     * but you need to call it manually on some specific configurations.
     */
    CV_EXPORTS void initModule();

    /** @brief
     *
     *
     * */
    struct CV_EXPORTS LayerParams : public Dict
    {
        std::vector<Blob> blobs; //!< List of learned parameters stored as blobs.

        String name; //!< Name of the layer instance (optional, can be used internal purposes).
        String type; //!< Type name which was used for creating layer by layer factory (optional).
    };

    /** @brief Interface class allows to build new Layers.
     */
    struct CV_EXPORTS Layer
    {
        ///List of learned parameters must be stored here to allow read them by using Net::getParam().
        std::vector<Blob> blobs;

        /** @brief Allocates internal buffers and output blobs with respect to the shape of inputs.
         * @param[in]  input  vector of already allocated input blobs
         * @param[out] output vector of output blobs, which must be allocated
         *
         * This method must create each produced blob according to shape of @p input blobs and internal layer params.
         * If this method is called first time then @p output vector consists from empty blobs and its size determined by number of output connections.
         * This method can be called multiple times if size of any @p input blob was changed.
         */
        virtual void allocate(const std::vector<Blob*> &input, std::vector<Blob> &output) = 0;

        virtual void forward(std::vector<Blob*> &inputs, std::vector<Blob> &outputs) = 0;

        /** @brief Returns index of input blob into the input array.
         * @param inputName label of input blob
         *
         * Each input and output blob can be labeled to easily identify them using "<layer_name>[.output_name]" notation.
         * This method map label of input blob to its index into input vector.
         */
        virtual int inputNameToIndex(String inputName);
        /** @brief Returns index of output blob in output array.
         * @see inputNameToIndex()
         */
        virtual int outputNameToIndex(String outputName);

        String name; //!< Name of the layer instance, can be used for logging or other internal purposes.
        String type; //!< Type name which was used for creating layer by layer factory.

        Layer();
        explicit Layer(const LayerParams &params); //!< Intialize only #name, #type and #blobs fields.
        virtual ~Layer();
    };

    typedef DictValue LayerId; //!< Container for strings and integers.

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
