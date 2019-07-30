// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef _OPENCV_DNN_SUPERRES_DNNSUPERRESIMPL_HPP_
#define _OPENCV_DNN_SUPERRES_DNNSUPERRESIMPL_HPP_

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "opencv2/dnn.hpp"

/** @defgroup dnn_superres DNN used for super resolution

This module contains functionality for upscaling an image via convolutional neural networks.
The following four models are implemented:

- EDSR <https://arxiv.org/abs/1707.02921>
- ESPCN <https://arxiv.org/abs/1609.05158>
- FSRCNN <https://arxiv.org/abs/1608.00367>
- LapSRN <https://arxiv.org/abs/1710.01992>

*/

namespace cv
{
namespace dnn
{
namespace dnn_superres
{
    //! @addtogroup dnn_superres
    //! @{

    /** @brief A class to upscale images via convolutional neural networks.
    The following four models are implemented:

    - edsr
    - espcn
    - fsrcnn
    - lapsrn
     */
    class CV_EXPORTS DnnSuperResImpl
    {
        private:

            /** @brief Net which holds the desired neural network
             */
            Net net;

            /** @brief String that specifies the model
             */
            std::string alg; //algorithm

            /** @brief Int that specifies the desired upscale factor
             */
            int sc; //scale factor

            void registerLayers();

            void preprocess(const Mat inpImg, Mat &outpImg);

            void preprocess_YCrCb(const Mat inpImg, Mat &outpImg);

            void reconstruct_YCrCb(const Mat inpImg, const Mat origImg, Mat &outpImg);

            void reconstruct_YCrCb(const Mat inpImg, const Mat origImg, Mat &outpImg, int scale);

        public:

            static int layer_loaded;

            /** @brief Empty constructor
             */
            DnnSuperResImpl();

            /** @brief Constructor which immediately sets the desired model
            @param algo String containing one of the desired models:
                - __edsr__
                - __espcn__
                - __fsrcnn__
                - __lapsrn__
            @param scale Integer specifying the upscale factor
             */
            DnnSuperResImpl(std::string algo, int scale);

            /** @brief Read the model from the given path
            @param path Path to the model file.
            */
            void readModel(std::string path);

            /** @brief Read the model from the given path
            @param weights Path to the model weights file.
            @param definition Path to the model definition file.
            */
            void readModel(std::string weights, std::string definition);

            /** @brief Set desired model
            @param algo String containing one of the desired models:
                - __edsr__
                - __espcn__
                - __fsrcnn__
                - __lapsrn__
            @param scale Integer specifying the upscale factor
             */
            void setModel(std::string algo, int scale);

            /** @brief Upsample via neural network
            @param img Image to upscale
            @param img_new Destination upscaled image
             */
            void upsample(Mat img, Mat &img_new);

            /** @brief Upsample via neural network of multiple outputs
            @param img Image to upscale
            @param imgs_new Destination upscaled images
            @param scale_factors Scaling factors of the output nodes
            @param node_names Names of the output nodes in the neural network
            */
            void upsample_multioutput(Mat img, std::vector<Mat> &imgs_new, std::vector<int> scale_factors, std::vector<String> node_names);

            /** @brief Returns the scale factor of the model:
            @return Current scale factor.
            */
            int getScale();

            /** @brief Returns the scale factor of the model:
            @return Current algorithm.
            */
            std::string getAlgorithm();

            private:
            /** @brief Class for importing DepthToSpace layer from the ESPCN model
            */
            class DepthToSpace CV_FINAL : public cv::dnn::Layer
            {
                public:

                /// @private
                DepthToSpace(const cv::dnn::LayerParams &params);

                /// @private
                static cv::Ptr<cv::dnn::Layer> create(cv::dnn::LayerParams& params);

                /// @private
                virtual bool getMemoryShapes(const std::vector<std::vector<int> > &inputs,
                                             const int,
                                             std::vector<std::vector<int> > &outputs,
                                             std::vector<std::vector<int> > &) const CV_OVERRIDE;

                /// @private
                virtual void forward(cv::InputArrayOfArrays inputs_arr,
                                     cv::OutputArrayOfArrays outputs_arr,
                                     cv::OutputArrayOfArrays) CV_OVERRIDE;
            };
    };
    //! @}
}
}
}
#endif