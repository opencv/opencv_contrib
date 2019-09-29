// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef _OPENCV_DNN_SUPERRES_HPP_
#define _OPENCV_DNN_SUPERRES_HPP_

/** @defgroup dnn_superres DNN used for super resolution

This module contains functionality for upscaling an image via convolutional neural networks.
The following four models are implemented:

- EDSR <https://arxiv.org/abs/1707.02921>
- ESPCN <https://arxiv.org/abs/1609.05158>
- FSRCNN <https://arxiv.org/abs/1608.00367>
- LapSRN <https://arxiv.org/abs/1710.01992>

*/

#include "opencv2/core.hpp"
#include "opencv2/dnn.hpp"

namespace cv
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
    dnn::Net net;

    std::string alg; //algorithm

    int sc; //scale factor

    void preprocess(InputArray inpImg, OutputArray outpImg);

    void reconstruct_YCrCb(InputArray inpImg, InputArray origImg, OutputArray outpImg, int scale);

    void reconstruct_YCrCb(InputArray inpImg, InputArray origImg, OutputArray outpImg);

    void preprocess_YCrCb(InputArray inpImg, OutputArray outpImg);

public:

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
    DnnSuperResImpl(const std::string& algo, int scale);

    /** @brief Read the model from the given path
    @param path Path to the model file.
    */
    void readModel(const std::string& path);

    /** @brief Read the model from the given path
    @param weights Path to the model weights file.
    @param definition Path to the model definition file.
    */
    void readModel(const std::string& weights, const std::string& definition);

    /** @brief Set desired model
    @param algo String containing one of the desired models:
        - __edsr__
        - __espcn__
        - __fsrcnn__
        - __lapsrn__
    @param scale Integer specifying the upscale factor
     */
    void setModel(const std::string& algo, int scale);

    /** @brief Upsample via neural network
    @param img Image to upscale
    @param result Destination upscaled image
     */
    void upsample(InputArray img, OutputArray result);

    /** @brief Upsample via neural network of multiple outputs
    @param img Image to upscale
    @param imgs_new Destination upscaled images
    @param scale_factors Scaling factors of the output nodes
    @param node_names Names of the output nodes in the neural network
    */
    void upsampleMultioutput(InputArray img, std::vector<Mat> &imgs_new, const std::vector<int>& scale_factors, const std::vector<String>& node_names);

    /** @brief Returns the scale factor of the model:
    @return Current scale factor.
    */
    int getScale();

    /** @brief Returns the scale factor of the model:
    @return Current algorithm.
    */
    std::string getAlgorithm();
};

//! @} dnn_superres

}} // cv::dnn_superres::
#endif
