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

class CV_EXPORTS_W DnnSuperResImpl
{
private:

    /** @brief Net which holds the desired neural network
     */
    dnn::Net net;

    std::string alg; //algorithm

    int sc; //scale factor

    void reconstruct_YCrCb(InputArray inpImg, InputArray origImg, OutputArray outpImg, int scale);

    void preprocess_YCrCb(InputArray inpImg, OutputArray outpImg);

public:

    /** @brief Empty constructor for python
     */
    CV_WRAP static Ptr<DnnSuperResImpl> create();

    // /** @brief Empty constructor
    //  */
    DnnSuperResImpl();

    /** @brief Constructor which immediately sets the desired model
    @param algo String containing one of the desired models:
        - __edsr__
        - __espcn__
        - __fsrcnn__
        - __lapsrn__
    @param scale Integer specifying the upscale factor
     */
    DnnSuperResImpl(const String& algo, int scale);

    /** @brief Read the model from the given path
    @param path Path to the model file.
    */
    CV_WRAP void readModel(const String& path);

    /** @brief Read the model from the given path
    @param weights Path to the model weights file.
    @param definition Path to the model definition file.
    */
    void readModel(const String& weights, const String& definition);

    /** @brief Set desired model
    @param algo String containing one of the desired models:
        - __edsr__
        - __espcn__
        - __fsrcnn__
        - __lapsrn__
    @param scale Integer specifying the upscale factor
     */
    CV_WRAP void setModel(const String& algo, int scale);

    /** @brief Set computation backend
    */
    CV_WRAP void setPreferableBackend(int backendId);

    /** @brief Set computation target
    */
    CV_WRAP void setPreferableTarget(int targetId);

    /** @brief Upsample via neural network
    @param img Image to upscale
    @param result Destination upscaled image
     */
    CV_WRAP void upsample(InputArray img, OutputArray result);

    /** @brief Upsample via neural network of multiple outputs
    @param img Image to upscale
    @param imgs_new Destination upscaled images
    @param scale_factors Scaling factors of the output nodes
    @param node_names Names of the output nodes in the neural network
    */
    CV_WRAP void upsampleMultioutput(InputArray img, std::vector<Mat> &imgs_new, const std::vector<int>& scale_factors, const std::vector<String>& node_names);

    /** @brief Returns the scale factor of the model:
    @return Current scale factor.
    */
    CV_WRAP int getScale();

    /** @brief Returns the scale factor of the model:
    @return Current algorithm.
    */
    CV_WRAP String getAlgorithm();
};

//! @} dnn_superres

}} // cv::dnn_superres::
#endif
