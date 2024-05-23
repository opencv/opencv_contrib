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

#ifndef OPENCV_DNNLEGACY_HPP_HPP
#define OPENCV_DNNLEGACY_HPP_HPP

#include <vector>
#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>
#include "opencv2/core/async.hpp"

#include <opencv2/dnn/dict.hpp>

namespace cv {
namespace dnnlegacy {

    template <typename Importer, typename ... Args>
    cv::dnn::Net readNet(Args&& ... args)
    {
        cv::dnn::Net net;
        Importer importer(net, std::forward<Args>(args)...);
        return net;
    }
//! @addtogroup dnnlegacy
//! @{

    /** @brief Reads a network model stored in <a href="https://pjreddie.com/darknet/">Darknet</a> model files.
    *  @param cfgFile      path to the .cfg file with text description of the network architecture.
    *  @param darknetModel path to the .weights file with learned network.
    *  @returns Network object that ready to do forward, throw an exception in failure cases.
    */
    CV_EXPORTS_W cv::dnn::Net readNetFromDarknet(CV_WRAP_FILE_PATH const String &cfgFile, CV_WRAP_FILE_PATH const String &darknetModel = String());

    /** @brief Reads a network model stored in <a href="https://pjreddie.com/darknet/">Darknet</a> model files.
     *  @param bufferCfg   A buffer contains a content of .cfg file with text description of the network architecture.
     *  @param bufferModel A buffer contains a content of .weights file with learned network.
     *  @returns Net object.
     */
    CV_EXPORTS_W cv::dnn::Net readNetFromDarknet(const std::vector<uchar>& bufferCfg,
                                        const std::vector<uchar>& bufferModel = std::vector<uchar>());

    /** @brief Reads a network model stored in <a href="https://pjreddie.com/darknet/">Darknet</a> model files.
     *  @param bufferCfg   A buffer contains a content of .cfg file with text description of the network architecture.
     *  @param lenCfg      Number of bytes to read from bufferCfg
     *  @param bufferModel A buffer contains a content of .weights file with learned network.
     *  @param lenModel    Number of bytes to read from bufferModel
     *  @returns Net object.
     */
    CV_EXPORTS cv::dnn::Net readNetFromDarknet(const char *bufferCfg, size_t lenCfg,
                                      const char *bufferModel = NULL, size_t lenModel = 0);

    /** @brief Reads a network model stored in <a href="http://caffe.berkeleyvision.org">Caffe</a> framework's format.
      * @param prototxt   path to the .prototxt file with text description of the network architecture.
      * @param caffeModel path to the .caffemodel file with learned network.
      * @returns Net object.
      */
    CV_EXPORTS_W cv::dnn::Net readNetFromCaffe(CV_WRAP_FILE_PATH const String& prototxt, CV_WRAP_FILE_PATH const String& caffeModel = String());

    /** @brief Reads a network model stored in Caffe model in memory.
      * @param bufferProto buffer containing the content of the .prototxt file
      * @param bufferModel buffer containing the content of the .caffemodel file
      * @returns Net object.
      */
    CV_EXPORTS_W cv::dnn::Net readNetFromCaffe(const std::vector<uchar>& bufferProto,
        const std::vector<uchar>& bufferModel = std::vector<uchar>());

    /** @brief Reads a network model stored in Caffe model in memory.
      * @details This is an overloaded member function, provided for convenience.
      * It differs from the above function only in what argument(s) it accepts.
      * @param bufferProto buffer containing the content of the .prototxt file
      * @param lenProto length of bufferProto
      * @param bufferModel buffer containing the content of the .caffemodel file
      * @param lenModel length of bufferModel
      * @returns Net object.
      */
    CV_EXPORTS cv::dnn::Net readNetFromCaffe(const char* bufferProto, size_t lenProto,
        const char* bufferModel = NULL, size_t lenModel = 0);

    /** @brief Convert all weights of Caffe network to half precision floating point.
     * @param src Path to origin model from Caffe framework contains single
     *            precision floating point weights (usually has `.caffemodel` extension).
     * @param dst Path to destination model with updated weights.
     * @param layersTypes Set of layers types which parameters will be converted.
     *                    By default, converts only Convolutional and Fully-Connected layers'
     *                    weights.
     *
     * @note Shrinked model has no origin float32 weights so it can't be used
     *       in origin Caffe framework anymore. However the structure of data
     *       is taken from NVidia's Caffe fork: https://github.com/NVIDIA/caffe.
     *       So the resulting model may be used there.
     */
    CV_EXPORTS_W void shrinkCaffeModel(CV_WRAP_FILE_PATH const String& src, CV_WRAP_FILE_PATH const String& dst,
        const std::vector<String>& layersTypes = std::vector<String>());
//! @}
}
}

#
#endif  /* OPENCV_DNNLEGACY_HPP_HPP */
