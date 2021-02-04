// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Tencent is pleased to support the open source community by making WeChat QRCode available.
// Copyright (C) 2020 THL A29 Limited, a Tencent company. All rights reserved.

#ifndef __OPENCV_WECHAT_QRCODE_HPP__
#define __OPENCV_WECHAT_QRCODE_HPP__
#include "opencv2/core.hpp"
/** @defgroup wechat_qrcode WeChat QR code detector for detecting and parsing QR code.
 */
namespace cv {
namespace wechat_qrcode {
//! @addtogroup wechat_qrcode
//! @{
/**
 * @brief  WeChat QRCode includes two CNN-based models:
 * A object detection model and a super resolution model.
 * Object detection model is applied to detect QRCode with the bounding box.
 * super resolution model is applied to zoom in QRCode when it is small.
 *
 */
class CV_EXPORTS_W WeChatQRCode {
public:
    /**
     * @brief Initialize the WeChatQRCode.
     * It includes two models, which are packaged with caffe format.
     * Therefore, there are prototxt and caffe models (In total, four paramenters).
     *
     * @param detector_prototxt_path prototxt file path for the detector
     * @param detector_caffe_model_path caffe model file path for the detector
     * @param super_resolution_prototxt_path prototxt file path for the super resolution model
     * @param super_resolution_caffe_model_path caffe file path for the super resolution model
     */
    CV_WRAP WeChatQRCode(const std::string& detector_prototxt_path = "",
                         const std::string& detector_caffe_model_path = "",
                         const std::string& super_resolution_prototxt_path = "",
                         const std::string& super_resolution_caffe_model_path = "");
    ~WeChatQRCode(){};

    /**
     * @brief  Both detects and decodes QR code.
     * To simplify the usage, there is a only API: detectAndDecode
     *
     * @param img supports grayscale or color (BGR) image.
     * @param points optional output array of vertices of the found QR code quadrangle. Will be
     * empty if not found.
     * @return list of decoded string.
     */
    CV_WRAP std::vector<std::string> detectAndDecode(InputArray img,
                                                     OutputArrayOfArrays points = noArray());

protected:
    class Impl;
    Ptr<Impl> p;
};

//! @}
}  // namespace wechat_qrcode
}  // namespace cv
#endif  // __OPENCV_WECHAT_QRCODE_HPP__
