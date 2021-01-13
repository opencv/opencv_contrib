// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Tencent is pleased to support the open source community by making WeChat QRCode available.
// Copyright (C) 2020 THL A29 Limited, a Tencent company. All rights reserved.

#ifndef __OPENCV_WECHAT_QRCODE_HPP__
#define __OPENCV_WECHAT_QRCODE_HPP__
#include "opencv2/core.hpp"
#include "opencv2/wechat_qrcode/align.hpp"
#include "opencv2/wechat_qrcode/ssd_detector.hpp"
#include "opencv2/wechat_qrcode/super_scale.hpp"

/** @defgroup wechat_qrcode WeChat QR code detector for detecting and parsing QR code.
*/
namespace cv {
namespace wechat_qrcode {
using std::string;
using std::vector;
//! @addtogroup wechat_qrcode
//! @{

/**
 * @brief  QRCodeDetector includes two CNN-based models:
 * A object detection model and a super resolution model.
 * Object detection model is applied to detect QRCode with the bounding box.
 * super resolution model is applied to zoom in QRCode when it is small.
 *
 */
class CV_EXPORTS_W QRCodeDetector {
public:
    /**
     * @brief Initialize the QRCodeDetector.
     * Two models are packaged with caffe format.
     * Therefore, there are prototxt and caffe model two files.
     *
     * @param detector_prototxt_path prototxt file path for the detector
     * @param detector_caffe_model_path caffe model file path for the detector
     * @param super_resolution_prototxt_path prototxt file path for the super resolution model
     * @param super_resolution_caffe_model_path caffe file path for the super resolution model
     */
    CV_WRAP QRCodeDetector(const String& detector_prototxt_path = "",
                           const String& detector_caffe_model_path = "",
                           const String& super_resolution_prototxt_path = "",
                           const String& super_resolution_caffe_model_path = "");
    ~QRCodeDetector(){};

    /**
     * @brief  Both detects and decodes QR code. 
     * To simplify the usage, there is a only API: detectAndDecode
     *
     * @param img supports grayscale or color (BGR) image.
     * @param points optional output array of vertices of the found QR code quadrangle. Will be
     * empty if not found.
     * @return list of decoded string.
     */
    CV_WRAP vector<string> detectAndDecode(InputArray img, OutputArrayOfArrays points = noArray());

private:
    /**
     * @brief detect QR codes from the given image
     *
     * @param img supports grayscale or color (BGR) image.
     * @return vector<Mat> detected QR code bounding boxes.
     */
    vector<Mat> detect(const Mat& img);
    /**
     * @brief decode QR codes from detected points
     *
     * @param img supports grayscale or color (BGR) image.
     * @param candidate_points detected points. we name it "candidate points" which means no
     * all the qrcode can be decoded.
     * @param points succussfully decoded qrcode with bounding box points.
     * @return vector<string>
     */
    vector<string> decode(const Mat& img, vector<Mat>& candidate_points, vector<Mat>& points);
    int applyDetector(const Mat& img, vector<Mat>& points);
    Mat cropObj(const Mat& img, const Mat& point, Align& aligner);
    vector<float> getScaleList(const int width, const int height);
    std::shared_ptr<SSDDetector> detector_;
    std::shared_ptr<SuperScale> super_resolution_model_;
    bool use_nn_detector_, use_nn_sr_;
};

//! @}
}  // namespace wechat_qrcode
}  // namespace cv
#endif  // __OPENCV_WECHAT_QRCODE_HPP__
