/*
Copyright 2020 OpenCV Foundation

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

#ifndef __OPENCV_BARCODE_HPP__
#define __OPENCV_BARCODE_HPP__

#include <opencv2/core.hpp>

namespace cv {
namespace barcode {
class CV_EXPORTS_W BarcodeDetector
{
public:
    CV_WRAP BarcodeDetector();
    
    ~BarcodeDetector();
    
    /** @brief Detects Barcode in image and returns the rectangle(s) containing the code.

    @param img grayscale or color (BGR) image containing (or not) Barcode.
    @param rects Output vector of rotated rectangle(s) containing the barcode.
     */
    CV_WRAP bool detect(InputArray img, CV_OUT std::vector <RotatedRect> &rects) const;
    
    /** @brief Decodes barcode in image once it's found by the detect() method.

    Returns UTF8-encoded output string(s) or empty string if the code cannot be decoded.
    @param img grayscale or color (BGR) image containing barcode.
    @param rects vector of rotated rectangle(s) found by detect() method (or some other algorithm).
    @param decoded_info UTF8-encoded output vector of string(s) or empty vector of string if the codes cannot be decoded.
     */
    CV_WRAP bool decode(InputArray img, const std::vector <RotatedRect> &rects,
                        CV_OUT std::vector <std::string> &decoded_info) const;
    
    /** @brief Both detects and decodes barcode

    @param img grayscale or color (BGR) image containing QR code.
    @param rects optional output vector of the found barcode rectangle(s). Will be empty if not found.
    @param decoded_info UTF8-encoded output vector of string(s) or empty vector of string if the codes cannot be decoded.
     */
    CV_WRAP bool detectAndDecode(InputArray img, CV_OUT std::vector <std::string> &decoded_info,
                                 CV_OUT std::vector <RotatedRect> &rects) const;

    /** @brief Decode without detects

    @param img grayscale or color (BGR) image containing barcode.
    @param decoded_info UTF8-encoded output vector of string(s) or empty vector of string if the codes do not contain barcode.
    */
    CV_WRAP bool docodeDirectly(InputArray img, CV_OUT std::vector <std::string> &decoded_info) const;

protected:
    struct Impl;
    Ptr<Impl> p;
};
}} // cv::barcode::
#endif //__OPENCV_BARCODE_HPP__