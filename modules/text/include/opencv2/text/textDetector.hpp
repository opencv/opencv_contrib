/*M//////////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                          License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
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

#ifndef __OPENCV_TEXT_TEXTDETECTOR_HPP__
#define __OPENCV_TEXT_TEXTDETECTOR_HPP__

#include <vector>
#include <string>
#include <iostream>
#include <sstream>
#include"ocr.hpp"


namespace cv
{
namespace text
{

//! @addtogroup text_recognize
//! @{



//base class BaseDetector declares a common API that would be used in a typical text
//detection scenario
class CV_EXPORTS_W BaseDetector
{
 public:
    virtual ~BaseDetector() {};

    virtual void run(Mat& image,
                     std::vector<Rect>* component_rects=NULL,                     
                     std::vector<float>* component_confidences=NULL,
                     int component_level=0) = 0;

    virtual void run(Mat& image, Mat& mask,
                     std::vector<Rect>* component_rects=NULL,                     
                     std::vector<float>* component_confidences=NULL,
                     int component_level=0) = 0;


class CV_EXPORTS_W textDetector : public BaseDetector
{
public:
    virtual void run(Mat& image,  std::vector<Rect>* component_rects=NULL,
                     std::vector<float>* component_confidences=NULL,
                     int component_level=OCR_LEVEL_WORD)=0;

    /** @brief detect text with a cnn, input is one image with (multiple) ocuurance of text.

    Takes image on input and returns recognized text in the output_text parameter. Optionally
    provides also the Rects for individual text elements found (e.g. words), and the list of those
    text elements with their confidence values.

    @param image Input image CV_8UC1 or CV_8UC3

    @param mask is totally ignored and is only available for compatibillity reasons


    @param component_rects a vector of Rects, each rect is one text bounding box.



    @param component_confidences A vector of float returns confidence of text bounding boxes

    @param component_level must be OCR_LEVEL_WORD.
     */

    virtual void run(Mat& image, Mat& mask, std::vector<Rect>* component_rects=NULL,
                     std::vector<float>* component_confidences=NULL,
                     int component_level=OCR_LEVEL_WORD)=0;


    /**
    @brief Method that provides a quick and simple interface to detect text inside an image

    @param inputImage an image expected to be a CV_U8C3 of any size

    @param Bbox a vector of Rect that will store the detected word bounding box

    @param confidence a vector of float that will be updated with the confidence the classifier has for the selected bounding box
    */
    CV_WRAP virtual void textDetectInImage(InputArray inputImage,CV_OUT std::vector<Rect>& Bbox,CV_OUT std::vector<float>& confidence)=0;




   /** @brief simple getter for the preprocessing functor
     */
    CV_WRAP virtual Ptr<TextImageClassifier> getClassifier()=0;

    /** @brief Creates an instance of the textDetector class.

    @param classifierPtr an instance of TextImageClassifier, normaly a DeepCNN instance


     */
    CV_WRAP static Ptr<textDetector> create(Ptr<TextImageClassifier> classifierPtr);


    /** @brief Creates an instance of the textDetector class and implicitly also a DeepCNN classifier.

    @param modelArchFilename the relative or absolute path to the prototxt file describing the classifiers architecture.

    @param modelWeightsFilename the relative or absolute path to the file containing the pretrained weights of the model in caffe-binary form.


    */
    CV_WRAP static Ptr<textDetector> create(String modelArchFilename, String modelWeightsFilename);


};


}//namespace text
}//namespace cv


#endif // _OPENCV_TEXT_OCR_HPP_
