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
//recognition scenario
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

    /** @brief Main functionality of the OCR Hierarchy. Subclasses provide
     * default parameters for all parameters other than the input image.
     */
//    virtual std::vector<Rect>* run(InputArray image){
//        //std::string res;
//        std::vector<Rect> component_rects;
//        std::vector<float> component_confidences;
//        //std::vector<std::string> component_texts;
//        Mat inputImage=image.getMat();
//        this->run(inputImage,&component_rects,
//                  &component_confidences,OCR_LEVEL_WORD);
//        return *component_rects;
//    }

};


//Classifiers should provide diferent backends
//For the moment only caffe is implemeted
//enum{
//    OCR_HOLISTIC_BACKEND_NONE,
//    OCR_HOLISTIC_BACKEND_CAFFE
//};





/** @brief OCRHolisticWordRecognizer class provides the functionallity of segmented wordspotting.
 * Given a predefined vocabulary , a TextImageClassifier is employed to select the most probable
 * word given an input image.
 *
 * This class implements the logic of providing transcriptions given a vocabulary and and an image
 * classifer. The classifier has to be any TextImageClassifier but the classifier for which this
 * class was built is the DictNet. In order to load it the following files should be downloaded:

 * <http://nicolaou.homouniversalis.org/assets/vgg_text/dictnet_vgg_deploy.prototxt>
 * <http://nicolaou.homouniversalis.org/assets/vgg_text/dictnet_vgg.caffemodel>
 * <http://nicolaou.homouniversalis.org/assets/vgg_text/dictnet_vgg_labels.txt>
 */
class CV_EXPORTS_W textDetector : public BaseDetector
{
public:
    virtual void run(Mat& image,  std::vector<Rect>* component_rects=NULL,
                     std::vector<float>* component_confidences=NULL,
                     int component_level=OCR_LEVEL_WORD)=0;

    /** @brief Recognize text using a segmentation based word-spotting/classifier cnn.

    Takes image on input and returns recognized text in the output_text parameter. Optionally
    provides also the Rects for individual text elements found (e.g. words), and the list of those
    text elements with their confidence values.

    @param image Input image CV_8UC1 or CV_8UC3

    @param mask is totally ignored and is only available for compatibillity reasons

    @param output_text Output text of the the word spoting, always one that exists in the dictionary.

    @param component_rects Not applicable for word spotting can be be NULL if not, a single elemnt will
        be put in the vector.

    @param component_texts Not applicable for word spotting can be be NULL if not, a single elemnt will
        be put in the vector.

    @param component_confidences Not applicable for word spotting can be be NULL if not, a single elemnt will
        be put in the vector.

    @param component_level must be OCR_LEVEL_WORD.
     */

    virtual void run(Mat& image, Mat& mask, std::vector<Rect>* component_rects=NULL,
                     std::vector<float>* component_confidences=NULL,
                     int component_level=OCR_LEVEL_WORD)=0;


    /**
    @brief Method that provides a quick and simple interface to a single word image classifcation

    @param inputImage an image expected to be a CV_U8C1 or CV_U8C3 of any size

    @param transcription an opencv string that will store the detected word transcription

    @param confidence a double that will be updated with the confidence the classifier has for the selected word
    */
    CV_WRAP virtual void textDetectInImage(InputArray inputImage,CV_OUT std::vector<Rect>& Bbox,CV_OUT std::vector<float>& confidence)=0;

    /**
    @brief Method that provides a quick and simple interface to a multiple word image classifcation taking advantage
    the classifiers parallel capabilities.

    @param inputImageList an list of images expected to be a CV_U8C1 or CV_U8C3 each image can be of any size and is assumed
    to contain a single word.

    @param transcriptions a vector of opencv strings that will store the detected word transcriptions, one for each
    input image

    @param confidences a vector of double that will be updated with the confidence the classifier has for each of the
    selected words.
    */
    //CV_WRAP virtual void recogniseImageBatch(InputArrayOfArrays inputImageList,CV_OUT std::vector<String>& transcriptions,CV_OUT std::vector<double>& confidences)=0;


   /** @brief simple getter for the preprocessing functor
     */
    CV_WRAP virtual Ptr<TextImageClassifier> getClassifier()=0;

    /** @brief Creates an instance of the OCRHolisticWordRecognizer class.

    @param classifierPtr an instance of TextImageClassifier, normaly a DeepCNN instance

    @param vocabularyFilename the relative or absolute path to the file containing all words in the vocabulary. Each text line
    in the file is assumed to be a single word. The number of words in the vocabulary must be exactly the same as the outputSize
    of the classifier.
     */
    CV_WRAP static Ptr<textDetector> create(Ptr<TextImageClassifier> classifierPtr);


    /** @brief Creates an instance of the OCRHolisticWordRecognizer class and implicitly also a DeepCNN classifier.

    @param modelArchFilename the relative or absolute path to the prototxt file describing the classifiers architecture.

    @param modelWeightsFilename the relative or absolute path to the file containing the pretrained weights of the model in caffe-binary form.

    @param vocabularyFilename the relative or absolute path to the file containing all words in the vocabulary. Each text line
    in the file is assumed to be a single word. The number of words in the vocabulary must be exactly the same as the outputSize
    of the classifier.
    */
    CV_WRAP static Ptr<textDetector> create(String modelArchFilename, String modelWeightsFilename);

    /** @brief
     *
     * @param classifierPtr
     *
     * @param vocabulary
     */
 //   CV_WRAP static Ptr<textDetectImage> create(Ptr<TextImageClassifier> classifierPtr,const std::vector<String>& vocabulary);

    /** @brief
     *
     * @param modelArchFilename
     *
     * @param modelWeightsFilename
     *
     * @param vocabulary
     */
 //   CV_WRAP static Ptr<textDetectImage> create (String modelArchFilename, String modelWeightsFilename, const std::vector<String>& vocabulary);
};


}//namespace text
}//namespace cv


#endif // _OPENCV_TEXT_OCR_HPP_
