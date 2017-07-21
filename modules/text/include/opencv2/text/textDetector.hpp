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

};
/** A virtual class for different models of text detection (including CNN based deep models)
 */

class CV_EXPORTS_W TextRegionDetector
{
protected:
    /** Stores input and output size
     */
    //netGeometry inputGeometry_;
    //netGeometry outputGeometry_;
    Size inputGeometry_;
    Size outputGeometry_;
    int inputChannelCount_;
    int outputChannelCount_;

public:
    virtual ~TextRegionDetector() {}

    /** @brief produces a list of Bounding boxes and an estimate of text-ness confidence of Bounding Boxes
     */
    CV_WRAP virtual void detect(InputArray image, OutputArray bboxProb ) = 0;


    /** @brief simple getter method returning the size (height, width) of the input sample
     */
    CV_WRAP virtual Size  getInputGeometry(){return this->inputGeometry_;}

    /** @brief simple getter method returning the shape of the oputput
     *   Any text detector should output a number of text regions alongwith a score of text-ness
     *   From the shape it can be inferred the number of text regions and number of returned value
     *   for each region
     */
    CV_WRAP virtual Size getOutputGeometry(){return this->outputGeometry_;}



};

/** Generic structure of Deep CNN based Text Detectors
 * */
class CV_EXPORTS_W  DeepCNNTextDetector : public TextRegionDetector
{
    /** @brief Class that uses a pretrained caffe model for text detection.
     * Any text detection should
     * This network is described in detail in:
     * Minghui Liao et al.: TextBoxes: A Fast Text Detector with a Single Deep Neural Network
     * https://arxiv.org/abs/1611.06779
     */
protected:
    /** all deep CNN based text detectors have a preprocessor (normally)
         */
    Ptr<ImagePreprocessor> preprocessor_;
    /** @brief all image preprocessing is handled here including whitening etc.
         *
         *  @param input the image to be preprocessed for the classifier. If the depth
         * is CV_U8 values should be in [0,255] otherwise values are assumed to be in [0,1]
         *
         * @param output reference to the image to be fed to the classifier, the preprocessor will
         * resize the image to the apropriate size and convert it to the apropriate depth\
         *
         * The method preprocess should never be used externally, it is up to classify and classifyBatch
         * methods to employ it.
         */
    virtual void preprocess(const Mat& input,Mat& output);
public:
    virtual ~DeepCNNTextDetector() {};

    /** @brief Constructs a DeepCNNTextDetector object from a caffe pretrained model
     *
     * @param archFilename is the path to the prototxt file containing the deployment model architecture description.
     *
     * @param weightsFilename is the path to the pretrained weights of the model in binary fdorm.
     *
     * @param preprocessor is a pointer to the instance of a ImagePreprocessor implementing the preprocess_ protecteed method;
     *
     * @param minibatchSz the maximum number of samples that can processed in parallel. In practice this parameter
     * has an effect only when computing in the GPU and should be set with respect to the memory available in the GPU.
     *
     * @param backEnd integer parameter selecting the coputation framework. For now OCR_HOLISTIC_BACKEND_CAFFE is
     * the only option
     */
    CV_WRAP static Ptr<DeepCNNTextDetector> create(String archFilename,String weightsFilename,Ptr<ImagePreprocessor> preprocessor,int minibatchSz=100,int backEnd=OCR_HOLISTIC_BACKEND_CAFFE);

    /** @brief Constructs a DeepCNNTextDetector intended to be used for text area detection.
     *
     * This method loads a pretrained classifier and couples with a preprocessor that preprocess the image with mean subtraction of ()
     * The architecture and models weights can be downloaded from:
     * https://github.com/sghoshcvc/TextBox-Models.git (size is around 100 MB)

     * @param archFilename is the path to the prototxt file containing the deployment model architecture description.
     * When employing OCR_HOLISTIC_BACKEND_CAFFE this is the path to the deploy ".prototxt".
     *
     * @param weightsFilename is the path to the pretrained weights of the model. When employing
     * OCR_HOLISTIC_BACKEND_CAFFE this is the path to the ".caffemodel" file.
     *
     * @param backEnd integer parameter selecting the coputation framework. For now OCR_HOLISTIC_BACKEND_CAFFE is
     * the only option
     */
    CV_WRAP static Ptr<DeepCNNTextDetector> createTextBoxNet(String archFilename,String weightsFilename,int backEnd=OCR_HOLISTIC_BACKEND_CAFFE);
    friend class ImagePreprocessor;

};

/** @brief textDetector class provides the functionallity of text bounding box detection.
 * A TextRegionDetector is employed to find bounding boxes of text
 * words given an input image.
 *
 * This class implements the logic of providing text bounding boxes in a vector of rects given an TextRegionDetector
 * The TextRegionDetector can be any text detector
 *
 */

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
    CV_WRAP virtual Ptr<TextRegionDetector> getClassifier()=0;

    /** @brief Creates an instance of the textDetector class.

    @param classifierPtr an instance of TextImageClassifier, normaly a DeepCNN instance


     */
    CV_WRAP static Ptr<textDetector> create(Ptr<TextRegionDetector> classifierPtr);


    /** @brief Creates an instance of the textDetector class and implicitly also a DeepCNN classifier.

    @param modelArchFilename the relative or absolute path to the prototxt file describing the classifiers architecture.

    @param modelWeightsFilename the relative or absolute path to the file containing the pretrained weights of the model in caffe-binary form.


    */
    CV_WRAP static Ptr<textDetector> create(String modelArchFilename, String modelWeightsFilename);


};


}//namespace text
}//namespace cv


#endif // _OPENCV_TEXT_OCR_HPP_
