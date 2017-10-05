// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef __OPENCV_TEXT_TEXTDETECTOR_HPP__
#define __OPENCV_TEXT_TEXTDETECTOR_HPP__

#include"ocr.hpp"

namespace cv
{
namespace text
{

//! @addtogroup text_detect
//! @{

/** @brief An abstract class providing interface for text detection algorithms
 */
class CV_EXPORTS_W TextDetector
{
public:
    /**
    @brief Method that provides a quick and simple interface to detect text inside an image

    @param inputImage an image to process
    @param Bbox a vector of Rect that will store the detected word bounding box
    @param confidence a vector of float that will be updated with the confidence the classifier has for the selected bounding box
    */
    virtual void textDetectInImage(InputArray inputImage, CV_OUT std::vector<Rect>& Bbox, CV_OUT std::vector<float>& confidence) = 0;
    virtual ~TextDetector() {}
};

/** @brief TextDetectorCNN class provides the functionallity of text bounding box detection.
 * A TextDetectorCNN is employed to find bounding boxes of text words given an input image.
 */
class CV_EXPORTS_W TextDetectorCNN : public TextDetector
{
public:
    /**
    @overload

    @param inputImage an image expected to be a CV_U8C3 of any size
    @param Bbox a vector of Rect that will store the detected word bounding box
    @param confidence a vector of float that will be updated with the confidence the classifier has for the selected bounding box
    */
    CV_WRAP virtual void textDetectInImage(InputArray inputImage, CV_OUT std::vector<Rect>& Bbox, CV_OUT std::vector<float>& confidence) = 0;

    /** @brief Creates an instance of the textDetector class and implicitly also a DeepCNN classifier.

    @param modelArchFilename the relative or absolute path to the prototxt file describing the classifiers architecture.
    @param modelWeightsFilename the relative or absolute path to the file containing the pretrained weights of the model in caffe-binary form.
    @param detectMultiscale if true, multiple scales of the input image will be used as network input
    */
    CV_WRAP static Ptr<TextDetectorCNN> create(const String& modelArchFilename, const String& modelWeightsFilename, bool detectMultiscale = false);
};

//! @}
}//namespace text
}//namespace cv


#endif // _OPENCV_TEXT_OCR_HPP_
