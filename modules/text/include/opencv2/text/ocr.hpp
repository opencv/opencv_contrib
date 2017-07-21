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

#ifndef __OPENCV_TEXT_OCR_HPP__
#define __OPENCV_TEXT_OCR_HPP__

#include <vector>
#include <string>
#include <iostream>
#include <sstream>



namespace cv
{
namespace text
{

//! @addtogroup text_recognize
//! @{

enum
{
    OCR_LEVEL_WORD,
    OCR_LEVEL_TEXTLINE
};

//! Tesseract.PageSegMode Enumeration
enum page_seg_mode
{
    PSM_OSD_ONLY,
    PSM_AUTO_OSD,
    PSM_AUTO_ONLY,
    PSM_AUTO,
    PSM_SINGLE_COLUMN,
    PSM_SINGLE_BLOCK_VERT_TEXT,
    PSM_SINGLE_BLOCK,
    PSM_SINGLE_LINE,
    PSM_SINGLE_WORD,
    PSM_CIRCLE_WORD,
    PSM_SINGLE_CHAR
};

//! Tesseract.OcrEngineMode Enumeration
enum ocr_engine_mode
{
    OEM_TESSERACT_ONLY,
    OEM_CUBE_ONLY,
    OEM_TESSERACT_CUBE_COMBINED,
    OEM_DEFAULT
};

//base class BaseOCR declares a common API that would be used in a typical text recognition scenario

class CV_EXPORTS_W BaseOCR
{
 public:
    virtual ~BaseOCR() {};

    virtual void run(Mat& image, std::string& output_text,
                     std::vector<Rect>* component_rects=NULL,
                     std::vector<std::string>* component_texts=NULL,
                     std::vector<float>* component_confidences=NULL,
                     int component_level=0) = 0;

    virtual void run(Mat& image, Mat& mask, std::string& output_text,
                     std::vector<Rect>* component_rects=NULL,
                     std::vector<std::string>* component_texts=NULL,
                     std::vector<float>* component_confidences=NULL,
                     int component_level=0) = 0;

    /** @brief Main functionality of the OCR Hierarchy. Subclasses provide
     * default parameters for all parameters other than the input image.
     */
    virtual String run(InputArray image){
        std::string res;
        std::vector<Rect> component_rects;
        std::vector<float> component_confidences;
        std::vector<std::string> component_texts;
        Mat inputImage=image.getMat();
        this->run(inputImage,res,&component_rects,&component_texts,
                  &component_confidences,OCR_LEVEL_WORD);
        return res;
    }

};

/** @brief OCRTesseract class provides an interface with the tesseract-ocr API
 * (v3.02.02) in C++.

Notice that it is compiled only when tesseract-ocr is correctly installed.

@note
   -   (C++) An example of OCRTesseract recognition combined with scene text
        detection can be found at the end_to_end_recognition demo:
        <https://github.com/Itseez/opencv_contrib/blob/master/modules/text/samples/end_to_end_recognition.cpp>
    -   (C++) Another example of OCRTesseract recognition combined with scene
        text detection can be found at the webcam_demo:
        <https://github.com/Itseez/opencv_contrib/blob/master/modules/text/samples/webcam_demo.cpp>
 */
class CV_EXPORTS_W OCRTesseract : public BaseOCR
{
public:
    /** @brief Recognize text using the tesseract-ocr API.

    Takes image on input and returns recognized text in the output_text
    parameter. Optionally provides also the Rects for individual text elements
    found (e.g. words), and the list of those text elements with their
    confidence values.

    @param image Input image CV_8UC1 or CV_8UC3

    @param output_text Output text of the tesseract-ocr.

    @param component_rects If provided the method will output a list of Rects
    for the individual text elements found (e.g. words or text lines).

    @param component_texts If provided the method will output a list of text
    strings for the recognition of individual text elements found (e.g. words or
    text lines).

    @param component_confidences If provided the method will output a list of
    confidence values for the recognition of individual text elements found
    (e.g. words or text lines).

    @param component_level OCR_LEVEL_WORD (by default), or OCR_LEVEL_TEXT_LINE.
     */
    virtual void run (Mat& image, std::string& output_text,
                     std::vector<Rect>* component_rects=NULL,
                     std::vector<std::string>* component_texts=NULL,
                     std::vector<float>* component_confidences=NULL,
                     int component_level=0);

    virtual void run (Mat& image, Mat& mask, std::string& output_text,
                      std::vector<Rect>* component_rects=NULL,
                      std::vector<std::string>* component_texts=NULL,
                      std::vector<float>* component_confidences=NULL,
                      int component_level=0);

    // aliases for scripting
    CV_WRAP String run (InputArray image, int min_confidence,
                        int component_level=0);

    CV_WRAP String run(InputArray image, InputArray mask,
                       int min_confidence, int component_level=0);

    CV_WRAP virtual void setWhiteList(const String& char_whitelist) = 0;


    /** @brief Creates an instance of the OCRTesseract class. Initializes Tesseract.

    @param datapath the name of the parent directory of tessdata ended with "/", or NULL to use the
    system's default directory.
    @param language an ISO 639-3 code or NULL will default to "eng".
    @param char_whitelist specifies the list of characters used for recognition. NULL defaults to
    "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ".
    @param oem tesseract-ocr offers different OCR Engine Modes (OEM), by default
    tesseract::OEM_DEFAULT is used. See the tesseract-ocr API documentation for other possible
    values.
    @param psmode tesseract-ocr offers different Page Segmentation Modes (PSM) tesseract::PSM_AUTO
    (fully automatic layout analysis) is used. See the tesseract-ocr API documentation for other
    possible values.
     */
    CV_WRAP static Ptr<OCRTesseract> create(const char* datapath=NULL, const char* language=NULL,
                                    const char* char_whitelist=NULL, int oem=OEM_DEFAULT, int psmode=PSM_AUTO);

};


/* OCR HMM Decoder */

enum decoder_mode
{
    OCR_DECODER_VITERBI = 0 // Other algorithms may be added
};

/* OCR classifier type*/
enum classifier_type
{
    OCR_KNN_CLASSIFIER = 0,
    OCR_CNN_CLASSIFIER = 1
};

/** @brief OCRHMMDecoder class provides an interface for OCR using Hidden Markov Models.


 * @note
 * -   (C++) An example on using OCRHMMDecoder recognition combined with scene
 *     text detection can be found at the webcam_demo sample:
 *      <https://github.com/Itseez/opencv_contrib/blob/master/modules/text/samples/webcam_demo.cpp>
 */
class CV_EXPORTS_W OCRHMMDecoder : public BaseOCR {
 public:

    /** @brief Callback with the character classifier is made a class.

    * This way it hides the feature extractor and the classifier itself, so
    * developers can write their own OCR code.

    The default character classifier and feature extractor can be loaded using the utility function
    loadOCRHMMClassifierNM and KNN model provided in
    <https://github.com/opencv/opencv_contrib/blob/master/modules/text/samples/OCRHMM_knn_model_data.xml.gz>.
     */
    class CV_EXPORTS_W ClassifierCallback
    {
    public:

        virtual ~ClassifierCallback() { }
        /** @brief The character classifier must return a (ranked list of)
         * class(es) id('s)

         * @param image Input image CV_8UC1 or CV_8UC3 with a single letter.
         * @param out_class The classifier returns the character class
         * categorical label, or list of class labels, to which the input image
         * corresponds.

         * @param out_confidence The classifier returns the probability of the
         * input image corresponding to each classes in out_class.
         */
        virtual void eval (InputArray image, std::vector<int>& out_class,
                           std::vector<double>& out_confidence);
    };

    /** @brief Recognize text using HMM.

    * Takes binary image on input and returns recognized text in the output_text
    * parameter. Optionally provides also the Rects for individual text elements
    * found (e.g. words), and the list of those text elements with their
    * confidence values.

    * @param image Input binary image CV_8UC1 with a single text line (or word).

    * @param output_text Output text. Most likely character sequence found by
    * the HMM decoder.

    * @param component_rects If provided the method will output a list of Rects
    * for the individual text elements found (e.g. words).

    * @param component_texts If provided the method will output a list of text
    * strings for the recognition of individual text elements found (e.g. words).

    * @param component_confidences If provided the method will output a list of
    * confidence values for the recognition of individual text elements found
    * (e.g. words).

    * @param component_level Only OCR_LEVEL_WORD is supported.
    */
    virtual void run (Mat& image, std::string& output_text,
                      std::vector<Rect>* component_rects=NULL,
                      std::vector<std::string>* component_texts=NULL,
                      std::vector<float>* component_confidences=NULL,
                      int component_level=0);

    /** @brief Recognize text using HMM.

    * Takes an image and a mask (where each connected component corresponds to a
    * segmented character) on input and returns recognized text in the
    * output_text parameter. Optionally provides also the Rects for individual
    * text elements found (e.g. words), and the list of those text elements with
    * their confidence values.

    * @param image Input image CV_8UC1 or CV_8UC3 with a single text line
    * (or word).

    * @param mask Input binary image CV_8UC1 same size as input image. Each
    * connected component in mask corresponds to a segmented character in the
    * input image.

    * @param output_text Output text. Most likely character sequence found by
    * the HMM decoder.

    * @param component_rects If provided the method will output a list of Rects
    * for the individual text elements found (e.g. words).

    * @param component_texts If provided the method will output a list of text
    * strings for the recognition of individual text elements found (e.g. words).

    * @param component_confidences If provided the method will output a list of
    * confidence values for the recognition of individual text elements found
    * (e.g. words).

    * @param component_level Only OCR_LEVEL_WORD is supported.
    */
    virtual void run(Mat& image, Mat& mask, std::string& output_text,
                     std::vector<Rect>* component_rects=NULL,
                     std::vector<std::string>* component_texts=NULL,
                     std::vector<float>* component_confidences=NULL,
                     int component_level=0);

    // aliases for scripting
    CV_WRAP String run(InputArray image,
                       int min_confidence,
                       int component_level=0);

    CV_WRAP String run(InputArray image,
                       InputArray mask,
                       int min_confidence,
                       int component_level=0);

    /** @brief Creates an instance of the OCRHMMDecoder class. Initializes
     * HMMDecoder.

     * @param classifier The character classifier with built in feature
     * extractor.

     * @param vocabulary The language vocabulary (chars when ascii english text)
     * . vocabulary.size() must be equal to the number of classes of the
     * classifier.

     * @param transition_probabilities_table Table with transition probabilities
     * between character pairs. cols == rows == vocabulary.size().

     * @param emission_probabilities_table Table with observation emission
     * probabilities. cols == rows == vocabulary.size().

     * @param mode HMM Decoding algorithm. Only OCR_DECODER_VITERBI is available
     * for the moment (<http://en.wikipedia.org/wiki/Viterbi_algorithm>).
     */

    static Ptr<OCRHMMDecoder> create(const Ptr<OCRHMMDecoder::ClassifierCallback> classifier,// The character classifier with built in feature extractor
                                     const std::string& vocabulary,                    // The language vocabulary (chars when ASCII English text)
                                                                                       //     size() must be equal to the number of classes
                                     InputArray transition_probabilities_table,        // Table with transition probabilities between character pairs
                                                                                       //     cols == rows == vocabulary.size()
                                     InputArray emission_probabilities_table,          // Table with observation emission probabilities
                                                                                       //     cols == rows == vocabulary.size()
                                     decoder_mode mode = OCR_DECODER_VITERBI);         // HMM Decoding algorithm (only Viterbi for the moment)

    CV_WRAP static Ptr<OCRHMMDecoder> create(const Ptr<OCRHMMDecoder::ClassifierCallback> classifier,// The character classifier with built in feature extractor
                                     const String& vocabulary,                    // The language vocabulary (chars when ASCII English text)
                                                                                       //     size() must be equal to the number of classes
                                     InputArray transition_probabilities_table,        // Table with transition probabilities between character pairs
                                                                                       //     cols == rows == vocabulary.size()
                                     InputArray emission_probabilities_table,          // Table with observation emission probabilities
                                                                                       //     cols == rows == vocabulary.size()
                                     int mode = OCR_DECODER_VITERBI);         // HMM Decoding algorithm (only Viterbi for the moment)

    /** @brief Creates an instance of the OCRHMMDecoder class. Loads and initializes HMMDecoder from the specified path

     @overload
     */
    CV_WRAP static Ptr<OCRHMMDecoder> create(const String& filename,

                                     const String& vocabulary,                    // The language vocabulary (chars when ASCII English text)
                                                                                       //     size() must be equal to the number of classes
                                     InputArray transition_probabilities_table,        // Table with transition probabilities between character pairs
                                                                                       //     cols == rows == vocabulary.size()
                                     InputArray emission_probabilities_table,          // Table with observation emission probabilities
                                                                                       //     cols == rows == vocabulary.size()
                                     int mode = OCR_DECODER_VITERBI,                    // HMM Decoding algorithm (only Viterbi for the moment)

                                     int classifier = OCR_KNN_CLASSIFIER);              // The character classifier type
protected:

    Ptr<OCRHMMDecoder::ClassifierCallback> classifier;
    std::string vocabulary;
    Mat transition_p;
    Mat emission_p;
    decoder_mode mode;
};

/** @brief Allow to implicitly load the default character classifier when
 * creating an OCRHMMDecoder object.

 @param filename The XML or YAML file with the classifier model (e.g.OCRHMM_knn_model_data.xml)


The KNN default classifier is based in the scene text recognition method proposed by Lukás Neumann &
Jiri Matas in [Neumann11b]. Basically, the region (contour) in the input image is normalized to a
fixed size, while retaining the centroid and aspect ratio, in order to extract a feature vector
based on gradient orientations along the chain-code of its perimeter. Then, the region is classified
using a KNN model trained with synthetic data of rendered characters with different standard font
types.

@deprecated loadOCRHMMClassifier instead

 */
CV_EXPORTS_W Ptr<OCRHMMDecoder::ClassifierCallback> loadOCRHMMClassifierNM (
        const String& filename);

/** @brief Allow to implicitly load the default character classifier when
 * creating an OCRHMMDecoder object.

 @param filename The XML or YAML file with the classifier model (e.g.OCRBeamSearch_CNN_model_data.xml.gz)


@param filename The XML or YAML file with the classifier model (e.g. OCRBeamSearch_CNN_model_data.xml.gz)

The CNN default classifier is based in the scene text recognition method proposed by Adam Coates &
Andrew NG in [Coates11a]. The character classifier consists in a Single Layer Convolutional Neural Network and
a linear classifier. It is applied to the input image in a sliding window fashion, providing a set of recognitions
at each window location.

@deprecated use loadOCRHMMClassifier instead

 */
CV_EXPORTS_W Ptr<OCRHMMDecoder::ClassifierCallback> loadOCRHMMClassifierCNN (
        const String& filename);

/** @brief Allow to implicitly load the default character classifier when creating an OCRHMMDecoder object.

 @param filename The XML or YAML file with the classifier model (e.g. OCRBeamSearch_CNN_model_data.xml.gz)

 @param classifier Can be one of classifier_type enum values.

 */
CV_EXPORTS_W Ptr<OCRHMMDecoder::ClassifierCallback> loadOCRHMMClassifier(const String& filename, int classifier);
//! @}


/** @brief Utility function to create a tailored language model transitions table from a given list of words (lexicon).
 *
 * @param vocabulary The language vocabulary (chars when ASCII English text).
 *
 * @param lexicon The list of words that are expected to be found in a particular image.

 * @param transition_probabilities_table Output table with transition
 * probabilities between character pairs. cols == rows == vocabulary.size().

 * The function calculate frequency statistics of character pairs from the given
 * lexicon and fills the output transition_probabilities_table with them. The
 * transition_probabilities_table can be used as input in the
 * OCRHMMDecoder::create() and OCRBeamSearchDecoder::create() methods.
 * @note
 *    -   (C++) An alternative would be to load the default generic language
 *        transition table provided in the text module samples folder (created
 *        from ispell 42869 english words list) :
 *            <https://github.com/Itseez/opencv_contrib/blob/master/modules/text/samples/OCRHMM_transitions_table.xml>
 **/
CV_EXPORTS void createOCRHMMTransitionsTable (
        std::string& vocabulary, std::vector<std::string>& lexicon,
        OutputArray transition_probabilities_table);

CV_EXPORTS_W Mat createOCRHMMTransitionsTable (
        const String& vocabulary, std::vector<cv::String>& lexicon);

/* OCR BeamSearch Decoder */

/** @brief OCRBeamSearchDecoder class provides an interface for OCR using Beam
 * Search algorithm.

@note
   -   (C++) An example on using OCRBeamSearchDecoder recognition combined with
        scene text detection can be found at the demo sample:
        <https://github.com/Itseez/opencv_contrib/blob/master/modules/text/samples/word_recognition.cpp>
 */


/* Forward declaration of class that can be used to generate an OCRBeamSearchDecoder::ClassifierCallbac */
class TextImageClassifier;

class CV_EXPORTS_W OCRBeamSearchDecoder : public BaseOCR{

 public:

    /** @brief Callback with the character classifier is made a class.

     * This way it hides the feature extractor and the classifier itself, so
     * developers can write their own OCR code.

     * The default character classifier and feature extractor can be loaded
     * using the utility funtion loadOCRBeamSearchClassifierCNN with all its
     * parameters provided in
     * <https://github.com/Itseez/opencv_contrib/blob/master/modules/text/samples/OCRBeamSearch_CNN_model_data.xml.gz>.
     */
    class CV_EXPORTS_W ClassifierCallback{
     public:
        virtual ~ClassifierCallback() { }
        /** @brief The character classifier must return a (ranked list of) class(es) id('s)

        @param image Input image CV_8UC1 or CV_8UC3 with a single letter.
        @param recognition_probabilities For each of the N characters found the classifier returns a list with
        class probabilities for each class.
        @param oversegmentation The classifier returns a list of N+1 character locations' x-coordinates,
        including 0 as start-sequence location.
         */
        virtual void eval( InputArray image, std::vector< std::vector<double> >& recognition_probabilities, std::vector<int>& oversegmentation );

        virtual int getWindowSize() {return 0;}
        virtual int getStepSize() {return 0;}
    };

public:
    /** @brief Recognize text using Beam Search.

    Takes image on input and returns recognized text in the output_text parameter. Optionally
    provides also the Rects for individual text elements found (e.g. words), and the list of those
    text elements with their confidence values.

    @param image Input binary image CV_8UC1 with a single text line (or word).

    @param output_text Output text. Most likely character sequence found by the HMM decoder.

    @param component_rects If provided the method will output a list of Rects for the individual
    text elements found (e.g. words).

    @param component_texts If provided the method will output a list of text strings for the
    recognition of individual text elements found (e.g. words).

    @param component_confidences If provided the method will output a list of confidence values
    for the recognition of individual text elements found (e.g. words).

    @param component_level Only OCR_LEVEL_WORD is supported.
     */
    virtual void run(Mat& image, std::string& output_text, std::vector<Rect>* component_rects=NULL,
                     std::vector<std::string>* component_texts=NULL, std::vector<float>* component_confidences=NULL,
                     int component_level=0);

    virtual void run(Mat& image, Mat& mask, std::string& output_text, std::vector<Rect>* component_rects=NULL,
                     std::vector<std::string>* component_texts=NULL, std::vector<float>* component_confidences=NULL,
                     int component_level=0);

    // aliases for scripting
    CV_WRAP String run(InputArray image, int min_confidence, int component_level=0);

    CV_WRAP String run(InputArray image, InputArray mask, int min_confidence, int component_level=0);

    /** @brief Creates an instance of the OCRBeamSearchDecoder class. Initializes HMMDecoder.

    @param classifier The character classifier with built in feature extractor.

    @param vocabulary The language vocabulary (chars when ASCII English text). vocabulary.size()
    must be equal to the number of classes of the classifier.

    @param transition_probabilities_table Table with transition probabilities between character
    pairs. cols == rows == vocabulary.size().

    @param emission_probabilities_table Table with observation emission probabilities. cols ==
    rows == vocabulary.size().

    @param mode HMM Decoding algorithm. Only OCR_DECODER_VITERBI is available for the moment
    (<http://en.wikipedia.org/wiki/Viterbi_algorithm>).

    @param beam_size Size of the beam in Beam Search algorithm.
     */

    static Ptr<OCRBeamSearchDecoder> create(const Ptr<OCRBeamSearchDecoder::ClassifierCallback> classifier,// The character classifier with built in feature extractor
                                     const std::string& vocabulary,                    // The language vocabulary (chars when ASCII English text)
                                                                                       //     size() must be equal to the number of classes
                                     InputArray transition_probabilities_table,        // Table with transition probabilities between character pairs
                                                                                       //     cols == rows == vocabulary.size()
                                     InputArray emission_probabilities_table,          // Table with observation emission probabilities
                                                                                       //     cols == rows == vocabulary.size()
                                     decoder_mode mode = OCR_DECODER_VITERBI,          // HMM Decoding algorithm (only Viterbi for the moment)
                                     int beam_size = 500);                              // Size of the beam in Beam Search algorithm

    CV_WRAP static Ptr<OCRBeamSearchDecoder> create(const Ptr<OCRBeamSearchDecoder::ClassifierCallback> classifier, // The character classifier with built in feature extractor
                                     const String& vocabulary,                    // The language vocabulary (chars when ASCII English text)
                                                                                       //     size() must be equal to the number of classes
                                     InputArray transition_probabilities_table,        // Table with transition probabilities between character pairs
                                                                                       //     cols == rows == vocabulary.size()
                                     InputArray emission_probabilities_table,          // Table with observation emission probabilities
                                                                                       //     cols == rows == vocabulary.size()
                                     int mode = OCR_DECODER_VITERBI,          // HMM Decoding algorithm (only Viterbi for the moment)
                                     int beam_size = 500);                              // Size of the beam in Beam Search algorithm





    /** @brief Creates an instance of the OCRBeamSearchDecoder class. Initializes HMMDecoder from the specified path.

    @overload

    @param filename path to a character classifier file

    @param vocabulary The language vocabulary (chars when ASCII English text). vocabulary.size()
    must be equal to the number of classes of the classifier..

    @param transition_probabilities_table Table with transition probabilities between character
    pairs. cols == rows == vocabulary.size().

    @param emission_probabilities_table Table with observation emission probabilities. cols ==
    rows == vocabulary.size().

    @param windowWidth The width of the windows to which the sliding window will be iterated. The height will
    be the height of the image. The windows might be resized to fit the classifiers input by the classifiers
    preprocessor.

    @param mode HMM Decoding algorithm (only Viterbi for the moment)

    @param beam_size Size of the beam in Beam Search algorithm

     */
    CV_WRAP static Ptr<OCRBeamSearchDecoder> create(const String& filename, // The character classifier file
                                     const String& vocabulary,                    // The language vocabulary (chars when ASCII English text)
                                                                                       //     size() must be equal to the number of classes
                                     InputArray transition_probabilities_table,        // Table with transition probabilities between character pairs
                                                                                       //     cols == rows == vocabulary.size()
                                     InputArray emission_probabilities_table,          // Table with observation emission probabilities
                                                                                       //     cols == rows == vocabulary.size()
                                     int mode = OCR_DECODER_VITERBI,          // HMM Decoding algorithm (only Viterbi for the moment)
                                     int beam_size = 500);

protected:

    Ptr<OCRBeamSearchDecoder::ClassifierCallback> classifier;
    std::string vocabulary;
    Mat transition_p;
    Mat emission_p;
    decoder_mode mode;
    int beam_size;
};

/** @brief Allow to implicitly load the default character classifier when creating an OCRBeamSearchDecoder object.

@param filename The XML or YAML file with the classifier model (e.g. OCRBeamSearch_CNN_model_data.xml.gz)

The CNN default classifier is based in the scene text recognition method proposed by Adam Coates &
Andrew NG in [Coates11a]. The character classifier consists in a Single Layer Convolutional Neural Network and
a linear classifier. It is applied to the input image in a sliding window fashion, providing a set of recognitions
at each window location.
 */

CV_EXPORTS_W Ptr<OCRBeamSearchDecoder::ClassifierCallback> loadOCRBeamSearchClassifierCNN(const String& filename);

//! @}


//Classifiers should provide diferent backends
//For the moment only caffe is implemeted
enum{
    OCR_HOLISTIC_BACKEND_NONE,
    OCR_HOLISTIC_BACKEND_CAFFE
};

class TextImageClassifier;

/**
 * @brief The ImagePreprocessor class
 */
class CV_EXPORTS_W ImagePreprocessor{
protected:
    virtual void preprocess_(const Mat& input,Mat& output,Size outputSize,int outputChannels)=0;
    virtual void set_mean_(Mat){}

public:
    virtual ~ImagePreprocessor(){}

    /** @brief this method in provides public acces to the preprocessing with respect to a specific
     * classifier
     *
     * This method's main use would be to use the preprocessor without feeding it to a classifier.
     * Determining the exact behavior of a preprocessor is the main motivation for this.
     *
     * @param input an image without any constraints
     *
     * @param output in most cases an image of fixed depth size and whitened
     *
     * @param sz the size to which the image would be resize if the preprocessor resizes inputs
     *
     * @param outputChannels the number of channels for the output image
     */
    CV_WRAP void preprocess(InputArray input,OutputArray output,Size sz,int outputChannels);

    /** @brief this method in provides public acces to set the mean of the input images
     * mean can be a mat either of same size of the image or one value per color channel
     * A preprocessor can be created without the mean( the pre processor will calculate mean for every image
     * in that case
     *

     * @param mean which will be subtracted from the images
     *
     */

    CV_WRAP void set_mean(Mat mean);

    /** @brief Creates a functor that only resizes and changes the channels of the input
     *  without further processing.
     *
     * @return shared pointer to the generated preprocessor
     */
    CV_WRAP static Ptr<ImagePreprocessor> createResizer();

    /** @brief
     *
     * @param sigma
     *
     * @return shared pointer to generated preprocessor
     */
    CV_WRAP static Ptr<ImagePreprocessor> createImageStandarizer(double sigma);

    /** @brief
     *
     * @return shared pointer to generated preprocessor
     */
    CV_WRAP static Ptr<ImagePreprocessor> createImageMeanSubtractor(InputArray meanImg);
    /** @brief
     * create a functor with the parameters, parameters can be changes by corresponding set functions
     * @return shared pointer to generated preprocessor
     */

    CV_WRAP static Ptr<ImagePreprocessor>createImageCustomPreprocessor(double rawval=1.0,String channel_order="BGR");

    friend class TextImageClassifier;

};

/** @brief Abstract class that implements the classifcation of text images.
 *
 * The interface is generic enough to describe any image classifier. And allows
 * to take advantage of compouting in batches. While word classifiers are the default
 * networks, any image classifers should work.
 *
 */
class CV_EXPORTS_W TextImageClassifier
{
protected:
    Size inputGeometry_;
    Size outputGeometry_;
    int channelCount_;
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
    virtual ~TextImageClassifier() {}

    /** @brief
     */
    CV_WRAP virtual void setPreprocessor(Ptr<ImagePreprocessor> ptr);

    /** @brief
     */
    CV_WRAP Ptr<ImagePreprocessor> getPreprocessor();

    /** @brief produces a class confidence row-vector given an image
     */
    CV_WRAP virtual void classify(InputArray image, OutputArray classProbabilities) = 0;

    /** @brief produces a matrix containing class confidence row-vectors given an collection of images
     */
    CV_WRAP virtual void classifyBatch(InputArrayOfArrays image, OutputArray classProbabilities) = 0;

    /** @brief simple getter method returning the number of channels each input sample has
     */
    CV_WRAP virtual int getInputChannelCount(){return this->channelCount_;}

    /** @brief simple getter method returning the size of the input sample
     */
    CV_WRAP virtual Size getInputSize(){return this->inputGeometry_;}

    /** @brief simple getter method returning the size of the oputput row-vector
     */
    CV_WRAP virtual int getOutputSize()=0;
    /** @brief simple getter method returning the shape of the oputput from caffe
     */
    CV_WRAP virtual Size getOutputGeometry()=0;

    /** @brief simple getter method returning the size of the minibatches for this classifier.
     * If not applicabe this method should return 1
     */
    CV_WRAP virtual int getMinibatchSize()=0;

    friend class ImagePreprocessor;
};



class CV_EXPORTS_W DeepCNN:public TextImageClassifier
{
    /** @brief Class that uses a pretrained caffe model for word classification.
     *
     * This network is described in detail in:
     * Max Jaderberg et al.: Reading Text in the Wild with Convolutional Neural Networks, IJCV 2015
     * http://arxiv.org/abs/1412.1842
     */
public:
    virtual ~DeepCNN() {};

    /** @brief Constructs a DeepCNN object from a caffe pretrained model
     *
     * @param archFilename is the path to the prototxt file containing the deployment model architecture description.
     *
     * @param weightsFilename is the path to the pretrained weights of the model in binary fdorm. This file can be
     * very large, up to 2GB.
     *
     * @param preprocessor is a pointer to the instance of a ImagePreprocessor implementing the preprocess_ protecteed method;
     *
     * @param minibatchSz the maximum number of samples that can processed in parallel. In practice this parameter
     * has an effect only when computing in the GPU and should be set with respect to the memory available in the GPU.
     *
     * @param backEnd integer parameter selecting the coputation framework. For now OCR_HOLISTIC_BACKEND_CAFFE is
     * the only option
     */
    CV_WRAP static Ptr<DeepCNN> create(String archFilename,String weightsFilename,Ptr<ImagePreprocessor> preprocessor,int minibatchSz=100,int backEnd=OCR_HOLISTIC_BACKEND_CAFFE);

    /** @brief Constructs a DeepCNN intended to be used for word spotting.
     *
     * This method loads a pretrained classifier and couples him with a preprocessor that standarises pixels with a
     * deviation of 113. The architecture file can be downloaded from:
     * <http://nicolaou.homouniversalis.org/assets/vgg_text/dictnet_vgg_deploy.prototxt>
     * While the weights can be downloaded from:
     * <http://nicolaou.homouniversalis.org/assets/vgg_text/dictnet_vgg.caffemodel>
     * The words assigned to the network outputs are available at:
     * <http://nicolaou.homouniversalis.org/assets/vgg_text/dictnet_vgg_labels.txt>
     *
     * @param archFilename is the path to the prototxt file containing the deployment model architecture description.
     * When employing OCR_HOLISTIC_BACKEND_CAFFE this is the path to the deploy ".prototxt".
     *
     * @param weightsFilename is the path to the pretrained weights of the model. When employing
     * OCR_HOLISTIC_BACKEND_CAFFE this is the path to the ".caffemodel" file. This file can be very large, the
     * pretrained DictNet uses 2GB.
     *
     * @param backEnd integer parameter selecting the coputation framework. For now OCR_HOLISTIC_BACKEND_CAFFE is
     * the only option
     */
    CV_WRAP static Ptr<DeepCNN> createDictNet(String archFilename,String weightsFilename,int backEnd=OCR_HOLISTIC_BACKEND_CAFFE);

};

namespace cnn_config{
namespace caffe_backend{

/** @brief Prompts Caffe on the computation device beeing used
 *
 * Caffe can only be controlled globally on whether the GPU or the CPU is used has a
 * global behavior. This function queries the current state of caffe.
 * If the module is built without caffe, this method throws an exception.
 *
 * @return true if caffe is computing on the GPU, false if caffe is computing on the CPU
 */
CV_EXPORTS_W bool getCaffeGpuMode();

/** @brief Sets the computation device beeing used by Caffe
 *
 * Caffe can only be controlled globally on whether the GPU or the CPU is used has a
 * global behavior. This function queries the current state of caffe.
 * If the module is built without caffe, this method throws an exception.
 *
 * @param useGpu  set to true for caffe to be computing on the GPU, false if caffe is
 * computing on the CPU
 */
CV_EXPORTS_W void setCaffeGpuMode(bool useGpu);

/** @brief Provides runtime information on whether Caffe support was compiled in.
 *
 * The text module API is the same regardless of whether CAffe was available or not
 * During compilation. When methods that require Caffe are invocked while Caffe support
 * is not compiled in, exceptions are thrown. This method allows to test whether the
 * text module was built with caffe during runtime.
 *
 * @return true if Caffe support for the the text module was provided during compilation,
 * false if Caffe was unavailable.
 */
CV_EXPORTS_W bool getCaffeAvailable();

}//caffe
}//cnn_config

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
class CV_EXPORTS_W OCRHolisticWordRecognizer : public BaseOCR
{
public:
    virtual void run(Mat& image, std::string& output_text, std::vector<Rect>* component_rects=NULL,
                     std::vector<std::string>* component_texts=NULL, std::vector<float>* component_confidences=NULL,
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

    virtual void run(Mat& image, Mat& mask, std::string& output_text, std::vector<Rect>* component_rects=NULL,
                     std::vector<std::string>* component_texts=NULL, std::vector<float>* component_confidences=NULL,
                     int component_level=OCR_LEVEL_WORD)=0;


    /**
    @brief Method that provides a quick and simple interface to a single word image classifcation

    @param inputImage an image expected to be a CV_U8C1 or CV_U8C3 of any size assumed to contain a single word

    @param transcription an opencv string that will store the detected word transcription

    @param confidence a double that will be updated with the confidence the classifier has for the selected word
    */
    CV_WRAP virtual void recogniseImage(InputArray inputImage,CV_OUT String& transcription,CV_OUT double& confidence)=0;

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
    CV_WRAP virtual void recogniseImageBatch(InputArrayOfArrays inputImageList,CV_OUT std::vector<String>& transcriptions,CV_OUT std::vector<double>& confidences)=0;


    /**
    @brief simple getter for the vocabulary employed
    */
    CV_WRAP virtual const std::vector<String>& getVocabulary()=0;

    /** @brief simple getter for the preprocessing functor
     */
    CV_WRAP virtual Ptr<TextImageClassifier> getClassifier()=0;

    /** @brief Creates an instance of the OCRHolisticWordRecognizer class.

    @param classifierPtr an instance of TextImageClassifier, normaly a DeepCNN instance

    @param vocabularyFilename the relative or absolute path to the file containing all words in the vocabulary. Each text line
    in the file is assumed to be a single word. The number of words in the vocabulary must be exactly the same as the outputSize
    of the classifier.
     */
    CV_WRAP static Ptr<OCRHolisticWordRecognizer> create(Ptr<TextImageClassifier> classifierPtr,String vocabularyFilename);


    /** @brief Creates an instance of the OCRHolisticWordRecognizer class and implicitly also a DeepCNN classifier.

    @param modelArchFilename the relative or absolute path to the prototxt file describing the classifiers architecture.

    @param modelWeightsFilename the relative or absolute path to the file containing the pretrained weights of the model in caffe-binary form.

    @param vocabularyFilename the relative or absolute path to the file containing all words in the vocabulary. Each text line
    in the file is assumed to be a single word. The number of words in the vocabulary must be exactly the same as the outputSize
    of the classifier.
    */
    CV_WRAP static Ptr<OCRHolisticWordRecognizer> create(String modelArchFilename, String modelWeightsFilename, String vocabularyFilename);

    /** @brief
     *
     * @param classifierPtr
     *
     * @param vocabulary
     */
    CV_WRAP static Ptr<OCRHolisticWordRecognizer> create(Ptr<TextImageClassifier> classifierPtr,const std::vector<String>& vocabulary);

    /** @brief
     *
     * @param modelArchFilename
     *
     * @param modelWeightsFilename
     *
     * @param vocabulary
     */
    CV_WRAP static Ptr<OCRHolisticWordRecognizer> create (String modelArchFilename, String modelWeightsFilename, const std::vector<String>& vocabulary);
};


}//namespace text
}//namespace cv


#endif // _OPENCV_TEXT_OCR_HPP_
