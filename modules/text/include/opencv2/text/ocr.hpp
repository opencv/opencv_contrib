/*M///////////////////////////////////////////////////////////////////////////////////////
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

namespace cv
{
namespace text
{

enum
{
    OCR_LEVEL_WORD,
    OCR_LEVEL_TEXTLINE
};

//base class BaseOCR declares a common API that would be used in a typical text recognition scenario
class CV_EXPORTS BaseOCR
{
public:
    virtual ~BaseOCR() {};
    virtual void run(Mat& image, std::string& output_text, std::vector<Rect>* component_rects=NULL,
                     std::vector<std::string>* component_texts=NULL, std::vector<float>* component_confidences=NULL,
                     int component_level=0) = 0;
};

/* OCR Tesseract */

class CV_EXPORTS OCRTesseract : public BaseOCR
{
public:
    virtual void run(Mat& image, std::string& output_text, std::vector<Rect>* component_rects=NULL,
                     std::vector<std::string>* component_texts=NULL, std::vector<float>* component_confidences=NULL,
                     int component_level=0);

    static Ptr<OCRTesseract> create(const char* datapath=NULL, const char* language=NULL,
                                    const char* char_whitelist=NULL, int oem=3, int psmode=3);
};


/* OCR HMM Decoder */

enum decoder_mode
{
    OCR_DECODER_VITERBI = 0 // Other algorithms may be added
};

class CV_EXPORTS OCRHMMDecoder : public BaseOCR
{
public:

    //! callback with the character classifier is made a class. This way we hide the feature extractor and the classifier itself
    class CV_EXPORTS ClassifierCallback
    {
    public:
        virtual ~ClassifierCallback() { }
        //! The classifier must return a (ranked list of) class(es) id('s)
        virtual void eval( InputArray image, std::vector<int>& out_class, std::vector<double>& out_confidence);
    };

public:
    //! Decode a group of regions and output the most likely sequence of characters
    virtual void run(Mat& image, std::string& output_text, std::vector<Rect>* component_rects=NULL,
                     std::vector<std::string>* component_texts=NULL, std::vector<float>* component_confidences=NULL,
                     int component_level=0);

    static Ptr<OCRHMMDecoder> create(const Ptr<OCRHMMDecoder::ClassifierCallback> classifier,// The character classifier with built in feature extractor
                                     const std::string& vocabulary,                    // The language vocabulary (chars when ascii english text)
                                                                                       //     size() must be equal to the number of classes
                                     InputArray transition_probabilities_table,        // Table with transition probabilities between character pairs
                                                                                       //     cols == rows == vocabulari.size()
                                     InputArray emission_probabilities_table,          // Table with observation emission probabilities
                                                                                       //     cols == rows == vocabulari.size()
                                     decoder_mode mode = OCR_DECODER_VITERBI);         // HMM Decoding algorithm (only Viterbi for the moment)

protected:

    Ptr<OCRHMMDecoder::ClassifierCallback> classifier;
    std::string vocabulary;
    Mat transition_p;
    Mat emission_p;
    decoder_mode mode;
};

CV_EXPORTS Ptr<OCRHMMDecoder::ClassifierCallback> loadOCRHMMClassifierNM(const std::string& filename);

}
}
#endif // _OPENCV_TEXT_OCR_HPP_
