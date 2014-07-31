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

#include "text_config.hpp"

#ifdef HAVE_TESSERACT
#include <tesseract/baseapi.h>
#include <tesseract/resultiterator.h>
#endif

#include "opencv2/core.hpp"
#include <vector>
#include <string>


namespace cv
{
namespace text
{

using namespace std;

enum
{
    OCR_LEVEL_WORD,
    OCR_LEVEL_TEXTLINE
};

#ifdef HAVE_TESSERACT
class CV_EXPORTS OCRTesseract
{
private:
    tesseract::TessBaseAPI tess;

public:
    //Default constructor
    OCRTesseract(const char* datapath=NULL, const char* language=NULL, const char* char_whitelist=NULL,
                 tesseract::OcrEngineMode oem=tesseract::OEM_DEFAULT, tesseract::PageSegMode psmode=tesseract::PSM_AUTO);

    ~OCRTesseract();

    void run(Mat& image, string& output_text, vector<Rect>* component_rects=NULL,
             vector<string>* component_texts=NULL, vector<float>* component_confidences=NULL,
             int component_level=0);
};
#else
//stub
class CV_EXPORTS OCRTesseract
{
public:
    //Default constructor
    OCRTesseract(const char* datapath=NULL, const char* language=NULL, const char* char_whitelist=NULL,
                 int oem=0, int psmode=0);

    ~OCRTesseract();

    void run(Mat& image, string& output_text, vector<Rect>* component_rects=NULL,
             vector<string>* component_texts=NULL, vector<float>* component_confidences=NULL,
             int component_level=0);
};
#endif



}
}
#endif // _OPENCV_TEXT_OCR_HPP_
