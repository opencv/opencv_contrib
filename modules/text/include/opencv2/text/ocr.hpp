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
using namespace std;

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
    virtual void run(Mat& image, string& output_text, vector<Rect>* component_rects=NULL,
            vector<string>* component_texts=NULL, vector<float>* component_confidences=NULL,
            int component_level=0) = 0;
};

class CV_EXPORTS OCRTesseract : public BaseOCR
{
public:
    virtual void run(Mat& image, string& output_text, vector<Rect>* component_rects=NULL,
            vector<string>* component_texts=NULL, vector<float>* component_confidences=NULL,
            int component_level=0)
    {
      CV_Assert( (image.type() == CV_8UC1) || (image.type() == CV_8UC1) );
      CV_Assert( (component_level == OCR_LEVEL_TEXTLINE) || (component_level == OCR_LEVEL_WORD) );
      output_text.clear();
      if (component_rects != NULL)
        component_rects->clear();
      if (component_texts != NULL)
        component_texts->clear();
      if (component_confidences != NULL)
        component_confidences->clear();
    }
    static Ptr<OCRTesseract> create(const char* datapath=NULL, const char* language=NULL, const char* char_whitelist=NULL, int oem=3, int psmode=3);
};


}
}
#endif // _OPENCV_TEXT_OCR_HPP_
