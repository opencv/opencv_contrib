/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
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

#include "precomp.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/ml.hpp"

#include <iostream>
#include <fstream>
#include <queue>

namespace cv
{
namespace text
{

using namespace std;

void OCRTesseract::run(Mat& image, string& output_text, vector<Rect>* component_rects,
                       vector<string>* component_texts, vector<float>* component_confidences,
                       int component_level)
{
    CV_Assert( (image.type() == CV_8UC1) || (image.type() == CV_8UC3) );
    CV_Assert( (component_level == OCR_LEVEL_TEXTLINE) || (component_level == OCR_LEVEL_WORD) );
    output_text.clear();
    if (component_rects != NULL)
        component_rects->clear();
    if (component_texts != NULL)
        component_texts->clear();
    if (component_confidences != NULL)
        component_confidences->clear();
}

void OCRTesseract::run(Mat& image, Mat& mask, string& output_text, vector<Rect>* component_rects,
                       vector<string>* component_texts, vector<float>* component_confidences,
                       int component_level)
{
    CV_Assert( (image.type() == CV_8UC1) || (image.type() == CV_8UC3) );
    CV_Assert( mask.type() == CV_8UC1 );
    CV_Assert( (component_level == OCR_LEVEL_TEXTLINE) || (component_level == OCR_LEVEL_WORD) );
    output_text.clear();
    if (component_rects != NULL)
        component_rects->clear();
    if (component_texts != NULL)
        component_texts->clear();
    if (component_confidences != NULL)
        component_confidences->clear();
}

CV_WRAP String OCRTesseract::run(InputArray image, int min_confidence, int component_level)
{
    std::string output1;
    std::string output2;
    vector<string> component_texts;
    vector<float> component_confidences;
    Mat image_m = image.getMat();
    run(image_m, output1, NULL, &component_texts, &component_confidences, component_level);
    for(unsigned int i = 0; i < component_texts.size(); i++)
    {
        // cout << "confidence: " << component_confidences[i] << " text:" << component_texts[i] << endl;
        if(component_confidences[i] > min_confidence)
        {
            output2 += component_texts[i];
        }
    }
    return String(output2);
}

CV_WRAP String OCRTesseract::run(InputArray image, InputArray mask, int min_confidence, int component_level)
{
    std::string output1;
    std::string output2;
    vector<string> component_texts;
    vector<float> component_confidences;
    Mat image_m = image.getMat();
    Mat mask_m = mask.getMat();
    run(image_m, mask_m, output1, NULL, &component_texts, &component_confidences, component_level);
    for(unsigned int i = 0; i < component_texts.size(); i++)
    {
        cout << "confidence: " << component_confidences[i] << " text:" << component_texts[i] << endl;

        if(component_confidences[i] > min_confidence)
        {
            output2 += component_texts[i];
        }
    }
    return String(output2);
}


class OCRTesseractImpl : public OCRTesseract
{
private:
#ifdef HAVE_TESSERACT
    tesseract::TessBaseAPI tess;
#endif

public:
    //Default constructor
    OCRTesseractImpl(const char* datapath, const char* language, const char* char_whitelist, int oemode, int psmode)
    {

#ifdef HAVE_TESSERACT
        const char *lang = "eng";
        if (language != NULL)
            lang = language;

        if (tess.Init(datapath, lang, (tesseract::OcrEngineMode)oemode))
        {
            cout << "OCRTesseract: Could not initialize tesseract." << endl;
            throw 1;
        }

        //cout << "OCRTesseract: tesseract version " << tess.Version() << endl;

        tesseract::PageSegMode pagesegmode = (tesseract::PageSegMode)psmode;
        tess.SetPageSegMode(pagesegmode);

        if(char_whitelist != NULL)
            tess.SetVariable("tessedit_char_whitelist", char_whitelist);
        else
            tess.SetVariable("tessedit_char_whitelist", "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ");

        tess.SetVariable("save_best_choices", "T");
#else
        cout << "OCRTesseract("<<oemode<<psmode<<"): Tesseract not found." << endl;
        if (datapath != NULL)
            cout << "            " << datapath << endl;
        if (language != NULL)
            cout << "            " << language << endl;
        if (char_whitelist != NULL)
            cout << "            " << char_whitelist << endl;
#endif
    }

    ~OCRTesseractImpl()
    {
#ifdef HAVE_TESSERACT
        tess.End();
#endif
    }

    void run(Mat& image, string& output, vector<Rect>* component_rects=NULL,
             vector<string>* component_texts=NULL, vector<float>* component_confidences=NULL,
             int component_level=0)
    {

        CV_Assert( (image.type() == CV_8UC1) || (image.type() == CV_8UC3) );

#ifdef HAVE_TESSERACT

        if (component_texts != 0)
            component_texts->clear();
        if (component_rects != 0)
            component_rects->clear();
        if (component_confidences != 0)
            component_confidences->clear();

        tess.SetImage((uchar*)image.data, image.size().width, image.size().height, image.channels(), image.step1());
        tess.Recognize(0);
        char *outText;
        outText = tess.GetUTF8Text();
        output = string(outText);
        delete [] outText;

        if ( (component_rects != NULL) || (component_texts != NULL) || (component_confidences != NULL) )
        {
            tesseract::ResultIterator* ri = tess.GetIterator();
            tesseract::PageIteratorLevel level = tesseract::RIL_WORD;
            if (component_level == OCR_LEVEL_TEXTLINE)
                level = tesseract::RIL_TEXTLINE;

            if (ri != 0) {
                do {
                    const char* word = ri->GetUTF8Text(level);
                    if (word == NULL)
                        continue;
                    float conf = ri->Confidence(level);
                    int x1, y1, x2, y2;
                    ri->BoundingBox(level, &x1, &y1, &x2, &y2);

                    if (component_texts != 0)
                        component_texts->push_back(string(word));
                    if (component_rects != 0)
                        component_rects->push_back(Rect(x1,y1,x2-x1,y2-y1));
                    if (component_confidences != 0)
                        component_confidences->push_back(conf);

                    delete[] word;
                } while (ri->Next(level));
            }
            delete ri;
        }

        tess.Clear();

#else

        cout << "OCRTesseract(" << component_level << image.type() <<"): Tesseract not found." << endl;
        output.clear();
        if(component_rects)
            component_rects->clear();
        if(component_texts)
            component_texts->clear();
        if(component_confidences)
            component_confidences->clear();
#endif
    }

    void run(Mat& image, Mat& mask, string& output, vector<Rect>* component_rects=NULL,
             vector<string>* component_texts=NULL, vector<float>* component_confidences=NULL,
             int component_level=0)
    {
        CV_Assert( mask.type() == CV_8UC1 );
        CV_Assert( (image.type() == CV_8UC1) || (image.type() == CV_8UC3) );

        run( mask, output, component_rects, component_texts, component_confidences, component_level);
    }

    void setWhiteList(const String& char_whitelist)
    {
  #ifdef HAVE_TESSERACT
        tess.SetVariable("tessedit_char_whitelist", char_whitelist.c_str());
  #else
        (void)char_whitelist;
  #endif
    }
};

Ptr<OCRTesseract> OCRTesseract::create(const char* datapath, const char* language,
                                       const char* char_whitelist, int oem, int psmode)
{
    return makePtr<OCRTesseractImpl>(datapath, language, char_whitelist, oem, psmode);
}


}
}
