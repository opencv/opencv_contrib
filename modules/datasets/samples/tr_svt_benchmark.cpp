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
// Copyright (C) 2014, Itseez Inc, all rights reserved.
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
// In no event shall the Itseez Inc or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#include <iostream>
#include <opencv2/opencv_modules.hpp>

#ifdef HAVE_OPENCV_TEXT

#include "opencv2/datasets/tr_svt.hpp"
#include <opencv2/core.hpp>
#include "opencv2/text.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"

#include <cstdio>
#include <cstdlib> // atoi

#include <string>
#include <vector>

using namespace std;
using namespace cv;
using namespace cv::datasets;
using namespace cv::text;

//Calculate edit distance between two words
size_t edit_distance(const string& A, const string& B);
size_t min(size_t x, size_t y, size_t z);
bool   isRepetitive(const string& s);
bool   sort_by_length(const string &a, const string &b);
//Draw ER's in an image via floodFill
void   er_draw(vector<Mat> &channels, vector<vector<ERStat> > &regions, vector<Vec2i> group, Mat& segmentation);

size_t min(size_t x, size_t y, size_t z)
{
    return x < y ? min(x,z) : min(y,z);
}

size_t edit_distance(const string& A, const string& B)
{
    size_t NA = A.size();
    size_t NB = B.size();

    vector< vector<size_t> > M(NA + 1, vector<size_t>(NB + 1));

    for (size_t a = 0; a <= NA; ++a)
        M[a][0] = a;

    for (size_t b = 0; b <= NB; ++b)
        M[0][b] = b;

    for (size_t a = 1; a <= NA; ++a)
        for (size_t b = 1; b <= NB; ++b)
        {
            size_t x = M[a-1][b] + 1;
            size_t y = M[a][b-1] + 1;
            size_t z = M[a-1][b-1] + (A[a-1] == B[b-1] ? 0 : 1);
            M[a][b] = min(x,y,z);
        }

    return M[A.size()][B.size()];
}

bool sort_by_length(const string &a, const string &b){return (a.size()>b.size());}

bool isRepetitive(const string& s)
{
    int count = 0;
    for (int i=0; i<(int)s.size(); i++)
    {
        if ((s[i] == 'i') ||
                (s[i] == 'l') ||
                (s[i] == 'I'))
            count++;
    }
    if (count > ((int)s.size()+1)/2)
    {
        return true;
    }
    return false;
}


void er_draw(vector<Mat> &channels, vector<vector<ERStat> > &regions, vector<Vec2i> group, Mat& segmentation)
{
    for (int r=0; r<(int)group.size(); r++)
    {
        ERStat er = regions[group[r][0]][group[r][1]];
        if (er.parent != NULL) // deprecate the root region
        {
            int newMaskVal = 255;
            int flags = 4 + (newMaskVal << 8) + FLOODFILL_FIXED_RANGE + FLOODFILL_MASK_ONLY;
            floodFill(channels[group[r][0]],segmentation,Point(er.pixel%channels[group[r][0]].cols,er.pixel/channels[group[r][0]].cols),
                      Scalar(255),0,Scalar(er.level),Scalar(0),flags);
        }
    }
}

// std::toupper is int->int
static char char_toupper(char ch)
{
    return (char)std::toupper((int)ch);
}

int main(int argc, char *argv[])
{
    const char *keys =
            "{ help h usage ? |    | show this message }"
            "{ path p         |true| path to dataset xml files }";
    CommandLineParser parser(argc, argv, keys);
    string path(parser.get<string>("path"));
    if (parser.has("help") || path=="true")
    {
        parser.printMessage();
        return -1;
    }

    // loading train & test images description
    Ptr<TR_svt> dataset = TR_svt::create();
    dataset->load(path);


    vector<double> f1Each;

    unsigned int correctNum = 0;
    unsigned int returnedNum = 0;
    unsigned int returnedCorrectNum = 0;

    vector< Ptr<Object> >& test = dataset->getTest();
    unsigned int num = 0;
    for (vector< Ptr<Object> >::iterator itT=test.begin(); itT!=test.end(); ++itT)
    {
        TR_svtObj *example = static_cast<TR_svtObj *>((*itT).get());

        num++;
        printf("processed image: %u, name: %s\n", num, example->fileName.c_str());

        correctNum += example->tags.size();
/*    printf("\ntags:\n");
    for (vector<tag>::iterator it=example->tags.begin(); it!=example->tags.end(); ++it)
    {
        tag &t = (*it);
        printf("%s\nx: %u, y: %u, width: %u, height: %u\n",
               t.value.c_str(), t.x, t.y, t.x+t.width, t.y+t.height);
    }*/
        unsigned int correctNumEach = example->tags.size();
        unsigned int returnedNumEach = 0;
        unsigned int returnedCorrectNumEach = 0;

        Mat image = imread((path+example->fileName).c_str());
        /*Text Detection*/

        // Extract channels to be processed individually
        vector<Mat> channels;

        Mat grey;
        cvtColor(image,grey,COLOR_RGB2GRAY);

        // Notice here we are only using grey channel, see textdetection.cpp for example with more channels
        channels.push_back(grey);
        channels.push_back(255-grey);

        // Create ERFilter objects with the 1st and 2nd stage default classifiers
        Ptr<ERFilter> er_filter1 = createERFilterNM1(loadClassifierNM1("trained_classifierNM1.xml"),8,0.00015f,0.13f,0.2f,true,0.1f);
        Ptr<ERFilter> er_filter2 = createERFilterNM2(loadClassifierNM2("trained_classifierNM2.xml"),0.5);

        vector<vector<ERStat> > regions(channels.size());
        // Apply the default cascade classifier to each independent channel (could be done in parallel)
        for (int c=0; c<(int)channels.size(); c++)
        {
            er_filter1->run(channels[c], regions[c]);
            er_filter2->run(channels[c], regions[c]);
        }

        // Detect character groups
        vector< vector<Vec2i> > nm_region_groups;
        vector<Rect> nm_boxes;
        erGrouping(image, channels, regions, nm_region_groups, nm_boxes, ERGROUPING_ORIENTATION_HORIZ);


        /*Text Recognition (OCR)*/

        Ptr<OCRTesseract> ocr = OCRTesseract::create();
        for (int i=0; i<(int)nm_boxes.size(); i++)
        {
            Mat group_img = Mat::zeros(image.rows+2, image.cols+2, CV_8UC1);
            er_draw(channels, regions, nm_region_groups[i], group_img);
            group_img(nm_boxes[i]).copyTo(group_img);
            copyMakeBorder(group_img,group_img,15,15,15,15,BORDER_CONSTANT,Scalar(0));

            string output;
            vector<Rect>   boxes;
            vector<string> words;
            vector<float>  confidences;
            ocr->run(group_img, output, &boxes, &words, &confidences, OCR_LEVEL_WORD);

            output.erase(remove(output.begin(), output.end(), '\n'), output.end());
            //cout << "OCR output = \"" << output << "\" length = " << output.size() << endl;
            if (output.size() < 3)
                continue;

            for (int j=0; j<(int)boxes.size(); j++)
            {
                boxes[j].x += nm_boxes[i].x-15;
                boxes[j].y += nm_boxes[i].y-15;

                //cout << "  word = " << words[j] << "\t confidence = " << confidences[j] << endl;
                if ((words[j].size() < 2) || (confidences[j] < 51) ||
                    ((words[j].size()==2) && (words[j][0] == words[j][1])) ||
                    ((words[j].size()< 4) && (confidences[j] < 60)) ||
                     isRepetitive(words[j]))
                {
                    continue;
                }

                std::transform(words[j].begin(), words[j].end(), words[j].begin(), char_toupper);

                if (find(example->lex.begin(), example->lex.end(), words[j]) == example->lex.end())
                {
                    continue;
                }

                returnedNum++;
                returnedNumEach++;
                /*printf("%s\nx: %u, y: %u, width: %u, height: %u\n",
                        words[j].c_str(), boxes[j].tl().x, boxes[j].tl().y, boxes[j].br().x, boxes[j].br().y);*/
                for (vector<tag>::iterator it=example->tags.begin(); it!=example->tags.end(); ++it)
                {
                    tag &t = (*it);
                    if (t.value==words[j] &&
                        !(boxes[j].tl().x > t.x+t.width || boxes[j].br().x < t.x ||
                          boxes[j].tl().y > t.y+t.height || boxes[j].br().y < t.y))
                    {
                        returnedCorrectNum++;
                        returnedCorrectNumEach++;
                        break;
                    }
                }
            }
        }
        double p = 0.0;
        if (0 != returnedNumEach)
        {
            p = 1.0*returnedCorrectNumEach/returnedNumEach;
        }
        double r = 0.0;
        if (0 != correctNumEach)
        {
            r = 1.0*returnedCorrectNumEach/correctNumEach;
        }
        double f1 = 0.0;
        if (0 != p+r)
        {
            f1 = 2*(p*r)/(p+r);
        }
        //printf("|%f|\n", f1);
        f1Each.push_back(f1);
    }

    double p = 1.0*returnedCorrectNum/returnedNum;
    double r = 1.0*returnedCorrectNum/correctNum;
    double f1 = 2*(p*r)/(p+r);
    printf("f1: %f\n", f1);

    /*double f1 = 0.0;
    for (vector<double>::iterator it=f1Each.begin(); it!=f1Each.end(); ++it)
    {
        f1 += *it;
    }
    f1 /= f1Each.size();
    printf("mean f1: %f\n", f1);*/

    return 0;
}

#else

int main()
{
    std::cerr << "OpenCV was built without text module" << std::endl;
    return 0;
}

#endif // HAVE_OPENCV_TEXT
