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

#include "opencv2/datasets/tr_icdar.hpp"
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
bool   sort_by_lenght(const string &a, const string &b);
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

bool sort_by_lenght(const string &a, const string &b){return (a.size()>b.size());}

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
            "{ path p         |true| path to dataset root folder }"
            "{ ws wordspotting|    | evaluate \"word spotting\" results }"
            "{ lex lexicon    |1   | 0:no-lexicon, 1:100-words, 2:full-lexicon }";

    CommandLineParser parser(argc, argv, keys);
    string path(parser.get<string>("path"));
    if (parser.has("help") || path=="true")
    {
        parser.printMessage();
        return -1;
    }

    bool is_word_spotting = parser.has("ws");
    int selected_lex = parser.get<int>("lex");
    if ((selected_lex < 0) || (selected_lex > 2))
    {
        parser.printMessage();
        printf("Unsupported lex value.\n");
        return -1;
    }

    // loading train & test images description
    Ptr<TR_icdar> dataset = TR_icdar::create();
    dataset->load(path);


    vector<double> f1Each;

    unsigned int correctNum = 0;
    unsigned int returnedNum = 0;
    unsigned int returnedCorrectNum = 0;

    vector< Ptr<Object> >& test = dataset->getTest();
    unsigned int num = 0;
    for (vector< Ptr<Object> >::iterator itT=test.begin(); itT!=test.end(); ++itT)
    {
        TR_icdarObj *example = static_cast<TR_icdarObj *>((*itT).get());

        num++;
        printf("processed image: %u, name: %s\n", num, example->fileName.c_str());

        vector<string> empty_lexicon;
        vector<string> *lex;
        switch (selected_lex)
        {
            case 0:
                lex = &empty_lexicon;
                break;
            case 2:
                lex = &example->lexFull;
                break;
            default:
                lex = &example->lex100;
                break;
        }

        correctNum += example->words.size();
        unsigned int correctNumEach = example->words.size();

        // Take care of dontcare regions t.value == "###"
        for (size_t w=0; w<example->words.size(); w++)
        {
            string w_upper = example->words[w].value;
            transform(w_upper.begin(), w_upper.end(), w_upper.begin(), char_toupper);
            if ((find (lex->begin(), lex->end(), w_upper) == lex->end()) &&
                (is_word_spotting) && (selected_lex != 0))
                example->words[w].value = "###";
            if ( (example->words[w].value == "###") || (example->words[w].value.size()<3) )
            {
                correctNum --;
                correctNumEach --;
            }
        }

        unsigned int returnedNumEach = 0;
        unsigned int returnedCorrectNumEach = 0;

        Mat image = imread((path+"/test/"+example->fileName).c_str());

        /*Text Detection*/

        // Extract channels to be processed individually
        vector<Mat> channels;

        Mat grey;
        cvtColor(image,grey,COLOR_RGB2GRAY);

        // Notice here we are only using grey channel, see textdetection.cpp for example with more channels
        channels.push_back(grey);
        channels.push_back(255-grey);


        // Create ERFilter objects with the 1st and 2nd sworde default classifiers
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
        bool ocr_is_tesseract = true;

        vector<string> final_words;
        vector<Rect>   final_boxes;
        vector<float>  final_confs;
        for (int i=0; i<(int)nm_boxes.size(); i++)
        {
            Mat group_img = Mat::zeros(image.rows+2, image.cols+2, CV_8UC1);
            er_draw(channels, regions, nm_region_groups[i], group_img);
            if (ocr_is_tesseract)
            {
                group_img(nm_boxes[i]).copyTo(group_img);
                copyMakeBorder(group_img,group_img,15,15,15,15,BORDER_CONSTANT,Scalar(0));
            } else {
                group_img(Rect(1,1,image.cols,image.rows)).copyTo(group_img);
            }

            string output;
            vector<Rect>   boxes;
            vector<string> words;
            vector<float>  confidences;
            ocr->run(grey, group_img, output, &boxes, &words, &confidences, OCR_LEVEL_WORD);

            output.erase(remove(output.begin(), output.end(), '\n'), output.end());
            //cout << "OCR output = \"" << output << "\" lenght = " << output.size() << endl;

            if (output.size() < 3)
                continue;

            for (int j=0; j<(int)boxes.size(); j++)
            {
                if (ocr_is_tesseract)
                {
                    boxes[j].x += nm_boxes[i].x-15;
                    boxes[j].y += nm_boxes[i].y-15;
                }

                float min_confidence  = (ocr_is_tesseract)? (float)51. : (float)0.;
                float min_confidence4 = (ocr_is_tesseract)? (float)60. : (float)0.;
                //cout << "  word = " << words[j] << "\t confidence = " << confidences[j] << endl;
                if ((words[j].size() < 2) || (confidences[j] < min_confidence) ||
                    ((words[j].size()==2) && (words[j][0] == words[j][1])) ||
                    ((words[j].size()< 4) && (confidences[j] < min_confidence4)) ||
                     isRepetitive(words[j]))
                {
                    continue;
                }

                std::transform(words[j].begin(), words[j].end(), words[j].begin(), char_toupper);

                /* Increase confidence of predicted words matching a word in the lexicon */
                if (lex->size() > 0)
                {
                    if (find(lex->begin(), lex->end(), words[j]) == lex->end())
                        confidences[j] = 200;
                }

                final_words.push_back(words[j]);
                final_boxes.push_back(boxes[j]);
                final_confs.push_back(confidences[j]);
            }

        }

        /* Non Maximal Suppression using OCR confidence */
        float thr = 0.5;

        for (size_t i=0; i<final_words.size(); )
        {
            int to_delete = -1;
            for (size_t j=i+1; j<final_words.size(); )
            {
                to_delete = -1;
                Rect intersection = final_boxes[i] & final_boxes[j];
                float IoU = (float)intersection.area() / (final_boxes[i].area() + final_boxes[j].area() - intersection.area());
                if ((IoU > thr) || (intersection.area() > 0.8*final_boxes[i].area()) || (intersection.area() > 0.8*final_boxes[j].area()))
                {
                    // if regions overlap more than thr delete the one with lower confidence
                    to_delete = (final_confs[i] < final_confs[j]) ? i : j;

                    if (to_delete == (int)j )
                    {
                        final_words.erase(final_words.begin()+j);
                        final_boxes.erase(final_boxes.begin()+j);
                        final_confs.erase(final_confs.begin()+j);
                        continue;
                    } else {
                        break;
                    }
                }
                j++;
            }
            if (to_delete == (int)i )
            {
                final_words.erase(final_words.begin()+i);
                final_boxes.erase(final_boxes.begin()+i);
                final_confs.erase(final_confs.begin()+i);
                continue;
            }
            i++;
        }

        /* Predicted words which are not in the lexicon are filtered
           or changed to match one (when edit distance ratio < 0.34)*/
        float max_edit_distance_ratio = (float)0.34;
        for (size_t j=0; j<final_boxes.size(); j++)
        {

            if (lex->size() > 0)
            {
                if (find(lex->begin(), lex->end(), final_words[j]) == lex->end())
                {
                    int best_match = -1;
                    int best_dist  = final_words[j].size();
                    for (size_t l=0; l<lex->size(); l++)
                    {
                        int dist = edit_distance(lex->at(l),final_words[j]);
                        if (dist < best_dist)
                        {
                            best_match = l;
                            best_dist = dist;
                        }
                    }
                    if (best_dist/final_words[j].size() < max_edit_distance_ratio)
                        final_words[j] = lex->at(best_match);
                    else
                        continue;
                }
            }

            if ((find (lex->begin(), lex->end(), final_words[j])
                 == lex->end()) && (is_word_spotting) && (selected_lex != 0))
               continue;

            // Output final recognition in csv format compatible with the ICDAR Competition
            /*cout << final_boxes[j].tl().x << ","
                 << final_boxes[j].tl().y << ","
                 << min(final_boxes[j].br().x,image.cols-2)
                 << "," << final_boxes[j].tl().y << ","
                 << min(final_boxes[j].br().x,image.cols-2) << ","
                 << min(final_boxes[j].br().y,image.rows-2) << ","
                 << final_boxes[j].tl().x << ","
                 << min(final_boxes[j].br().y,image.rows-2) << ","
                 << final_words[j] << endl ;*/

            returnedNum++;
            returnedNumEach++;

            bool matched = false;
            for (vector<word>::iterator it=example->words.begin(); it!=example->words.end(); ++it)
            {
                word &t = (*it);

                // ICDAR protocol accepts recognition up to the first non alphanumeric char
                string alnum_value = t.value;
                for (size_t c=0; c<alnum_value.size(); c++)
                {
                    if (!isalnum(alnum_value[c]))
                    {
                        alnum_value = alnum_value.substr(0,c);
                        break;
                    }
                }

                std::transform(t.value.begin(), t.value.end(), t.value.begin(), char_toupper);
                if (((t.value==final_words[j]) || (alnum_value==final_words[j])) &&
                    !(final_boxes[j].tl().x > t.x+t.width || final_boxes[j].br().x < t.x ||
                      final_boxes[j].tl().y > t.y+t.height || final_boxes[j].br().y < t.y))
                {
                    matched = true;
                    returnedCorrectNum++;
                    returnedCorrectNumEach++;
                    //cout << "OK!" << endl;
                    break;
                }
            }

            if (!matched) // Take care of dontcare regions t.value == "###"
            for (vector<word>::iterator it=example->words.begin(); it!=example->words.end(); ++it)
            {
                word &t = (*it);
                std::transform(t.value.begin(), t.value.end(), t.value.begin(), char_toupper);
                if ((t.value == "###") &&
                     !(final_boxes[j].tl().x > t.x+t.width || final_boxes[j].br().x < t.x ||
                      final_boxes[j].tl().y > t.y+t.height || final_boxes[j].br().y < t.y))
                {
                    matched = true;
                    returnedNum--;
                    returnedNumEach--;
                    //cout << "DontCare!" << endl;
                    break;
                }
            }
            //if (!matched) cout << "FAIL." << endl;
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
        if ( (correctNumEach == 0) && (returnedNumEach == 0) )
        {
            p  = 1.;
            r  = 1.;
            f1 = 1.;
        }
        //printf("|%f|%f|%f|\n",r,p,f1);
        f1Each.push_back(f1);
    }

    double p = 1.0*returnedCorrectNum/returnedNum;
    double r = 1.0*returnedCorrectNum/correctNum;
    double f1 = 2*(p*r)/(p+r);

    printf("\n-------------------------------------------------------------------------\n");
    printf("ICDAR2015 -- Challenge 2: \"Focused Scene Text\" -- Task 4 \"End-to-End\"\n");
    if (is_word_spotting) printf("             Word spotting results -- ");
    else printf("             End-to-End recognition results -- ");
    switch (selected_lex)
    {
        case 0:
            printf("generic recognition (no given lexicon)\n");
            break;
        case 2:
            printf("weakly contextualized lexicon (624 words)\n");
            break;
        default:
            printf("strongly contextualized lexicon (100 words)\n");
            break;
    }
    printf("             Recall: %f | Precision: %f | F-score: %f\n", r, p, f1);
    printf("-------------------------------------------------------------------------\n\n");

    /*double mf1 = 0.0;
    for (vector<double>::iterator it=f1Each.begin(); it!=f1Each.end(); ++it)
    {
        mf1 += *it;
    }
    mf1 /= f1Each.size();
    printf("mean f1: %f\n", mf1);*/

    return 0;
}

#else

int main()
{
    std::cerr << "OpenCV was built without text module" << std::endl;
    return 0;
}

#endif // HAVE_OPENCV_TEXT
