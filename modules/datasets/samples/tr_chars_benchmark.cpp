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

#include "opencv2/datasets/tr_chars.hpp"
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

int main(int argc, char *argv[])
{
    const char *keys =
            "{ help h usage ? |    | show this message }"
            "{ path p         |true| path to dataset description file ( list_English_Img.m ) and Img folder.}";
    CommandLineParser parser(argc, argv, keys);
    string path(parser.get<string>("path"));
    if (parser.has("help") || path=="true")
    {
        parser.printMessage();
        return -1;
    }

    Ptr<TR_chars> dataset = TR_chars::create();
    dataset->load(path);

    // ***************
    // dataset. train, test contain information about each element of appropriate sets and splits.
    // For example, let output first elements of these vectors and their sizes for last split.
    // And number of splits.
    int numSplits = dataset->getNumSplits();
    printf("splits number: %u\n", numSplits);

    vector< Ptr<Object> > &currTrain = dataset->getTrain(numSplits-1);
    vector< Ptr<Object> > &currTest = dataset->getTest(numSplits-1);
    vector< Ptr<Object> > &currValidation = dataset->getValidation(numSplits-1);
    printf("train size: %u\n", (unsigned int)currTrain.size());
    printf("test size: %u\n", (unsigned int)currTest.size());
    printf("validation size: %u\n", (unsigned int)currValidation.size());


    // WARNING: The order of classes' labels is different in Chars74k and in the output of our classifier
    string src_classes = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"; // labels order as in the clasifier output
    string tar_classes = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"; // labels order as in the Chars74k dataset

    Ptr<OCRHMMDecoder::ClassifierCallback> ocr = loadOCRHMMClassifierCNN("OCRBeamSearch_CNN_model_data.xml.gz");

    int numOK = 0;
    int upperNumOK = 0;

    for (unsigned int i=0; i<(unsigned int)currTest.size(); i++)
    {
        TR_charsObj *exampleTest = static_cast<TR_charsObj *>(currTest[i].get());
        printf("processed image: %u, name: %s\n", i, exampleTest->imgName.c_str());
        printf("  label: %u,", exampleTest->label);

        string imfilename = path+string("/Img/")+exampleTest->imgName.c_str()+string(".png");
        Mat image  = imread(imfilename);
        vector<int> out_classes;
        vector<double> out_confidences;
        ocr->eval(image, out_classes, out_confidences);
        int prediction = 1 + tar_classes.find_first_of(src_classes[out_classes[0]]);
        printf(" prediction: %u\n", prediction);

        if (exampleTest->label == prediction)
            numOK++;

        char l = tar_classes[exampleTest->label];
        char p = tar_classes[prediction];
        if (toupper(l) == toupper(p))
            upperNumOK++;
    }

    printf("\n---------------------------------------------\n");
    printf("Chars74k Classification Accuracy (case-sensitive): %f\n",(float)numOK/currTest.size());
    printf("Chars74k Classification Accuracy (case-insensitive): %f\n",(float)upperNumOK/currTest.size());

    return 0;
}

#else

int main()
{
    std::cerr << "OpenCV was built without text module" << std::endl;
    return 0;
}

#endif // HAVE_OPENCV_TEXT
