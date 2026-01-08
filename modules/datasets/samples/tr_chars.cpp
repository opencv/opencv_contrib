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

#include "opencv2/datasets/tr_chars.hpp"

#include <opencv2/core.hpp>

#include <cstdio>
#include <cstdlib> // atoi

#include <string>
#include <vector>

using namespace std;
using namespace cv;
using namespace cv::datasets;

int main(int argc, char *argv[])
{
    const char *keys =
            "{ help h usage ? |    | show this message }"
            "{ path p         |true| path to dataset description file: list_English_Img.m }";
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

    TR_charsObj *exampleTrain = static_cast<TR_charsObj *>(currTrain[0].get());
    TR_charsObj *exampleTest = static_cast<TR_charsObj *>(currTest[0].get());
    TR_charsObj *exampleValidation = static_cast<TR_charsObj *>(currValidation[0].get());
    printf("first train element:\nname: %s\n", exampleTrain->imgName.c_str());
    printf("label: %u\n", exampleTrain->label);
    printf("first test element:\nname: %s\n", exampleTest->imgName.c_str());
    printf("label: %u\n", exampleTest->label);
    printf("first validation element:\nname: %s\n", exampleValidation->imgName.c_str());
    printf("label: %u\n", exampleValidation->label);

    return 0;
}
