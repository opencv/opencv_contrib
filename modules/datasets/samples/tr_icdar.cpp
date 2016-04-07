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

#include "opencv2/datasets/tr_icdar.hpp"

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
            "{ path p         |true| path to dataset root folder }";
    CommandLineParser parser(argc, argv, keys);
    string path(parser.get<string>("path"));
    if (parser.has("help") || path=="true")
    {
        parser.printMessage();
        return -1;
    }

    // loading train & test images description
    Ptr<TR_icdar> dataset = TR_icdar::create();
    dataset->load(path);

    // ***************
    // dataset. train & test contains images description.
    // For example, let output the last element in train set and it's description.
    // And their sizes.
    printf("train size: %u\n", (unsigned int)dataset->getTrain().size());
    printf("test size: %u\n", (unsigned int)dataset->getTest().size());

    TR_icdarObj *example = static_cast<TR_icdarObj *>(dataset->getTrain().back().get());
    printf("last element:\nfile name: %s", example->fileName.c_str());
    printf("\nlex100: ");
    for (vector<string>::iterator it=example->lex100.begin(); it!=example->lex100.end(); ++it)
    {
        printf("%s,", (*it).c_str());
    }
    printf("\nlexFULL: ");
    for (vector<string>::iterator it=example->lexFull.begin(); it!=example->lexFull.end(); ++it)
    {
        printf("%s,", (*it).c_str());
    }
    printf("\nwords:\n");
    for (vector<word>::iterator it=example->words.begin(); it!=example->words.end(); ++it)
    {
        word &t = (*it);
        printf("%s\nheight: %u, width: %u, x: %u, y: %u\n",
               t.value.c_str(), t.height, t.width, t.x, t.y);
    }

    return 0;
}
