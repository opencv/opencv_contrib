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
// Copyright (C) 2015, Itseez Inc, all rights reserved.
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

#include "opencv2/datasets/or_pascal.hpp"

#include <opencv2/core.hpp>

#include <cstdio>
#include <cstdlib> // atoi

#include <string>
#include <vector>
#include <set>

using namespace std;
using namespace cv;
using namespace cv::datasets;

int main(int argc, char *argv[])
{
    const char *keys =
            "{ help h usage ? |    | show this message }"
            "{ path p         |true| path to folder with dataset }";
    CommandLineParser parser(argc, argv, keys);
    string path(parser.get<string>("path"));
    if (parser.has("help") || path=="true")
    {
        parser.printMessage();
        return -1;
    }

    Ptr<OR_pascal> dataset = OR_pascal::create();
    dataset->load(path);

    // Print train/test/validation size and first example
    OR_pascalObj *example;
    vector< Ptr<Object> > &train = dataset->getTrain();
    printf("\ntrain:\nsize: %u", (unsigned int)train.size());
    example = static_cast<OR_pascalObj *>(train[0].get());
    printf("\nfirst image: \n%s", example->filename.c_str());

    printf("\nsize:");
    printf("\n - width: %d", example->width);
    printf("\n - height: %d", example->height);
    printf("\n - depth: %d", example->depth);

    for (unsigned int i = 0; i < example->objects.size(); i++)
    {
        printf("\nobject %d", i);
        printf("\nname: %s", example->objects[i].name.c_str());
        printf("\npose: %s", example->objects[i].pose.c_str());
        printf("\ntruncated: %d", example->objects[i].truncated);
        printf("\ndifficult: %d", example->objects[i].difficult);
        printf("\noccluded: %d", example->objects[i].occluded);

        printf("\nbounding box:");
        printf("\n - xmin: %d", example->objects[i].xmin);
        printf("\n - ymin: %d", example->objects[i].ymin);
        printf("\n - xmax: %d", example->objects[i].xmax);
        printf("\n - ymax: %d", example->objects[i].ymax);
    }

    vector< Ptr<Object> > &test = dataset->getTest();
    printf("\ntest:\nsize: %u", (unsigned int)test.size());
    example = static_cast<OR_pascalObj *>(test[0].get());
    printf("\nfirst image: \n%s", example->filename.c_str());

    vector< Ptr<Object> > &validation = dataset->getValidation();
    printf("\nvalidation:\nsize: %u", (unsigned int)validation.size());
    example = static_cast<OR_pascalObj *>(validation[0].get());
    printf("\nfirst image: \n%s\n", example->filename.c_str());

    return 0;
}
