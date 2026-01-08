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

#include "opencv2/datasets/hpe_humaneva.hpp"

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
            "{ path p         |true| path to dataset folder }";
    CommandLineParser parser(argc, argv, keys);
    string path(parser.get<string>("path"));
    if (parser.has("help") || path=="true")
    {
        parser.printMessage();
        return -1;
    }

    for (unsigned int i=1; i<=2; ++i)
    {
        printf("\tHumanEva %u\n", i);
        char number[2];
        sprintf(number, "%u", i);
        string pathCurr(path+"HumanEva_"+number+"/");

        Ptr<HPE_humaneva> dataset = HPE_humaneva::create(i);
        dataset->load(pathCurr);

        // ***************
        // dataset contains pair of rgb\dep images
        // For example, let output train size and last element.
        HPE_humanevaObj *example = static_cast<HPE_humanevaObj *>(dataset->getTrain().back().get());
        printf("train size: %u\n", (unsigned int)dataset->getTrain().size());
        printf("last train video:\n");
        printf("person: %u\n", example->person);
        printf("action: %s\n", example->action.c_str());
        printf("type1: %u\n", example->type1);
        printf("type2: %s\n", example->type2.c_str());
        printf("filename: %s\n", example->fileName.c_str());
        printf("num images: %u\n", (int)example->imageNames.size());
        printf("ofs:");
        for (int j=0; j<3; ++j)
        {
            printf(" %f", example->ofs(0, j));
        }
        printf("\n");
    }

    return 0;
}
