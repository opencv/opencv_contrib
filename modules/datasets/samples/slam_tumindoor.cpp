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

#include "opencv2/datasets/slam_tumindoor.hpp"

#include <opencv2/core.hpp>

#include <cstdio>

#include <string>
#include <vector>

using namespace std;
using namespace cv;
using namespace cv::datasets;

int main(int argc, char *argv[])
{
    const char *keys =
            "{ help h usage ? |    | show this message }"
            "{ path p         |true| path to dataset folders }";
    CommandLineParser parser(argc, argv, keys);
    string path(parser.get<string>("path"));
    if (parser.has("help") || path=="true")
    {
        parser.printMessage();
        return -1;
    }

    Ptr<SLAM_tumindoor> dataset = SLAM_tumindoor::create();
    dataset->load(path);

    // ***************
    // dataset contains image and its information.
    // For example, let output first image information and dataset size.
    printf("dataset size: %u\n", (unsigned int)dataset->getTrain().size());

    SLAM_tumindoorObj *example = static_cast<SLAM_tumindoorObj *>(dataset->getTrain()[0].get());
    printf("first image:\ntype: %u\n", example->type);

    string imagePath(path);
    if (LADYBUG == example->type)
    {
        imagePath += "ladybug/";
    } else
    {
        imagePath += "dslr/";
    }
    printf("file name: %s\n", (imagePath + example->name).c_str());

    printf("\ntransformation matrix:\n");
    for (unsigned int i=0; i<4; ++i)
    {
        for (unsigned int j=0; j<4; ++j)
        {
            printf("%f ", example->transformMat(i, j));
        }
        printf("\n");
    }

    return 0;
}
