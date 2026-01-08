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

#include "opencv2/datasets/msm_epfl.hpp"

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
            "{ path p         |true| path to dataset (bounding/, camera/, P/, png/ folders) }";
    CommandLineParser parser(argc, argv, keys);
    string path(parser.get<string>("path"));
    if (parser.has("help") || path=="true")
    {
        parser.printMessage();
        return -1;
    }

    Ptr<MSM_epfl> dataset = MSM_epfl::create();
    dataset->load(path);

    // ***************
    // dataset contains all information for each image.
    // For example, let output dataset size and first object.
    printf("dataset size: %u\n", (unsigned int)dataset->getTrain().size());
    MSM_epflObj *example = static_cast<MSM_epflObj *>(dataset->getTrain()[0].get());
    printf("first image:\nname: %s\n", example->imageName.c_str());

    printf("\nbounding:\n");
    for (int i=0; i<2; ++i)
    {
        for (int j=0; j<3; ++j)
        {
            printf("%f ", example->bounding(i, j));
        }
        printf("\n");
    }

    printf("\ncamera:\n");
    for (int i=0; i<3; ++i)
    {
        for (int j=0; j<3; ++j)
        {
            printf("%f ", example->camera.mat1(i, j));
        }
        printf("\n");
    }
    printf("\n");

    for (int i=0; i<3; ++i)
    {
        printf("%f ", example->camera.mat2[i]);
    }
    printf("\n\n");

    for (int i=0; i<3; ++i)
    {
        for (int j=0; j<3; ++j)
        {
            printf("%f ", example->camera.mat3(i, j));
        }
        printf("\n");
    }
    printf("\n");

    for (int i=0; i<3; ++i)
    {
        printf("%f ", example->camera.mat4[i]);
    }
    printf("\n\n");

    printf("image width: %u, height: %u\n", example->camera.imageWidth, example->camera.imageHeight);

    printf("\nP:\n");
    for (int i=0; i<3; ++i)
    {
        for (int j=0; j<4; ++j)
        {
            printf("%f ", example->p(i, j));
        }
        printf("\n");
    }

    return 0;
}
