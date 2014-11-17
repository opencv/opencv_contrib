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

#include "opencv2/datasets/fr_adience.hpp"

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
            "{ path p         |true| path to dataset folder and splits }";
    CommandLineParser parser(argc, argv, keys);
    string path(parser.get<string>("path"));
    if (parser.has("help") || path=="true")
    {
        parser.printMessage();
        return -1;
    }

    Ptr<FR_adience> dataset = FR_adience::create();
    dataset->load(path);

    // ***************
    // dataset contains for each object its images.
    // For example, let output splits number, dataset size and last image.
    int numSplits = dataset->getNumSplits();
    printf("splits number: %u\n", numSplits);
    printf("dataset size: %u\n", (unsigned int)dataset->getTrain().size());

    FR_adienceObj *example = static_cast<FR_adienceObj *>(dataset->getTrain().back().get());
    printf("last image:\n");
    printf("user_id: %s\n", example->user_id.c_str());
    printf("original_image: %s\n", example->original_image.c_str());
    printf("face_id: %u\n", example->face_id);
    printf("age: %s\n", example->age.c_str());
    printf("gender: ");
    if (example->gender == male)
    {
        printf("m\n");
    } else
    if (example->gender == female)
    {
        printf("f\n");
    } else
    {
        printf("none\n");
    }
    printf("x: %u\n", example->x);
    printf("y: %u\n", example->y);
    printf("dx: %u\n", example->dx);
    printf("dy: %u\n", example->dy);
    printf("tilt_ang: %d\n", example->tilt_ang);
    printf("fiducial_yaw_angle: %d\n", example->fiducial_yaw_angle);
    printf("fiducial_score: %u\n", example->fiducial_score);

    return 0;
}
