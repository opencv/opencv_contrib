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

#include "opencv2/datasets/slam_kitti.hpp"

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

    Ptr<SLAM_kitti> dataset = SLAM_kitti::create();
    dataset->load(path);

    // ***************
    // dataset contains sequence with name and its data.
    // For example, let output first sequence and dataset size.
    printf("dataset size: %u\n", (unsigned int)dataset->getTrain().size());

    SLAM_kittiObj *example = static_cast<SLAM_kittiObj *>(dataset->getTrain()[0].get());
    printf("first dataset sequence:\n%s\n", example->name.c_str());

    /*string pathVelodyne(path + "sequences/" + example->name + "/velodyne/");
    for (vector<string>::iterator it=example->velodyne.begin(); it!=example->velodyne.end(); ++it)
    {
        printf("%s\n", (pathVelodyne + (*it)).c_str());
    }*/
    printf("number of velodyne images: %u\n", (unsigned int)example->velodyne.size());

    for (unsigned int i=0; i<=3; ++i)
    {
        /*char tmp[2];
        sprintf(tmp, "%u", i);
        // 0,1 - gray, 2,3 - color
        string currPath(path + "sequences/" + example->name + "/image_" + tmp + "/");
        for (vector<string>::iterator it=example->images[i].begin(); it!=example->images[i].end(); ++it)
        {
            printf("%s\n", (currPath + (*it)).c_str());
        }*/
        printf("number of images %u: %u\n", i, (unsigned int)example->images[i].size());
    }

    /*printf("times:\n");
    for (vector<double>::iterator it=example->times.begin(); it!=example->times.end(); ++it)
    {
        printf("%f ", *it);
    }
    printf("\n");*/
    printf("number of times: %u\n", (unsigned int)example->times.size());

    /*printf("poses:\n");
    for (vector<pose>::iterator it=example->posesArray.begin(); it!=example->posesArray.end(); ++it)
    {
        for (unsigned int i=0; i<12; ++i)
        {
            printf("%f ", (*it).elem[i]);
        }
        printf("\n");
    }*/
    printf("number of poses: %u\n", (unsigned int)example->posesArray.size());

    for (unsigned int i=0; i<4; ++i)
    {
        printf("calibration %u:\n", i);
        for (vector<double>::iterator it=example->p[i].begin(); it!=example->p[i].end(); ++it)
        {
            printf("%f ", *it);
        }
        printf("\n");
    }

    return 0;
}
