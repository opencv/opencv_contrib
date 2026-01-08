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

#include "opencv2/datasets/gr_chalearn.hpp"

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
            "{ path p         |true| path to dataset folder }";
    CommandLineParser parser(argc, argv, keys);
    string path(parser.get<string>("path"));
    if (parser.has("help") || path=="true")
    {
        parser.printMessage();
        return -1;
    }

    Ptr<GR_chalearn> dataset = GR_chalearn::create();
    dataset->load(path);

    // ***************
    // dataset contains information for each sample.
    // For example, let output dataset size and first element.
    printf("train size: %u\n", (unsigned int)dataset->getTrain().size());
    printf("validation size: %u\n", (unsigned int)dataset->getValidation().size());
    GR_chalearnObj *example = static_cast<GR_chalearnObj *>(dataset->getTrain()[0].get());
    printf("first dataset sample:\n%s\n", example->name.c_str());
    printf("color video:\n%s\n", example->nameColor.c_str());
    printf("depth video:\n%s\n", example->nameDepth.c_str());
    printf("user video:\n%s\n", example->nameUser.c_str());
    printf("video:\nnumber of frames: %u\nfps: %u\nmaximum depth: %u\n", example->numFrames, example->fps, example->depth);
    for (vector<groundTruth>::iterator it=example->groundTruths.begin(); it!=example->groundTruths.end(); ++it)
    {
        printf("gestureID: %u, initial frame: %u, last frame: %u\n", (*it).gestureID, (*it).initialFrame, (*it).lastFrame);
    }
    printf("skeletons number: %u\n", (unsigned int)example->skeletons.size());
    skeleton &last = example->skeletons.back();
    printf("last skeleton:\n");
    for (unsigned int i=0; i<20; ++i)
    {
        printf("Wx: %f, Wy: %f, Wz: %f, Rx: %f, Ry: %f, Rz: %f, Rw: %f, Px: %f, Py: %f\n",
               last.s[i].Wx, last.s[i].Wy, last.s[i].Wz, last.s[i].Rx,
               last.s[i].Ry, last.s[i].Rz, last.s[i].Rw, last.s[i].Px, last.s[i].Py);
    }

    return 0;
}
