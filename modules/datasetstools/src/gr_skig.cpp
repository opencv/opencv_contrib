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

#include "opencv2/datasetstools/gr_skig.hpp"
#include "precomp.hpp"

#include <cstring>

namespace cv
{
namespace datasetstools
{

using namespace std;

GR_skig::GR_skig(const string &path)
{
    loadDataset(path);
}

void GR_skig::load(const string &path, int number)
{
    if (number!=0)
    {
        return;
    }

    loadDataset(path);
}

void GR_skig::loadDataset(const string &path)
{
    for (unsigned int i=1; i<=6; ++i)
    {
        char number[2];
        sprintf(number, "%u", i);
        string pathDatasetRgb(path + "subject" + number + "_rgb/");
        string pathDatasetDep(path + "subject" + number + "_dep/");

        vector<string> fileNames;
        getDirList(pathDatasetRgb, fileNames);
        for (vector<string>::iterator it=fileNames.begin(); it!=fileNames.end(); ++it)
        {
            string &file = *it;

            gestureSkig curr;
            curr.rgb = pathDatasetRgb + file;
            curr.dep = file;
            curr.dep[0] = 'K';
            curr.dep = pathDatasetDep + curr.dep;

            size_t pos = file.find("person_"); // TODO: check ::npos
            curr.person = (unsigned char)atoi( file.substr(pos+strlen("person_"), 1).c_str() );
            pos = file.find("backgroud_");
            curr.background = (backgroundType)atoi( file.substr(pos+strlen("backgroud_"), 1).c_str() );
            pos = file.find("illumination_");
            curr.illumination = (illuminationType)atoi( file.substr(pos+strlen("illumination_"), 1).c_str() );
            pos = file.find("pose_");
            curr.pose = (poseType)atoi( file.substr(pos+strlen("pose_"), 1).c_str() );
            pos = file.find("actionType_");
            curr.type = (actionType)atoi( file.substr(pos+strlen("actionType_"), 2).c_str() );

            train.push_back(curr);
        }
    }
}

}
}
