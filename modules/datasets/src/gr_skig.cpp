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

#include "opencv2/datasets/gr_skig.hpp"
#include "opencv2/datasets/util.hpp"

#include <cstring>

namespace cv
{
namespace datasets
{

using namespace std;

class GR_skigImp : public GR_skig
{
public:
    GR_skigImp() {}
    //GR_skigImp(const string &path);
    virtual ~GR_skigImp() {}

    virtual void load(const string &path);

private:
    void loadDataset(const string &path);
};

/*GR_skigImp::GR_skigImp(const string &path)
{
    loadDataset(path);
}*/

void GR_skigImp::load(const string &path)
{
    loadDataset(path);
}

void GR_skigImp::loadDataset(const string &path)
{
    train.push_back(vector< Ptr<Object> >());
    test.push_back(vector< Ptr<Object> >());
    validation.push_back(vector< Ptr<Object> >());

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

            Ptr<GR_skigObj> curr(new GR_skigObj);
            curr->rgb = pathDatasetRgb + file;
            curr->dep = file;
            curr->dep[0] = 'K';
            curr->dep = pathDatasetDep + curr->dep;

            size_t posPerson = file.find("person_");
            size_t posBackground = file.find("backgroud_");
            size_t posIllumination = file.find("illumination_");
            size_t posPose = file.find("pose_");
            size_t posType = file.find("actionType_");
            if (string::npos != posPerson &&
                string::npos != posBackground &&
                string::npos != posIllumination &&
                string::npos != posPose &&
                string::npos != posType)
            {
                curr->person = (unsigned char)atoi( file.substr(posPerson + strlen("person_"), 1).c_str() );
                curr->background = (backgroundType)atoi( file.substr(posBackground + strlen("backgroud_"), 1).c_str() );
                curr->illumination = (illuminationType)atoi( file.substr(posIllumination + strlen("illumination_"), 1).c_str() );
                curr->pose = (poseType)atoi( file.substr(posPose + strlen("pose_"), 1).c_str() );
                curr->type = (actionType)atoi( file.substr(posType + strlen("actionType_"), 2).c_str() );

                train.back().push_back(curr);
            } else
            {
                printf("incorrect file name: %s", file.c_str());
            }
        }
    }
}

Ptr<GR_skig> GR_skig::create()
{
    return Ptr<GR_skigImp>(new GR_skigImp);
}

}
}
