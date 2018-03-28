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
#include "opencv2/datasets/util.hpp"

namespace cv
{
namespace datasets
{

using namespace std;

class MSM_epflImp CV_FINAL : public MSM_epfl
{
public:
    MSM_epflImp() {}
    //MSM_epflImp(const string &path);
    virtual ~MSM_epflImp() CV_OVERRIDE {}

    virtual void load(const string &path) CV_OVERRIDE;

private:
    void loadDataset(const string &path);
};

/*MSM_epflImp::MSM_epflImp(const string &path)
{
    loadDataset(path);
}*/

void MSM_epflImp::load(const string &path)
{
    loadDataset(path);
}

void MSM_epflImp::loadDataset(const string &path)
{
    train.push_back(vector< Ptr<Object> >());
    test.push_back(vector< Ptr<Object> >());
    validation.push_back(vector< Ptr<Object> >());

    string pathBounding(path + "bounding/");
    string pathCamera(path + "camera/");
    string pathP(path + "P/");
    string pathPng(path + "png/");

    vector<string> fileNames;
    getDirList(pathPng, fileNames);
    for (vector<string>::iterator it=fileNames.begin(); it!=fileNames.end(); ++it)
    {
        Ptr<MSM_epflObj> curr(new MSM_epflObj);
        curr->imageName = *it;

        // load boundary
        string fileBounding(pathBounding + curr->imageName + ".bounding");
        ifstream infile(fileBounding.c_str());
        for (int k=0; k<2; ++k)
        {
            for (int j=0; j<3; ++j)
            {
                infile >> curr->bounding(k, j);
            }
        }

        // load camera parameters
        string fileCamera(pathCamera + curr->imageName + ".camera");
        ifstream infileCamera(fileCamera.c_str());
        for (int i=0; i<3; ++i)
        {
            for (int j=0; j<3; ++j)
            {
                infileCamera >> curr->camera.mat1(i, j);
            }
        }

        for (int i=0; i<3; ++i)
        {
            infileCamera >> curr->camera.mat2[i];
        }

        for (int i=0; i<3; ++i)
        {
            for (int j=0; j<3; ++j)
            {
                infileCamera >> curr->camera.mat3(i, j);
            }
        }

        for (int i=0; i<3; ++i)
        {
            infileCamera >> curr->camera.mat4[i];
        }

        infileCamera >> curr->camera.imageWidth >> curr->camera.imageHeight;

        // load P
        string fileP(pathP + curr->imageName + ".P");
        ifstream infileP(fileP.c_str());
        for (int k=0; k<3; ++k)
        {
            for (int j=0; j<4; ++j)
            {
                infileP >> curr->p(k, j);
            }
        }

        train.back().push_back(curr);
    }
}

Ptr<MSM_epfl> MSM_epfl::create()
{
    return Ptr<MSM_epflImp>(new MSM_epflImp);
}

}
}
