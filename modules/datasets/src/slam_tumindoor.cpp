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
#include "opencv2/datasets/util.hpp"

#include <cstring>

namespace cv
{
namespace datasets
{

using namespace std;

class SLAM_tumindoorImp CV_FINAL : public SLAM_tumindoor
{
public:
    SLAM_tumindoorImp() {}
    //SLAM_tumindoorImp(const string &path);
    virtual ~SLAM_tumindoorImp() CV_OVERRIDE {}

    virtual void load(const string &path) CV_OVERRIDE;

private:
    void loadDataset(const string &path);
};

/*SLAM_tumindoorImp::SLAM_tumindoorImp(const string &path)
{
    loadDataset(path);
}*/

void SLAM_tumindoorImp::load(const string &path)
{
    loadDataset(path);
}

void SLAM_tumindoorImp::loadDataset(const string &path)
{
    train.push_back(vector< Ptr<Object> >());
    test.push_back(vector< Ptr<Object> >());
    validation.push_back(vector< Ptr<Object> >());

    string infoPath(path + "info/");

    // get info map name, .csv should be only one such file in folder
    string csvName;
    vector<string> infoNames;
    getDirList(infoPath, infoNames);
    for (vector<string>::iterator it=infoNames.begin(); it!=infoNames.end(); ++it)
    {
        string &name = *it;
        if (name.length()>3 && name.substr( name.length()-4, 4 )==".csv")
        {
            if (csvName.length()==0)
            {
                csvName = name;
            } else
            {
                printf("more than one .csv file in info folder\n");
                return;
            }
        }
    }
    if (csvName.length()==0)
    {
        printf("didn't find .csv file in info folder\n");
        return;
    }

    ifstream infile((infoPath + csvName).c_str());
    string line;
    while (getline(infile, line))
    {
        vector<string> elems;
        split(line, elems, ';');

        Ptr<SLAM_tumindoorObj> curr(new SLAM_tumindoorObj);

        curr->name = elems[0];
        if (curr->name.substr(0, strlen("dslr_left")) == "dslr_left")
        {
            curr->type = LEFT;
        } else
        if (curr->name.substr(0, strlen("dslr_right")) == "dslr_right")
        {
            curr->type = RIGHT;
        } else
        {
            curr->type = LADYBUG;
        }

        for (unsigned int i=0; i<4; ++i)
        {
            for (unsigned int j=0; j<4; ++j)
            {
                curr->transformMat(i, j) = atof(elems[1 + j + i*4].c_str());
            }
        }

        train.back().push_back(curr);
    }
}

Ptr<SLAM_tumindoor> SLAM_tumindoor::create()
{
    return Ptr<SLAM_tumindoorImp>(new SLAM_tumindoorImp);
}

}
}
