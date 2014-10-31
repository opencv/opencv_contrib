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
#include "opencv2/datasets/util.hpp"

namespace cv
{
namespace datasets
{

using namespace std;

class SLAM_kittiImp : public SLAM_kitti
{
public:
    SLAM_kittiImp() {}
    //SLAM_kittiImp(const string &path);
    virtual ~SLAM_kittiImp() {}

    virtual void load(const string &path);

private:
    void loadDataset(const string &path);
};

/*SLAM_kittiImp::SLAM_kittiImp(const string &path)
{
    loadDataset(path);
}*/

void SLAM_kittiImp::load(const string &path)
{
    loadDataset(path);
}

void SLAM_kittiImp::loadDataset(const string &path)
{
    train.push_back(vector< Ptr<Object> >());
    test.push_back(vector< Ptr<Object> >());
    validation.push_back(vector< Ptr<Object> >());

    string pathSequence(path + "sequences/");
    vector<string> fileNames;
    getDirList(pathSequence, fileNames);
    for (vector<string>::iterator it=fileNames.begin(); it!=fileNames.end(); ++it)
    {
        Ptr<SLAM_kittiObj> curr(new SLAM_kittiObj);
        curr->name = *it;

        string currPath(pathSequence + curr->name);

        // loading velodyne
        string pathVelodyne(currPath + "/velodyne/");
        vector<string> velodyneNames;
        getDirList(pathVelodyne, velodyneNames);
        for (vector<string>::iterator itV=velodyneNames.begin(); itV!=velodyneNames.end(); ++itV)
        {
            curr->velodyne.push_back(*itV);
        }

        // loading gray & color images
        for (unsigned int i=0; i<=3; ++i)
        {
            char tmp[2];
            sprintf(tmp, "%u", i);
            string pathImage(currPath + "/image_" + tmp + "/");
            vector<string> imageNames;
            getDirList(pathImage, imageNames);
            for (vector<string>::iterator itImage=imageNames.begin(); itImage!=imageNames.end(); ++itImage)
            {
                curr->images[i].push_back(*itImage);
            }
        }

        // loading times
        ifstream infile((currPath + "/times.txt").c_str());
        string line;
        while (getline(infile, line))
        {
            curr->times.push_back(atof(line.c_str()));
        }

        // loading calibration
        ifstream infile2((currPath + "/calib.txt").c_str());
        for (unsigned int i=0; i<4; ++i)
        {
            getline(infile2, line);
            vector<string> elems;
            split(line, elems, ' ');
            vector<string>::iterator itE=elems.begin();
            for (++itE; itE!=elems.end(); ++itE)
            {
                curr->p[i].push_back(atof((*itE).c_str()));
            }
        }

        // loading poses
        ifstream infile3((path + "poses/" + curr->name + ".txt").c_str());
        while (getline(infile3, line))
        {
            pose p;

            unsigned int i=0;
            vector<string> elems;
            split(line, elems, ' ');
            for (vector<string>::iterator itE=elems.begin(); itE!=elems.end(); ++itE, ++i)
            {
                if (i>11)
                {
                    break;
                }
                p.elem[i] = atof((*itE).c_str());
            }

            curr->posesArray.push_back(p);
        }

        train.back().push_back(curr);
    }
}

Ptr<SLAM_kitti> SLAM_kitti::create()
{
    return Ptr<SLAM_kittiImp>(new SLAM_kittiImp);
}

}
}
