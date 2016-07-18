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
#include "opencv2/datasets/util.hpp"

namespace cv
{
namespace datasets
{

using namespace std;

class GR_chalearnImp : public GR_chalearn
{
public:
    GR_chalearnImp() {}
    //GR_chalearnImp(const string &path);
    virtual ~GR_chalearnImp() {}

    virtual void load(const string &path);

private:
    void loadDataset(const string &path);

    void loadDatasetPart(const string &path, vector< Ptr<Object> > &dataset_, bool loadLabels);
};

/*GR_chalearnImp::GR_chalearnImp(const string &path)
{
    loadDataset(path);
}*/

void GR_chalearnImp::load(const string &path)
{
    loadDataset(path);
}

void GR_chalearnImp::loadDatasetPart(const string &path, vector< Ptr<Object> > &dataset_, bool loadLabels)
{
    vector<string> fileNames;
    getDirList(path, fileNames);
    for (vector<string>::iterator it=fileNames.begin(); it!=fileNames.end(); ++it)
    {
        Ptr<GR_chalearnObj> curr(new GR_chalearnObj);
        curr->name = *it;
        curr->nameColor = curr->name + "/" + curr->name + "_color.mp4";
        curr->nameDepth = curr->name + "/" + curr->name + "_depth.mp4";
        curr->nameUser = curr->name + "/" + curr->name + "_user.mp4";

        // loading video info
        string fileVideoInfo(path + curr->name + "/" + curr->name + "_data.csv");
        ifstream infile(fileVideoInfo.c_str());
        string line;
        getline(infile, line);
        vector<string> elems;
        split(line, elems, ',');
        curr->numFrames = atoi(elems[0].c_str());
        curr->fps = atoi(elems[1].c_str());
        curr->depth = atoi(elems[2].c_str());

        // loading ground truth
        if (loadLabels)
        {
            string fileGroundTruth(path + curr->name + "/" + curr->name + "_labels.csv");
            ifstream infileGroundTruth(fileGroundTruth.c_str());
            while (getline(infileGroundTruth, line))
            {
                vector<string> elems2;
                split(line, elems2, ',');

                groundTruth currGroundTruth;
                currGroundTruth.gestureID = atoi(elems2[0].c_str());
                currGroundTruth.initialFrame = atoi(elems2[1].c_str());
                currGroundTruth.lastFrame = atoi(elems2[2].c_str());

                curr->groundTruths.push_back(currGroundTruth);
            }
        }

        // loading skeleton
        string fileSkeleton(path + curr->name + "/" + curr->name + "_skeleton.csv");
        ifstream infileSkeleton(fileSkeleton.c_str());
        while (getline(infileSkeleton, line))
        {
            skeleton currSkeleton;

            vector<string> elems2;
            split(line, elems2, ',');

            for (unsigned int i=0, numJoin=0; i<elems2.size(); i+=9, ++numJoin)
            {
                currSkeleton.s[numJoin].Wx = atof(elems2[i+0].c_str());
                currSkeleton.s[numJoin].Wy = atof(elems2[i+1].c_str());
                currSkeleton.s[numJoin].Wz = atof(elems2[i+2].c_str());
                currSkeleton.s[numJoin].Rx = atof(elems2[i+3].c_str());
                currSkeleton.s[numJoin].Ry = atof(elems2[i+4].c_str());
                currSkeleton.s[numJoin].Rz = atof(elems2[i+5].c_str());
                currSkeleton.s[numJoin].Rw = atof(elems2[i+6].c_str());
                currSkeleton.s[numJoin].Px = atof(elems2[i+7].c_str());
                currSkeleton.s[numJoin].Py = atof(elems2[i+8].c_str());
            }

            curr->skeletons.push_back(currSkeleton);
        }

        dataset_.push_back(curr);
    }
}

void GR_chalearnImp::loadDataset(const string &path)
{
    train.push_back(vector< Ptr<Object> >());
    test.push_back(vector< Ptr<Object> >());
    validation.push_back(vector< Ptr<Object> >());

    string pathTrain(path + "Train/");
    loadDatasetPart(pathTrain, train.back(), true);

    // freely available validation set doesn't have labels
    string pathValidation(path + "Validation/");
    loadDatasetPart(pathValidation, validation.back(), false);
}

Ptr<GR_chalearn> GR_chalearn::create()
{
    return Ptr<GR_chalearnImp>(new GR_chalearnImp);
}

}
}
