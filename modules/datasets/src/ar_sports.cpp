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

#include "opencv2/datasets/ar_sports.hpp"
#include "opencv2/datasets/util.hpp"

namespace cv
{
namespace datasets
{

using namespace std;

class AR_sportsImp : public AR_sports
{
public:
    AR_sportsImp() {}
    //AR_sportsImp(const string &path);
    virtual ~AR_sportsImp() {}

    virtual void load(const string &path) CV_OVERRIDE;

private:
    void loadDataset(const string &path);

    void loadDatasetPart(const string &fileName, vector< Ptr<Object> > &dataset_);
};

void AR_sportsImp::loadDatasetPart(const string &fileName, vector< Ptr<Object> > &dataset_)
{
    ifstream infile(fileName.c_str());
    string videoUrl, labels;
    while (infile >> videoUrl >> labels)
    {
        Ptr<AR_sportsObj> curr(new AR_sportsObj);
        curr->videoUrl = videoUrl;

        vector<string> elems;
        split(labels, elems, ',');
        for (vector<string>::iterator it=elems.begin(); it!=elems.end(); ++it)
        {
            curr->labels.push_back(atoi((*it).c_str()));
        }

        dataset_.push_back(curr);
    }
}

/*AR_sportsImp::AR_sportsImp(const string &path)
{
    loadDataset(path);
}*/

void AR_sportsImp::load(const string &path)
{
    loadDataset(path);
}

void AR_sportsImp::loadDataset(const string &path)
{
    train.push_back(vector< Ptr<Object> >());
    test.push_back(vector< Ptr<Object> >());
    validation.push_back(vector< Ptr<Object> >());

    string trainPath(path + "original/train_partition.txt");
    string testPath(path + "original/test_partition.txt");

    // loading train video urls & labels
    loadDatasetPart(trainPath, train.back());

    // loading test video urls & labels
    loadDatasetPart(testPath, test.back());
}

Ptr<AR_sports> AR_sports::create()
{
    return Ptr<AR_sportsImp>(new AR_sportsImp);
}

}
}
