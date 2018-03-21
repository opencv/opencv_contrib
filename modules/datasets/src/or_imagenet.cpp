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

#include "opencv2/datasets/or_imagenet.hpp"
#include "opencv2/datasets/util.hpp"

#include <map>

namespace cv
{
namespace datasets
{

using namespace std;

class OR_imagenetImp CV_FINAL : public OR_imagenet
{
public:
    OR_imagenetImp() {}
    //OR_imagenetImp(const string &path);
    virtual ~OR_imagenetImp() CV_OVERRIDE {}

    virtual void load(const string &path) CV_OVERRIDE;

private:
    void loadDataset(const string &path);

    void numberToString(int number, string &out);
};

/*OR_imagenetImp::OR_imagenetImp(const string &path)
{
    loadDataset(path);
}*/

void OR_imagenetImp::load(const string &path)
{
    loadDataset(path);
}

void OR_imagenetImp::numberToString(int number, string &out)
{
    char numberStr[9];
    sprintf(numberStr, "%u", number);
    for (unsigned int i=0; i<8-strlen(numberStr); ++i)
    {
        out += "0";
    }
    out += numberStr;
}

void OR_imagenetImp::loadDataset(const string &path)
{
    train.push_back(vector< Ptr<Object> >());
    test.push_back(vector< Ptr<Object> >());
    validation.push_back(vector< Ptr<Object> >());

    map<string, int> labels;
    ifstream infile((path + "labels.txt").c_str());
    string line;
    while (getline(infile, line))
    {
        vector<string> elems;
        split(line, elems, ',');
        string syn = elems[0];
        int number = atoi(elems[1].c_str());

        labels.insert(make_pair(syn, number));
    }

    string pathTrain(path + "train/");
    vector<string> fileNames;
    getDirList(pathTrain, fileNames);
    for (vector<string>::iterator it=fileNames.begin(); it!=fileNames.end(); ++it)
    {
        string pathSyn((*it) + "/");
        vector<string> fileNamesSyn;
        getDirList((pathTrain + pathSyn), fileNamesSyn);
        for (vector<string>::iterator itSyn=fileNamesSyn.begin(); itSyn!=fileNamesSyn.end(); ++itSyn)
        {
            Ptr<OR_imagenetObj> curr(new OR_imagenetObj);
            curr->image = "train/" + pathSyn + *itSyn;
            curr->id = labels[*it];

            train.back().push_back(curr);
        }
    }

    ifstream infileVal((path + "ILSVRC2010_validation_ground_truth.txt").c_str());
    while (getline(infileVal, line))
    {
        Ptr<OR_imagenetObj> curr(new OR_imagenetObj);
        curr->id = atoi(line.c_str());
        numberToString((int)validation.back().size()+1, curr->image);
        curr->image = "val/ILSVRC2010_val_" + curr->image + ".JPEG";

        validation.back().push_back(curr);
    }

    vector<int> testGT;
    ifstream infileTest((path + "ILSVRC2010_test_ground_truth.txt").c_str());
    while (getline(infileTest, line))
    {
        testGT.push_back(atoi(line.c_str()));
    }
    if (testGT.size()==0) // have no test labels, set them to 1000 - unknown
    {
        for (int i=0; i<150000; ++i)
        {
            testGT.push_back(1000); // unknown
        }
    }

    for (vector<int>::iterator it=testGT.begin(); it!=testGT.end(); ++it)
    {
        Ptr<OR_imagenetObj> curr(new OR_imagenetObj);
        curr->id = *it;
        numberToString((int)test.back().size()+1, curr->image);
        curr->image = "test/ILSVRC2010_test_" + curr->image + ".JPEG";

        test.back().push_back(curr);
    }
}

Ptr<OR_imagenet> OR_imagenet::create()
{
    return Ptr<OR_imagenetImp>(new OR_imagenetImp);
}

}
}
