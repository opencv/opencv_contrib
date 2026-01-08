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

#include "opencv2/datasets/fr_lfw.hpp"
#include "opencv2/datasets/util.hpp"

#include <map>

namespace cv
{
namespace datasets
{

using namespace std;

class FR_lfwImp CV_FINAL : public FR_lfw
{
public:
    FR_lfwImp() {}
    //FR_lfwImp(const string &path);
    virtual ~FR_lfwImp() {}

    virtual void load(const string &path) CV_OVERRIDE;

private:
    void loadDataset(const string &path);

    map< string, vector<string> > faces;
};

/*FR_lfwImp::FR_lfwImp(const string &path)
{
    loadDataset(path);
}*/

void FR_lfwImp::load(const string &path)
{
    loadDataset(path);
}

void FR_lfwImp::loadDataset(const string &path)
{
    vector<string> fileNames;
    getDirList(path, fileNames);
    for (vector<string>::iterator it=fileNames.begin(); it!=fileNames.end(); ++it)
    {
        string &name = *it;

        if (name.length()>3 && name.substr(name.length()-4) == ".txt")
        {
            continue;
        }

        vector<string> images;

        string pathFace(path + name + "/");
        vector<string> faceNames;
        getDirList(pathFace, faceNames);
        for (vector<string>::iterator itFace=faceNames.begin(); itFace!=faceNames.end(); ++itFace)
        {
            images.push_back(*itFace);
        }

        faces.insert(make_pair(name, images));
    }

    // test loading
    ifstream infile((path + "pairs.txt").c_str());
    string line;
    getline(infile, line); // should be 10 300
    CV_Assert(line=="10\t300");
    unsigned int num = 0;
    while (getline(infile, line))
    {
        if (0 == (num % 600))
        {
            train.push_back(vector< Ptr<Object> >());
            test.push_back(vector< Ptr<Object> >());
        }

        vector<string> elems;
        split(line, elems, '\t');

        Ptr<FR_lfwObj> curr(new FR_lfwObj);
        string &person1 = elems[0];
        unsigned int imageNumber1 = atoi(elems[1].c_str())-1;
        curr->image1 = person1 + "/" + faces[person1][imageNumber1];

        string person2;
        unsigned int imageNumber2;
        if (3 == elems.size())
        {
            person2 = elems[0];
            imageNumber2 = atoi(elems[2].c_str())-1;
            curr->same = true;
        } else
        {
            person2 = elems[2];
            imageNumber2 = atoi(elems[3].c_str())-1;
            curr->same = false;
        }
        curr->image2 = person2 + "/" + faces[person2][imageNumber2];

        test.back().push_back(curr);

        num++;
    }
    infile.close();

    // dev train loading to train[0]
    ifstream infile2((path + "pairsDevTrain.txt").c_str());
    getline(infile2, line); // should 1100
    CV_Assert(line=="1100");
    while (getline(infile2, line))
    {
        vector<string> elems;
        split(line, elems, '\t');

        Ptr<FR_lfwObj> curr(new FR_lfwObj);
        string &person1 = elems[0];
        unsigned int imageNumber1 = atoi(elems[1].c_str())-1;
        curr->image1 = person1 + "/" + faces[person1][imageNumber1];

        string person2;
        unsigned int imageNumber2;
        if (3 == elems.size())
        {
            person2 = elems[0];
            imageNumber2 = atoi(elems[2].c_str())-1;
            curr->same = true;
        } else
        {
            person2 = elems[2];
            imageNumber2 = atoi(elems[3].c_str())-1;
            curr->same = false;
        }
        curr->image2 = person2 + "/" + faces[person2][imageNumber2];

        train[0].push_back(curr);
    }
    infile2.close();

    // dev train loading to validation[0]
    ifstream infile3((path + "pairsDevTest.txt").c_str());
    getline(infile3, line); // should 500
    CV_Assert(line=="500");
    validation.push_back(vector< Ptr<Object> >());
    while (getline(infile3, line))
    {
        vector<string> elems;
        split(line, elems, '\t');

        Ptr<FR_lfwObj> curr(new FR_lfwObj);
        string &person1 = elems[0];
        unsigned int imageNumber1 = atoi(elems[1].c_str())-1;
        curr->image1 = person1 + "/" + faces[person1][imageNumber1];

        string person2;
        unsigned int imageNumber2;
        if (3 == elems.size())
        {
            person2 = elems[0];
            imageNumber2 = atoi(elems[2].c_str())-1;
            curr->same = true;
        } else
        {
            person2 = elems[2];
            imageNumber2 = atoi(elems[3].c_str())-1;
            curr->same = false;
        }
        curr->image2 = person2 + "/" + faces[person2][imageNumber2];

        validation[0].push_back(curr);
    }
    infile3.close();
}

Ptr<FR_lfw> FR_lfw::create()
{
    return Ptr<FR_lfwImp>(new FR_lfwImp);
}

}
}
