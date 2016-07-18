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

#include "opencv2/datasets/msm_middlebury.hpp"
#include "opencv2/datasets/util.hpp"

namespace cv
{
namespace datasets
{

using namespace std;

class MSM_middleburyImp : public MSM_middlebury
{
public:
    MSM_middleburyImp() {}
    //MSM_middleburyImp(const string &path);
    virtual ~MSM_middleburyImp() {}

    virtual void load(const string &path);

private:
    void loadDataset(const string &path);
};

/*MSM_middleburyImp::MSM_middleburyImp(const string &path)
{
    loadDataset(path);
}*/

void MSM_middleburyImp::load(const string &path)
{
    loadDataset(path);
}

void MSM_middleburyImp::loadDataset(const string &path)
{
    train.push_back(vector< Ptr<Object> >());
    test.push_back(vector< Ptr<Object> >());
    validation.push_back(vector< Ptr<Object> >());

    string name(path.substr(0, path.length()-1));
    size_t start = name.rfind('/');
    name = name.substr(start+1, name.length()-start);

    string angName(path + name + "_ang.txt");
    string parName(path + name + "_par.txt");

    ifstream infile(parName.c_str());
    string imageName;
    infile >> imageName; // skip header
    while (infile >> imageName)
    {
        Ptr<MSM_middleburyObj> curr(new MSM_middleburyObj);
        curr->imageName = imageName;

        for (int i=0; i<3; ++i)
        {
            for (int j=0; j<3; ++j)
            {
                infile >> curr->k(i, j);
            }
        }
        for (int i=0; i<3; ++i)
        {
            for (int j=0; j<3; ++j)
            {
                infile >> curr->r(i, j);
            }
        }
        for (int i=0; i<3; ++i)
        {
            infile >> curr->t[i];
        }

        train.back().push_back(curr);
    }
}

Ptr<MSM_middlebury> MSM_middlebury::create()
{
    return Ptr<MSM_middleburyImp>(new MSM_middleburyImp);
}

}
}
