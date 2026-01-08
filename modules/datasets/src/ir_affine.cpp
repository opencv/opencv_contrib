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

#include "opencv2/datasets/ir_affine.hpp"
#include "opencv2/datasets/util.hpp"

namespace cv
{
namespace datasets
{

using namespace std;

class IR_affineImp CV_FINAL : public IR_affine
{
public:
    IR_affineImp() {}
    //IR_affineImp(const string &path);
    virtual ~IR_affineImp() CV_OVERRIDE {}

    virtual void load(const string &path) CV_OVERRIDE;

private:
    void loadDataset(const string &path);
};

/*IR_affineImp::IR_affineImp(const string &path)
{
    loadDataset(path);
}*/

void IR_affineImp::load(const string &path)
{
    loadDataset(path);
}

void IR_affineImp::loadDataset(const string &path)
{
    train.push_back(vector< Ptr<Object> >());
    test.push_back(vector< Ptr<Object> >());
    validation.push_back(vector< Ptr<Object> >());

    // detect image extension
    string ext;
    vector<string> fileNames;
    getDirList(path, fileNames);
    for (vector<string>::iterator it=fileNames.begin(); it!=fileNames.end(); ++it)
    {
        string &name = *it;
        if (name.length()>=8 && name.substr(0, 3)=="img")
        {
            ext = name.substr(name.length()-4, 4);
            break;
        }
    }

    for (unsigned int i=1; i<=6; ++i)
    {
        Ptr<IR_affineObj> curr(new IR_affineObj);

        char tmp[2];
        sprintf(tmp, "%u", i);
        curr->imageName = path + "img" + tmp + ext;

        if (i>1)
        {
            string matName(path + "H1to" + tmp + "p");
            ifstream infile(matName.c_str());
            for (int k=0; k<3; ++k)
            {
                for (int j=0; j<3; ++j)
                {
                    infile >> curr->mat(k, j);
                }
            }
        }

        train.back().push_back(curr);
    }
}

Ptr<IR_affine> IR_affine::create()
{
    return Ptr<IR_affineImp>(new IR_affineImp);
}

}
}
