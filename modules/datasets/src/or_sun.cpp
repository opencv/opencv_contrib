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

#include "opencv2/datasets/or_sun.hpp"
#include "precomp.hpp"

namespace cv
{
namespace datasets
{

using namespace std;

class CV_EXPORTS OR_sunImp : public OR_sun
{
public:
    OR_sunImp() {}
    //OR_sunImp(const string &path);
    virtual ~OR_sunImp() {}

    virtual void load(const string &path);

private:
    void loadDataset(const string &path);
};

/*OR_sunImp::OR_sunImp(const string &path)
{
    loadDataset(path);
}*/

void OR_sunImp::load(const string &path)
{
    loadDataset(path);
}

void OR_sunImp::loadDataset(const string &path)
{
    train.push_back(vector< Ptr<Object> >());
    test.push_back(vector< Ptr<Object> >());
    validation.push_back(vector< Ptr<Object> >());

    string classNameFile(path + "ClassName.txt");
    ifstream infile(classNameFile.c_str());
    string line;
    while (getline(infile, line))
    {
        Ptr<OR_sunObj> curr(new OR_sunObj);
        curr->name = line;

        string currPath(path + curr->name);
        vector<string> fileNames;
        getDirList(currPath, fileNames);
        for (vector<string>::iterator it=fileNames.begin(); it!=fileNames.end(); ++it)
        {
            curr->imageNames.push_back(*it);
        }

        train.back().push_back(curr);
    }
}

Ptr<OR_sun> OR_sun::create()
{
    return Ptr<OR_sunImp>(new OR_sunImp);
}

}
}
