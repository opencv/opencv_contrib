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
#include "opencv2/datasets/util.hpp"

#include <map>

namespace cv
{
namespace datasets
{

using namespace std;

class OR_sunImp CV_FINAL : public OR_sun
{
public:
    OR_sunImp() {}
    //OR_sunImp(const string &path);
    virtual ~OR_sunImp() CV_OVERRIDE {}

    virtual void load(const string &path) CV_OVERRIDE;

private:
    void loadDataset(const string &path);

    void loadDatasetPart(const string &path, vector< Ptr<Object> > &dataset_);

    map<string, int> pathLabel;
};

/*OR_sunImp::OR_sunImp(const string &path)
{
    loadDataset(path);
}*/

void OR_sunImp::load(const string &path)
{
    loadDataset(path);
}

void OR_sunImp::loadDatasetPart(const string &path, vector< Ptr<Object> > &dataset_)
{
    string line;
    ifstream infile(path.c_str());
    while (getline(infile, line))
    {
        Ptr<OR_sunObj> curr(new OR_sunObj);
        curr->label = 397;
        curr->name = line;

        size_t pos = curr->name.rfind('/');
        if (pos != string::npos)
        {
            string labelStr(curr->name.substr(0, pos+1));
            map<string, int>::iterator it = pathLabel.find(labelStr);
            if (it != pathLabel.end())
            {
                curr->label = (*it).second;
            } else
            {
                curr->label = (int)pathLabel.size();
                pathLabel.insert(make_pair(labelStr, curr->label));
                paths.push_back(labelStr);
            }
            curr->name = curr->name.substr(pos+1);
        }

        dataset_.push_back(curr);
    }
}

void OR_sunImp::loadDataset(const string &path)
{
    /*string classNameFile(path + "ClassName.txt");
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
    }*/

    for (unsigned int i=1; i<=10; ++i)
    {
        char tmp[3];
        sprintf(tmp, "%u", i);
        string numStr;
        if (i<10)
        {
            numStr = string("0") + string(tmp);
        } else
        {
            numStr = tmp;
        }
        string trainFile(path + "Partitions/Training_" + numStr + ".txt");
        string testFile(path + "Partitions/Testing_" + numStr + ".txt");

        train.push_back(vector< Ptr<Object> >());
        test.push_back(vector< Ptr<Object> >());
        validation.push_back(vector< Ptr<Object> >());

        loadDatasetPart(trainFile, train.back());
        loadDatasetPart(testFile, test.back());
    }
}

Ptr<OR_sun> OR_sun::create()
{
    return Ptr<OR_sunImp>(new OR_sunImp);
}

}
}
