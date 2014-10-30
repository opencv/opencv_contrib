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

#include "opencv2/datasets/hpe_humaneva.hpp"
#include "opencv2/datasets/util.hpp"

#include <cstring>

namespace cv
{
namespace datasets
{

using namespace std;

class HPE_humanevaImp : public HPE_humaneva
{
public:
    HPE_humanevaImp() {}
    //HPE_humanevaImp(const string &path);
    virtual ~HPE_humanevaImp() {}

    virtual void load(const string &path);

private:
    void loadDataset(const string &path);
};

/*HPE_humanevaImp::HPE_humanevaImp(const string &path)
{
    loadDataset(path);
}*/

void HPE_humanevaImp::load(const string &path)
{
    loadDataset(path);
}

void HPE_humanevaImp::loadDataset(const string &path)
{
    train.push_back(vector< Ptr<Object> >());
    test.push_back(vector< Ptr<Object> >());
    validation.push_back(vector< Ptr<Object> >());

    for (unsigned int i=1; i<=4; ++i)
    {
        char number[2];
        sprintf(number, "%u", i);
        string pathDatasetI(path + "S" + number + "/Image_Data/");
        string pathDatasetS(path + "S" + number + "/Sync_Data/");

        vector<string> fileNames;
        getDirList(pathDatasetI, fileNames);
        for (vector<string>::iterator it=fileNames.begin(); it!=fileNames.end(); ++it)
        {
            string &file = *it;

            vector<string> elems;
            split(file, elems, '_');
            if (elems.size() != 3)
            {
                continue;
            }

            Ptr<HPE_humanevaObj> curr(new HPE_humanevaObj);
            curr->person = (char)i;
            curr->action = elems[0];
            curr->type1 = atoi(elems[1].c_str());
            curr->fileName = pathDatasetI+file;

            unsigned int type2End = 2;
            if (elems[2][type2End+1] != ')')
            {
                type2End = 3;
            }
            curr->type2 = elems[2].substr(1, type2End);

            file = file.substr(0, file.length()-3) + "ofs";
            ifstream infileOFS((pathDatasetS + file).c_str());
            string line;
            unsigned int j = 0;
            while (getline(infileOFS, line))
            {
                curr->ofs(0, j) = atof(line.c_str());
                ++j;
            }

            train.back().push_back(curr);
        }
    }
}

//
// HumanEva II
//
class HPE_humanevaImpII : public HPE_humaneva
{
public:
    HPE_humanevaImpII() {}
    //HPE_humanevaImpII(const string &path);
    virtual ~HPE_humanevaImpII() {}

    virtual void load(const string &path);

private:
    void loadDataset(const string &path);
};

/*HPE_humanevaImpII::HPE_humanevaImpII(const string &path)
{
    loadDataset(path);
}*/

void HPE_humanevaImpII::load(const string &path)
{
    loadDataset(path);
}

void HPE_humanevaImpII::loadDataset(const string &path)
{
    train.push_back(vector< Ptr<Object> >());
    test.push_back(vector< Ptr<Object> >());
    validation.push_back(vector< Ptr<Object> >());

    for (unsigned int i=1; i<=2; ++i)
    {
        char number[2];
        sprintf(number, "%u", i*2); // 2 & 4
        string pathDatasetI(path + "S" + number + "/Image_Data/");
        string pathDatasetS(path + "S" + number + "/Sync_Data/");

        vector<string> fileNames;
        getDirList(pathDatasetI, fileNames);
        for (vector<string>::iterator it=fileNames.begin(); it!=fileNames.end(); ++it)
        {
            string &file = *it;

            vector<string> elems;
            split(file, elems, '_');
            if (elems.size() != 3)
            {
                continue;
            }

            Ptr<HPE_humanevaObj> curr(new HPE_humanevaObj);
            curr->person = (char)i;
            curr->action = elems[0];
            curr->type1 = atoi(elems[1].c_str());
            curr->fileName = pathDatasetI+file;

            unsigned int type2End = 2;
            if (elems[2][type2End+1] != ')')
            {
                type2End = 3;
            }
            curr->type2 = elems[2].substr(1, type2End);

            vector<string> imageNames;
            getDirList(curr->fileName, imageNames);
            for (vector<string>::iterator itI=imageNames.begin(); itI!=imageNames.end(); ++itI)
            {
                string &image = *itI;
                if (image.substr(image.length()-3) == "png")
                {
                    curr->imageNames.push_back(image);
                }
            }

            file = file.substr(0, file.length()) + ".ofs";
            ifstream infileOFS((pathDatasetS + file).c_str());
            string line;
            unsigned int j = 0;
            while (getline(infileOFS, line))
            {
                curr->ofs(0, j) = atof(line.c_str());
                ++j;
            }

            train.back().push_back(curr);
        }
    }
}

Ptr<HPE_humaneva> HPE_humaneva::create(int num)
{
    if (humaneva_2==num)
    {
        return Ptr<HPE_humanevaImpII>(new HPE_humanevaImpII);
    }
    return Ptr<HPE_humanevaImp>(new HPE_humanevaImp);
}

}
}
