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

#include "opencv2/datasets/fr_adience.hpp"
#include "opencv2/datasets/util.hpp"

#include <map>
#include <set>

namespace cv
{
namespace datasets
{

using namespace std;

class FR_adienceImp CV_FINAL : public FR_adience
{
public:
    FR_adienceImp() {}
    //FR_adienceImp(const string &path);
    virtual ~FR_adienceImp() {}

    virtual void load(const string &path) CV_OVERRIDE;

private:
    void loadDataset(const string &path);

    void loadFile(const string &filename, vector< Ptr<FR_adienceObj> > &out);
    void cv5ToSplits(vector< Ptr<FR_adienceObj> > fileList[5]);

    map< string, vector<string> > realNames;
    set<string> missing;
};

/*FR_adienceImp::FR_adienceImp(const string &path)
{
    loadDataset(path);
}*/

void FR_adienceImp::load(const string &path)
{
    loadDataset(path);
}

void FR_adienceImp::loadFile(const string &filename, vector< Ptr<FR_adienceObj> > &out)
{
    string line;
    ifstream infile(filename.c_str());
    getline(infile, line); // skip header
    while (getline(infile, line))
    {
        vector<string> elems;
        split(line, elems, ',');

        string user_id = elems[0];
        string original_image = elems[1];

        // convert original_image to real image name
        bool isChanged = false;
        vector<string> &currImgs = realNames[user_id];
        for (vector<string>::iterator it=currImgs.begin(); it!=currImgs.end(); ++it)
        {
            string &name = *it;
            size_t origImgLen = original_image.length();
            if (name.length()>origImgLen && name.substr(name.length()-origImgLen) == original_image)
            {
                original_image = name;
                isChanged = true;
                break;
            }
        }
        if (!isChanged)
        {
            missing.insert(user_id+"/"+original_image);
            continue;
        }

        Ptr<FR_adienceObj> curr(new FR_adienceObj);
        curr->user_id = user_id;
        curr->original_image = original_image;
        curr->face_id = atoi(elems[2].c_str());
        curr->age = elems[3];
        if (elems[4]=="m")
        {
            curr->gender = male;
        } else
        if (elems[4]=="f")
        {
            curr->gender = female;
        } else
        {
            curr->gender = none;
        }
        curr->x = atoi(elems[5].c_str());
        curr->y = atoi(elems[6].c_str());
        curr->dx = atoi(elems[7].c_str());
        curr->dy = atoi(elems[8].c_str());
        curr->tilt_ang = atoi(elems[9].c_str());
        curr->fiducial_yaw_angle = atoi(elems[10].c_str());
        curr->fiducial_score = atoi(elems[11].c_str());

        out.push_back(curr);
    }
}

void FR_adienceImp::cv5ToSplits(vector< Ptr<FR_adienceObj> > fileList[5])
{
    for (unsigned int i=0; i<5; ++i)
    {
        train.push_back(vector< Ptr<Object> >());
        test.push_back(vector< Ptr<Object> >());
        validation.push_back(vector< Ptr<Object> >());
        for (unsigned int j=0; j<5; ++j)
        {
            vector< Ptr<FR_adienceObj> > &currlist = fileList[j];
            if (i!=j)
            {
                for (vector< Ptr<FR_adienceObj> >::iterator it=currlist.begin(); it!=currlist.end(); ++it)
                {
                    train.back().push_back(*it);
                }
            } else
            {
                for (vector< Ptr<FR_adienceObj> >::iterator it=currlist.begin(); it!=currlist.end(); ++it)
                {
                    test.back().push_back(*it);
                }
            }
        }
    }
}

void FR_adienceImp::loadDataset(const string &path)
{
    // collect real image names
    unsigned int num = 0;
    vector<string> userNames;
    getDirList(path+"faces/", userNames);
    for (vector<string>::iterator itU=userNames.begin(); itU!=userNames.end(); ++itU)
    {
        vector<string> fileNames;
        getDirList(path+"faces/"+*itU+"/", fileNames);
        for (vector<string>::iterator it=fileNames.begin(); it!=fileNames.end(); ++it)
        {
            string &name = *it;
            if (name.length()>3 && name.substr(name.length()-4) == ".jpg")
            {
                realNames[*itU].push_back(name);
                num++;
            }
        }
    }
    //printf("total images number: %u\n", num);

    vector< Ptr<FR_adienceObj> > fileList[5];
    for (unsigned int i=0; i<5; ++i)
    {
        char tmp[3];
        sprintf(tmp, "%u", i);
        string filename(path+"fold_"+string(tmp)+"_data.txt");

        loadFile(filename, fileList[i]);
    }
    cv5ToSplits(fileList);

    for (unsigned int i=0; i<5; ++i)
    {
        char tmp[3];
        sprintf(tmp, "%u", i);
        string filename(path+"fold_frontal_"+string(tmp)+"_data.txt");

        fileList[i].clear();
        loadFile(filename, fileList[i]);
    }
    cv5ToSplits(fileList);

    /*for (set<string>::iterator it=missing.begin(); it!=missing.end(); ++it)
    {
        printf("missing image: %s\n", (*it).c_str());
    }*/
    realNames.clear();
    missing.clear();
}

Ptr<FR_adience> FR_adience::create()
{
    return Ptr<FR_adienceImp>(new FR_adienceImp);
}

}
}
