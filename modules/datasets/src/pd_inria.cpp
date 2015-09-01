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
// Copyright (C) 2015, Itseez Inc, all rights reserved.
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

#include "opencv2/datasets/pd_inria.hpp"
#include "opencv2/datasets/util.hpp"

namespace cv
{
namespace datasets
{

using namespace std;

class PD_inriaImp : public PD_inria
{
public:
    PD_inriaImp() {}
    
    virtual ~PD_inriaImp() {}

    virtual void load(const string &path);

private:
    void loadDataset(const string &path, const string nameImageSet, vector< Ptr<Object> > &imageSet);
    void readTextLines(const string filename, vector< string > &lines);
    void parseAnnotation(const string filename, Ptr< PD_inriaObj > &object);
};

void PD_inriaImp::load(const string &path)
{
    // Training set
    train.push_back(vector< Ptr<Object> >());
    loadDataset(path, "Train", train.back());
    
    // Testing set
    test.push_back(vector< Ptr<Object> >());
    loadDataset(path, "Test", test.back());

    // There is no validation set
    validation.push_back(vector< Ptr<Object> >());
}

void PD_inriaImp::readTextLines(const string filename, vector< string > &lines)
{
    ifstream in(filename.c_str());
    string error_message = "";

    if (!in.is_open())
    {
        error_message = format("Unable to open file: \n%s\n", filename.c_str());
        CV_Error(Error::StsBadArg, error_message);
    }

    string currline = "";

    while (getline(in, currline))
        lines.push_back(currline);
}

void PD_inriaImp::parseAnnotation(const string filename, Ptr< PD_inriaObj > &object)
{
    string error_message = "";

    ifstream in(filename.c_str());

    if (!in.is_open())
    {
        error_message = format("Unable to open file: \n%s\n", filename.c_str());
        CV_Error(Error::StsBadArg, error_message);
    }
    
    string imageSizeHeader = "Image size (X x Y x C) : ";
    string imageSizeFmt = imageSizeHeader + "%d x %d x %d";
    string objWithGTHeader = "Objects with ground truth : ";
    string objWithGTFmt = objWithGTHeader + "%d { \"PASperson\" }";
    string boundBoxHeader = "Bounding box for object ";
    string boundBoxFmt = boundBoxHeader + "%*d \"PASperson\" (Xmin, Ymin) - (Xmax, Ymax) : (%d, %d) - (%d, %d)";
    
    string line = "";
    
    int width = 0;
    int height = 0;
    int depth = 0;
    int xmin, ymin, xmax, ymax;

    int numObjects = 0;

    while (getline(in, line))
    {
        if (line[0] == '#' || !line[0])
            continue;

        if (strstr(line.c_str(), imageSizeHeader.c_str()))
        {
            sscanf(line.c_str(), imageSizeFmt.c_str(), &width, &height, &depth);
            object->width = width;
            object->height = height;
            object->depth = depth;
        }
        else if (strstr(line.c_str(), objWithGTHeader.c_str()))
        {
            sscanf(line.c_str(), objWithGTFmt.c_str(), &numObjects);
            
            if (numObjects <= 0)
                break;
        }
        else if (strstr(line.c_str(), boundBoxHeader.c_str()))
        {
            sscanf(line.c_str(), boundBoxFmt.c_str(), &xmin, &ymin, &xmax, &ymax);
            Rect bndbox;
            bndbox.x = xmin;
            bndbox.y = ymin;
            bndbox.width = xmax - xmin;
            bndbox.height = ymax - ymin;
            (object->bndboxes).push_back(bndbox);
        }
    }

    CV_Assert((object->bndboxes).size() == (unsigned int)numObjects);
}

void PD_inriaImp::loadDataset(const string &path, const string nameImageSet, vector< Ptr<Object> > &imageSet)
{
    string listAnn = path + nameImageSet + "/annotations.lst";
    string listPos = path + nameImageSet + "/pos.lst";
    string listNeg = path + nameImageSet + "/neg.lst";
    
    vector< string > fsAnn;
    vector< string > fsPos;
    vector< string > fsNeg;
   
    // read file names
    readTextLines(listAnn, fsAnn);
    readTextLines(listPos, fsPos);
    readTextLines(listNeg, fsNeg);
   
    CV_Assert(fsAnn.size() == fsPos.size());

    for (unsigned int i = 0; i < fsPos.size(); i++)
    {
        Ptr<PD_inriaObj> curr(new PD_inriaObj);
        parseAnnotation(path + fsAnn[i], curr);
        curr->filename = path + fsPos[i];
        curr->sType = POS;
        
        imageSet.push_back(curr);
    }
    
    for (unsigned int i = 0; i < fsNeg.size(); i++)
    {
        Ptr<PD_inriaObj> curr(new PD_inriaObj);
        curr->filename = path + fsNeg[i]; 
        curr->sType = NEG;
        
        imageSet.push_back(curr);
    }
}

Ptr<PD_inria> PD_inria::create()
{
    return Ptr<PD_inriaImp>(new PD_inriaImp);
}

}
}
