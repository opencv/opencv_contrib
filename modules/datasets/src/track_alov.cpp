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

#include "opencv2/datasets/track_alov.hpp"

#include <sys/stat.h>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

using namespace std;

namespace cv
{
namespace datasets
{

class TRACK_alovImpl : public TRACK_alov
{
public:
    //Constructor
    TRACK_alovImpl()
    {
        activeDatasetID = 1;
        frameCounter = 0;
    }
    //Destructor
    virtual ~TRACK_alovImpl() {}

    //Load Dataset
    virtual void load(const string &path);
    virtual void loadAnnotatedOnly(const std::string &path);

protected:
    virtual int getDatasetsNum();

    virtual int getDatasetLength(int id);

    virtual bool initDataset(int id);

    virtual bool getNextFrame(Mat &frame);
    virtual bool getFrame(Mat &frame, int datasetID, int frameID);

    virtual vector <Point2f> getNextGT();
    virtual vector <Point2f> getGT(int datasetID, int frameID);

    void loadDataset(const string &path);
    void loadDatasetAnnotatedOnly(const string &path);

    string fullFramePath(string rootPath, int sectionID, int videoID, int frameID);
    string fullAnnoPath(string rootPath, int sectionID, int videoID);
};


void TRACK_alovImpl::load(const string &path)
{
    loadDataset(path);
}

void TRACK_alovImpl::loadAnnotatedOnly(const string &path)
{
    loadDatasetAnnotatedOnly(path);
}

string TRACK_alovImpl::fullFramePath(string rootPath, int sectionID, int videoID, int frameID)
{
    string out;
    char videoNum[9];
    sprintf(videoNum, "%u", videoID+1);
    char frameNum[9];
    sprintf(frameNum, "%u", frameID);
    out = rootPath + "/imagedata++/" + sectionNames[sectionID] + "/" + sectionNames[sectionID] + "_video";

    for (unsigned int i = 0; i < 5 - strlen(videoNum); ++i)
    {
        out += "0";
    }

    out += videoNum;
    out += "/";

    for (unsigned int i = 0; i < 8 - strlen(frameNum); ++i)
    {
        out += "0";
    }

    out += frameNum;
    out += ".jpg";
    return out;
}

string TRACK_alovImpl::fullAnnoPath(string rootPath, int sectionID, int videoID)
{
    string out;
    char videoNum[9];
    sprintf(videoNum, "%u", videoID+1);
    out = rootPath + "/alov300++_rectangleAnnotation_full/" + sectionNames[sectionID] + "/" + sectionNames[sectionID] + "_video";

    for (unsigned int i = 0; i < 5 - strlen(videoNum); ++i)
    {
        out += "0";
    }

    out += videoNum;
    out += ".ann";
    return out;
}

inline bool fileExists(const std::string& name)
{
    struct stat buffer;
    return (stat(name.c_str(), &buffer) == 0);
}

void TRACK_alovImpl::loadDataset(const string &rootPath)
{
    vector <int> datasetsLengths;

    printf("ALOV300++ Dataset Initialization...\n");

    //Load frames
    //Loop for all sections of ALOV300++ (14 sections)
    for (int i = 0; i < 14; i++)
    {
        //Loop for all videos in section
        for (int k = 0; k < sectionSizes[i]; k++)
        {
            vector <Ptr<TRACK_alovObj> > objects;

            //Make a list of datasets lengths
            int currFrameID = 0;

            for (;;)
            {
                currFrameID++;
                string fullPath = fullFramePath(rootPath, i, k, currFrameID);
                if (!fileExists(fullPath))
                    break;

                //Make ALOV300++ Object
                Ptr<TRACK_alovObj> currObj(new TRACK_alovObj);
                currObj->imagePath = fullPath;
                currObj->id = currFrameID;

                currObj->gtbb.push_back(Point2d(0, 0));
                currObj->gtbb.push_back(Point2d(0, 0));
                currObj->gtbb.push_back(Point2d(0, 0));
                currObj->gtbb.push_back(Point2d(0, 0));

                //Add object to storage
                objects.push_back(currObj);

            }

            datasetsLengths.push_back(currFrameID - 1);
            data.push_back(objects);
        }
    }

    //Load annotations
    //Loop for all sections of ALOV300++ (14 sections)
    int currDatasetID = 0;
    for (int i = 0; i < 14; i++)
    {
        //Loop for all videos in section
        for (int k = 0; k < sectionSizes[i]; k++)
        {
            currDatasetID++;

            //Open dataset's ground truth (annotation) file
            string annoPath = fullAnnoPath(rootPath, i, k);
            ifstream annoList(annoPath.c_str());
            if (!annoList.is_open())
            {
                printf("Error: Can't open annotation file *.ANN!!!\n");
                break;
            }

            //Ground Truth data
            int n = 0;
            double    x1 = 0, y1 = 0,
                x2 = 0, y2 = 0,
                x3 = 0, y3 = 0,
                x4 = 0, y4 = 0;

            do
            {
                //Make ALOV300++ Object
                string tmp;

                getline(annoList, tmp);
                std::istringstream in(tmp);
                in >> n >> x1 >> y1 >> x2 >> y2 >> x3 >> y3 >> x4 >> y4;

                Ptr<TRACK_alovObj> currObj = data[currDatasetID-1][n-1];

                currObj->gtbb.clear();
                currObj->gtbb.push_back(Point2d(x1, y1));
                currObj->gtbb.push_back(Point2d(x2, y2));
                currObj->gtbb.push_back(Point2d(x3, y3));
                currObj->gtbb.push_back(Point2d(x4, y4));

            } while (annoList.good());
        }
    }

    return;
}

void TRACK_alovImpl::loadDatasetAnnotatedOnly(const string &rootPath)
{
    vector <int> datasetsLengths;
    int currDatasetID = 0;

    printf("ALOV300++ Annotated Dataset Initialization...\n");

    //Loop for all sections of ALOV300++ (14 sections)
    for (int i = 0; i < 14; i++)
    {
        //Loop for all videos in section
        for (int k = 0; k < sectionSizes[i]; k++)
        {
            vector <Ptr<TRACK_alovObj> > objects;
            currDatasetID++;

            //Open dataset's ground truth (annotation) file
            string annoPath = fullAnnoPath(rootPath, i, k);
            ifstream annoList(annoPath.c_str());
            if (!annoList.is_open())
            {
                printf("Error: Can't open annotation file *.ANN!!!\n");
                break;
            }

            int framesNum = 0;

            do
            {
                //Make  ALOV300++ Object
                Ptr<TRACK_alovObj> currObj(new TRACK_alovObj);
                string tmp;
                framesNum++;

                //Ground Truth data
                int    n = 0;
                double x1 = 0, y1 = 0,
                    x2 = 0, y2 = 0,
                    x3 = 0, y3 = 0,
                    x4 = 0, y4 = 0;

                getline(annoList, tmp);
                std::istringstream in(tmp);
                in >> n >> x1 >> y1 >> x2 >> y2 >> x3 >> y3 >> x4 >> y4;

                currObj->gtbb.push_back(Point2d(x1, y1));
                currObj->gtbb.push_back(Point2d(x2, y2));
                currObj->gtbb.push_back(Point2d(x3, y3));
                currObj->gtbb.push_back(Point2d(x4, y4));

                string fullPath = fullFramePath(rootPath, i, k, n);
                if (!fileExists(fullPath))
                    break;

                currObj->imagePath = fullPath;
                currObj->id = n;

                //Add object to storage
                objects.push_back(currObj);

            } while (annoList.good());

            datasetsLengths.push_back(framesNum-1);
            data.push_back(objects);
        }
    }

    return;
}

int TRACK_alovImpl::getDatasetsNum()
{
    return (int)(data.size());
}

int TRACK_alovImpl::getDatasetLength(int id)
{
    if (id > 0 && id <= (int)data.size())
        return (int)(data[id - 1].size());
    else
    {
        printf("Dataset ID is out of range...\nAllowed IDs are: 1~%d\n", (int)data.size());
        return -1;
    }
}

bool TRACK_alovImpl::initDataset(int id)
{
    if (id > 0 && id <= (int)data.size())
    {
        activeDatasetID = id;
        return true;
    }
    else
    {
        printf("Dataset ID is out of range...\nAllowed IDs are: 1~%d\n", (int)data.size());
        return false;
    }
}

bool  TRACK_alovImpl::getNextFrame(Mat &frame)
{
    if (frameCounter >= (int)data[activeDatasetID - 1].size())
        return false;
    string imgPath = data[activeDatasetID - 1][frameCounter]->imagePath;
    frame = imread(imgPath);
    frameCounter++;
    return !frame.empty();
}

bool  TRACK_alovImpl::getFrame(Mat &frame, int datasetID, int frameID)
{
    if (frameID > (int)data[datasetID-1].size())
        return false;
    string imgPath = data[datasetID-1][frameID-1]->imagePath;
    frame = imread(imgPath);
    return !frame.empty();
}

Ptr<TRACK_alov> TRACK_alov::create()
{
    return Ptr<TRACK_alovImpl>(new TRACK_alovImpl);
}

vector <Point2f> TRACK_alovImpl::getNextGT()
{
    Ptr <TRACK_alovObj> currObj = data[activeDatasetID - 1][frameCounter - 1];
    return currObj->gtbb;
}

vector <Point2f> TRACK_alovImpl::getGT(int datasetID, int frameID)
{
    Ptr <TRACK_alovObj> currObj = data[datasetID - 1][frameID - 1];
    return currObj->gtbb;
}

}
}
