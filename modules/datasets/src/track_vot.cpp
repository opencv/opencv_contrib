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

#include "opencv2/datasets/track_vot.hpp"

#include <sys/stat.h>
#include <opencv2/core.hpp>
#include "opencv2/imgcodecs.hpp"

using namespace std;

namespace cv
{
    namespace datasets
    {

        class TRACK_votImpl CV_FINAL : public TRACK_vot
        {
        public:
            //Constructor
            TRACK_votImpl()
            {
                activeDatasetID = 1;
                frameCounter = 0;
            }
            //Destructor
            virtual ~TRACK_votImpl() CV_OVERRIDE {}

            //Load Dataset
            virtual void load(const string &path) CV_OVERRIDE;

		protected:
            virtual int getDatasetsNum() CV_OVERRIDE;

            virtual int getDatasetLength(int id) CV_OVERRIDE;

            virtual bool initDataset(int id) CV_OVERRIDE;

            virtual bool getNextFrame(Mat &frame) CV_OVERRIDE;

            virtual vector <Point2d> getGT() CV_OVERRIDE;

            void loadDataset(const string &path);

            string numberToString(int number);
        };

        void TRACK_votImpl::load(const string &path)
        {
            loadDataset(path);
        }

        string TRACK_votImpl::numberToString(int number)
        {
            string out;
            char numberStr[20];
            sprintf(numberStr, "%u", number);
            for (unsigned int i = 0; i < 8 - strlen(numberStr); ++i)
            {
                out += "0";
            }
            out += numberStr;
            return out;
        }

        inline bool fileExists(const std::string& name)
        {
            struct stat buffer;
            return (stat(name.c_str(), &buffer) == 0);
        }

        void TRACK_votImpl::loadDataset(const string &rootPath)
        {
            string nameListPath = rootPath + "/list.txt";
            ifstream namesList(nameListPath.c_str());
            vector <int> datasetsLengths;
            string datasetName;

            if (namesList.is_open())
            {
                int currDatasetID = 0;

                //All datasets/folders loop
                while (getline(namesList, datasetName))
                {
                    currDatasetID++;
                    vector <Ptr<TRACK_votObj> > objects;

                    //All frames/images loop
                    Ptr<TRACK_votObj> currDataset(new TRACK_votObj);

                    //Open dataset's ground truth file
                    string gtListPath = rootPath + "/" + datasetName + "/groundtruth.txt";
                    ifstream gtList(gtListPath.c_str());
                    if (!gtList.is_open())
                        printf("Error to open groundtruth.txt!!!");

                    //Make a list of datasets lengths
                    int currFrameID = 0;
                    if (currDatasetID == 0)
                        printf("VOT Dataset Initialization...\n");
                    bool trFLG = true;
                    do
                    {
                        currFrameID++;
                        string fullPath = rootPath + "/" + datasetName + "/" + numberToString(currFrameID) + ".jpg";
                        if (!fileExists(fullPath))
                            break;

                        //Make VOT Object
                        Ptr<TRACK_votObj> currObj(new TRACK_votObj);
                        currObj->imagePath = fullPath;
                        currObj->id = currFrameID;

                        //Get Ground Truth data
                        double	x1 = 0, y1 = 0,
                            x2 = 0, y2 = 0,
                            x3 = 0, y3 = 0,
                            x4 = 0, y4 = 0;
                        string tmp;
                        getline(gtList, tmp);
                        sscanf(tmp.c_str(), "%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf", &x1, &y1, &x2, &y2, &x3, &y3, &x4, &y4);
                        currObj->gtbb.push_back(Point2d(x1, y1));
                        currObj->gtbb.push_back(Point2d(x2, y2));
                        currObj->gtbb.push_back(Point2d(x3, y3));
                        currObj->gtbb.push_back(Point2d(x4, y4));

                        //Add object to storage
                        objects.push_back(currObj);

                    } while (trFLG);

                    datasetsLengths.push_back(currFrameID - 1);
                    data.push_back(objects);
                }
            }
            else
            {
                printf("Couldn't find a *list.txt* in VOT Dataset folder!!!");
            }

            namesList.close();
            return;
        }

        int TRACK_votImpl::getDatasetsNum()
        {
            return (int)(data.size());
        }

        int TRACK_votImpl::getDatasetLength(int id)
        {
            if (id > 0 && id <= (int)data.size())
                return (int)(data[id - 1].size());
            else
            {
                printf("Dataset ID is out of range...\nAllowed IDs are: 1~%d\n", (int)data.size());
                return -1;
            }
        }

        bool TRACK_votImpl::initDataset(int id)
        {
            if (id > 0 && id <= (int)data.size())
            {
                activeDatasetID = id;
                frameCounter = 0;
                return true;
            }
            else
            {
                printf("Dataset ID is out of range...\nAllowed IDs are: 1~%d\n", (int)data.size());
                return false;
            }
        }

        bool  TRACK_votImpl::getNextFrame(Mat &frame)
        {
            if (frameCounter >= (int)data[activeDatasetID - 1].size())
                return false;
            string imgPath = data[activeDatasetID - 1][frameCounter]->imagePath;
            frame = imread(imgPath);
            frameCounter++;
            return !frame.empty();
        }

        Ptr<TRACK_vot> TRACK_vot::create()
        {
            return Ptr<TRACK_votImpl>(new TRACK_votImpl);
        }

        vector <Point2d> TRACK_votImpl::getGT()
        {
            Ptr <TRACK_votObj> currObj = data[activeDatasetID - 1][frameCounter - 1];
            return currObj->gtbb;
        }

    }
}
