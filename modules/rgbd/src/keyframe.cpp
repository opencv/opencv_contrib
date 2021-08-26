// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "precomp.hpp"
#include "keyframe.hpp"

namespace cv
{
namespace large_kinfu
{

KeyFrame::KeyFrame():submapID(-1), preKeyFrameID(-1)
{
    nextKeyFrameID = -1;
}

KeyFrame::KeyFrame(Mat _DNNfeature, int _submapID) : DNNFeature(_DNNfeature), submapID(_submapID), preKeyFrameID(-1)
{
    nextKeyFrameID = -1;
};

KeyFrame::KeyFrame(Mat _DNNfeature, int _submapID, int _preKeyFrameID):DNNFeature(_DNNfeature), submapID(_submapID), preKeyFrameID(_preKeyFrameID)
{}

KeyFrame::KeyFrame(Mat _DNNfeature, int _submapID, int _preKeyFrameID, std::vector<KeyPoint> _keypoints, Mat _ORBFeatures):DNNFeature(_DNNfeature), submapID(_submapID), preKeyFrameID(_preKeyFrameID)
{
    keypoints = _keypoints;
    ORBFeatures = _ORBFeatures;
}

// Using INT_MAX by default.
KeyFrameDatabase::KeyFrameDatabase():maxSizeDB(INT_MAX), lastKeyFrameID(-1)
{
}

KeyFrameDatabase::KeyFrameDatabase(int _maxSizeDB):maxSizeDB(_maxSizeDB),lastKeyFrameID(-1)
{
}

void KeyFrameDatabase::addKeyFrame( const Mat& _DNNFeature, int _frameID, int _submapID)
{
    std::vector<KeyPoint> _keypoints;
    Mat _ORBFeatures;
    addKeyFrameT(_DNNFeature, _frameID, _submapID, _keypoints, _ORBFeatures);
}

void KeyFrameDatabase::addKeyFrame( const Mat& _DNNFeature, int _frameID, int _submapID, std::vector<KeyPoint>& _keypoints, const Mat& _ORBFeatures)
{
    addKeyFrameT(_DNNFeature, _frameID, _submapID, _keypoints, _ORBFeatures);
}

void KeyFrameDatabase::addKeyFrameT(const Mat& DNNFeature, int frameID, int submapID, std::vector<KeyPoint>& keypoints, const Mat& ORBFeatures)
{
    Ptr<KeyFrame> kf, preKF;
    preKF = getKeyFrameByID(lastKeyFrameID);

    // new start for KeyFrame in different submaps.
    if(preKF)
    {
        kf = makePtr<KeyFrame>(DNNFeature, submapID, lastKeyFrameID, keypoints, ORBFeatures);
        preKF->nextKeyFrameID = frameID;
    }
    else
    {
        kf = makePtr<KeyFrame>(DNNFeature, submapID, -1, keypoints, ORBFeatures);
    }

    // Adding new KF to DB
    DataBase[frameID] = kf;

    if(int(DataBase.size()) > maxSizeDB)
        shrinkDB();

    // Change the last
    lastKeyFrameID = frameID;
}

bool KeyFrameDatabase::deleteKeyFrameByID(int keyFrameID)
{
    auto keyFrame = DataBase.find(keyFrameID);

    if(keyFrame == DataBase.end())
    {
        return false;
    } else
    {
        // Remove nowKF, and link the perKF to nextKF.
        Ptr<KeyFrame> preKF, nextKF, nowKF;
        nowKF = keyFrame->second;
        preKF = getKeyFrameByID(nowKF->preKeyFrameID);
        nextKF = getKeyFrameByID(nowKF->nextKeyFrameID);

        if(preKF)
        {
            preKF->nextKeyFrameID = nowKF->nextKeyFrameID;
        }

        if(nextKF)
        {
            nextKF->preKeyFrameID = nowKF->preKeyFrameID;
        }

        DataBase.erase(keyFrame);
        return true;
    }
}

Ptr<KeyFrame> KeyFrameDatabase::getKeyFrameByID(int keyFrameID)
{
    if(keyFrameID < 0)
        return {};

    auto keyFrame = DataBase.find(keyFrameID);
    if(keyFrame == DataBase.end())
    {
        return {};
    } else
    {
        return keyFrame->second;
    }
}

int KeyFrameDatabase::getSize()
{
    return DataBase.size();
}

bool KeyFrameDatabase::empty()
{
    return DataBase.empty();
}

int KeyFrameDatabase::getLastKeyFrameID()
{
    return lastKeyFrameID;
}

double KeyFrameDatabase::score(InputArray feature1, InputArray feature2)
{
    Mat mat1, mat2;

    mat1 = feature1.getMat();
    mat2 = feature2.getMat();
    CV_Assert(mat1.size() == mat2.size());
    double out = mat2.dot(mat1);
    return out;
}

std::vector<int> KeyFrameDatabase::getCandidateKF(const Mat& currentFeature, const int currentSubmapID, const double& similarityLow, double& bestSimilarity, int& bestId )
{
    std::vector<int> cadidateKFs;
    float similarity;

    bestSimilarity = 0;

    for(std::map<int, Ptr<KeyFrame> >::const_iterator iter = DataBase.begin(); iter != DataBase.end(); iter++)
    {
        if( currentSubmapID != -1 && currentSubmapID == iter->second->submapID)
            continue;
        similarity = score(currentFeature, iter->second->DNNFeature);

        if(similarity > similarityLow)
        {
            cadidateKFs.push_back(iter->first);
        }

        if(similarity > bestSimilarity)
        {
            bestSimilarity = similarity;
            bestId = iter->first;
        }
    }

    return cadidateKFs;
}

// If size of DB is large than the maxSizeDB, then remove some KFs in DB.
void KeyFrameDatabase::shrinkDB()
{
    for(std::map<int, Ptr<KeyFrame> >::const_iterator iter = DataBase.begin(); iter != DataBase.end(); iter++)
    {
        deleteKeyFrameByID(iter->first);

        iter++;

        if(iter == DataBase.end())
        {
            break;
        }
    }
}

// Debug Only.
void KeyFrameDatabase::printDB()
{
    for(std::map<int, Ptr<KeyFrame> >::const_iterator iter = DataBase.begin(); iter != DataBase.end(); iter++)
    {
        std::cout<<"frame Id= "<<iter->first<<", feature = "<<iter->second->DNNFeature<<std::endl;
        std::cout<<"ORB Feature"<<iter->second->ORBFeatures<<std::endl;
    }
}

void KeyFrameDatabase::reset()
{
    DataBase.clear();
}

}// namespace large_kinfu
}// namespace cv
