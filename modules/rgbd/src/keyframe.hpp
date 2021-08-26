// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef __OPENCV_RGBD_KEYFRAME_HPP__
#define __OPENCV_RGBD_KEYFRAME_HPP__

#include "opencv2/core.hpp"

namespace cv
{
namespace large_kinfu
{

// It works like doubly linked list.
struct KeyFrame
{

    // DNN Feature.
    Mat DNNFeature;
    int submapID;

    int preKeyFrameID;
    int nextKeyFrameID;

    // ORB Feature
    std::vector<KeyPoint> keypoints;
    Mat ORBFeatures;

    KeyFrame();
    KeyFrame(Mat DNNfeature, int submapID);
    KeyFrame(Mat DNNfeature, int submapID, int preKeyFrameID);
    KeyFrame(Mat DNNfeature, int submapID, int preKeyFrameID, std::vector<KeyPoint> keypoints, Mat ORBdescriptors);
};

class KeyFrameDatabase
{
public:

    KeyFrameDatabase();

    KeyFrameDatabase(int maxSizeDB);

    ~KeyFrameDatabase() = default;

    void addKeyFrame( const Mat& DNNFeature, int frameID, int submapID);

    void addKeyFrame( const Mat& DNNFeature, int frameID, int submapID, std::vector<KeyPoint>& keypoints, const Mat& ORBFeatures);

    Ptr<KeyFrame> getKeyFrameByID(int keyFrameID);

    bool deleteKeyFrameByID(int keyFrameID);

    size_t getSize();

    bool empty();

    void reset();

    int getLastKeyFrameID();

    std::vector<int> getCandidateKF(const Mat& currentFeature, const int currentSubmapID, const double& similarityLow, double& bestSimilarity, int& bestId);

    double score(InputArray feature1, InputArray feature2);

    // Debug only
    void printDB();

private:
    void addKeyFrameT(const Mat& DNNFeature, int frameID, int submapID, std::vector<KeyPoint>& keypoints, const Mat& ORBFeatures);

    void shrinkDB();

    // < keyFrameID, KeyFrame>
    std::map<int, Ptr<KeyFrame> > DataBase;

    int maxSizeDB;
    int lastKeyFrameID;


};

}// namespace large_kinfu
}// namespace cv

#endif
