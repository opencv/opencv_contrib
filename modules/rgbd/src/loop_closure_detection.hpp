// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef __OPENCV_LOOP_CLOSURE_DETECTION_H__
#define __OPENCV_LOOP_CLOSURE_DETECTION_H__

#include "opencv2/dnn.hpp"
#include "keyframe.hpp"

#ifdef HAVE_OPENCV_FEATURES2D
#include <opencv2/features2d.hpp>
#endif

namespace cv{
namespace large_kinfu{

class LoopClosureDetectionImpl : public LoopClosureDetection
{
public:
    LoopClosureDetectionImpl(const String& modelBin, const String& modelTxt, const Size& input_size, int backendId = 0, int targetId = 0);

    bool addFrame(InputArray img, const int frameID, const int submapID, int& tarSubmapID) CV_OVERRIDE;

    bool loopCheck(int& tarSubmapID);

    void reset() CV_OVERRIDE;

    void processFrame(InputArray img, Mat& DNNfeature,std::vector<KeyPoint>& currentKeypoints, Mat& ORBFeature);

    bool ORBMather(InputArray feature1, InputArray feature2);

    bool newFrameCheck();

    void ORBExtract();

private:
    Ptr<KeyFrameDatabase> KFDataBase;
    Ptr<dnn::Net> net;
    Size inputSize;
    int currentFrameID;
    std::vector<KeyPoint> currentKeypoints;
    Mat currentDNNFeature;
    Mat currentORBFeature;
    Ptr<KeyFrame> bestLoopFrame;

    int currentSubmapID = -1;

#ifdef HAVE_OPENCV_FEATURES2D
    Ptr<FeatureDetector> ORBdetector = ORB::create();
    Ptr<DescriptorExtractor> ORBdescriptor = ORB::create();
    Ptr<DescriptorMatcher> ORBmatcher = DescriptorMatcher::create("BruteForce-Hamming");
#endif
    size_t ORBminMathing = 10;

    int minDatabaseSize = 50;
    int maxDatabaseSize = 2000;

    int preLoopedKFID = -1;

    // Param: HF-Net
    // Github Link: https://github.com/ethz-asl/hfnet
    std::vector<String> outNameDNN;
    double similarityHigh = 0.80;
    double similarityLow = 0.84;

};

}
}
#endif
