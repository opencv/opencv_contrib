// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef __OPENCV_LOOP_CLOSURE_DETECTION_HPP__
#define __OPENCV_LOOP_CLOSURE_DETECTION_HPP__

#include <opencv2/features2d.hpp>

namespace cv
{
namespace large_kinfu
{

class LoopClosureDetectionImpl : public LoopClosureDetection
{
public:
    LoopClosureDetectionImpl(const String& _dbowPath, double _simThreshold);

    bool addFrame(InputArray img, const int frameID, const int submapID, int& tarSubmapID) CV_OVERRIDE;

    void reset() CV_OVERRIDE;

private:
    int nfeatures = 20;
    Ptr<Feature2D> detector = ORB::create(nfeatures);
    std::vector<KeyPoint> keypoints;
    Mat descriptors;

    DBOWTrainer dbow = DBOWTrainer(10, 5);
    std::vector<DBOWTrainer::BOWVector> bowVectors;
    std::vector<int> frameIDs;
    std::vector<int> submapIDs;
    DBOWTrainer::BOWVector bowVector;
    double simThreshold;
};

}
}
#endif
