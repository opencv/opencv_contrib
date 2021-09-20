// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "precomp.hpp"
#include "loop_closure_detection.hpp"

namespace cv
{
namespace large_kinfu
{

LoopClosureDetectionImpl::LoopClosureDetectionImpl(const String& _dbowPath, double _simThreshold)
{
    std::cout << "Loading DBoW from " << _dbowPath << "...\n";
    dbow.load(_dbowPath);
    simThreshold = _simThreshold;
    reset();
}

bool LoopClosureDetectionImpl::addFrame(InputArray _img, const int frameID, const int submapID, int& tarSubmapID)
{
    CV_Assert(!_img.empty());

    Mat img;
    if (_img.isUMat())
        _img.copyTo(img);
    else
        img = _img.getMat();

    // Detect and transform features
    detector->detectAndCompute(img, cv::Mat(), keypoints, descriptors);
    dbow.transform(descriptors, bowVector);
    bowVectors.push_back(bowVector);
    frameIDs.push_back(frameID);
    submapIDs.push_back(submapID);

    if (bowVectors.size() > 2)
    {
        double score, maxScore = INT_MIN;
        double priorSimilarity = dbow.score(bowVector, bowVectors[(int)bowVectors.size() - 2]);

        for (int i = 0; i < (int)bowVectors.size() - 2; i++)
        {
            score = dbow.score(bowVector, bowVectors[i]) / priorSimilarity;
            if (score > maxScore)
            {
                maxScore = score;
                if (maxScore >= simThreshold)
                    tarSubmapID = submapIDs[i];
            }
        }
    }

    if (tarSubmapID != -1)
        return true;
    return false;
}

void LoopClosureDetectionImpl::reset()
{
    bowVectors.clear();
    frameIDs.clear();
    submapIDs.clear();
}

Ptr<LoopClosureDetection> LoopClosureDetection::create(const String& _dbowPath,  double _simThreshold)
{
    return makePtr<LoopClosureDetectionImpl>(_dbowPath, _simThreshold);
}

} // namespace large_kinfu
}// namespace cv
