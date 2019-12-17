// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html
#include "precomp.hpp"
#include "logos/Logos.hpp"

namespace cv
{
namespace xfeatures2d
{
void matchLOGOS(const std::vector<KeyPoint>& keypoints1, const std::vector<KeyPoint>& keypoints2,
                const std::vector<int>& nn1, const std::vector<int>& nn2, std::vector<DMatch>& matches1to2)
{
    CV_CheckEQ(keypoints1.size(), nn1.size(), "Number of keypoints1 must be equal to the number of nn1.");
    CV_CheckEQ(keypoints2.size(), nn2.size(), "Number of keypoints2 must be equal to the number of nn2.");
    if (keypoints1.empty() || keypoints2.empty())
    {
        return;
    }

    std::vector<logos::Point*> vP1, vP2;
    vP1.reserve(keypoints1.size());
    vP2.reserve(keypoints2.size());

    for (size_t i = 0; i < keypoints1.size(); i++)
    {
        logos::Point* pt1 = new logos::Point(keypoints1[i].pt.x, keypoints1[i].pt.y,
                                             static_cast<float>(keypoints1[i].angle*CV_PI/180),
                                             keypoints1[i].size, nn1[i]);
        vP1.push_back(pt1);
    }

    for (size_t i = 0; i < keypoints2.size(); i++)
    {
        logos::Point* pt2 = new logos::Point(keypoints2[i].pt.x, keypoints2[i].pt.y,
                                             static_cast<float>(keypoints2[i].angle*CV_PI/180),
                                             keypoints2[i].size, nn2[i]);
        vP2.push_back(pt2);
    }

    logos::Logos logos;
    std::vector<logos::PointPair*> globalMatches;
    logos.estimateMatches(vP1, vP2, globalMatches);

    matches1to2.clear();
    matches1to2.reserve(globalMatches.size());
    for (size_t i = 0; i < globalMatches.size(); i++)
    {
        logos::PointPair* pp = globalMatches[i];
        matches1to2.push_back(DMatch(pp->getPos1(), pp->getPos2(), 0));
    }

    for (size_t i = 0; i < globalMatches.size(); i++)
    {
        delete globalMatches[i];
    }

    for (size_t i = 0; i < vP1.size(); i++)
    {
        delete vP1[i];
    }

    for (size_t i = 0; i < vP2.size(); i++)
    {
        delete vP2[i];
    }
}
} //namespace xfeatures2d
} //namespace cv
