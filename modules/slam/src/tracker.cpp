// SPDX-License-Identifier: Apache-2.0
#include "opencv2/slam/tracker.hpp"

#include <opencv2/features2d.hpp>

namespace cv {
namespace vo {

Tracker::Tracker()
    : feat_(), matcher_(), frame_id_(0)
{
}

bool Tracker::processFrame(const Mat &gray, const std::string & /*imagePath*/, Mat &imgOut, Mat & /*R_out*/, Mat & /*t_out*/, std::string &info)
{
    if(gray.empty()) return false;
    // detect
    std::vector<KeyPoint> kps;
    Mat desc;
    feat_.detectAndCompute(gray, kps, desc);

    if(!prevGray_.empty() && !prevDesc_.empty() && !desc.empty()){
        // match
        std::vector<DMatch> goodMatches;
        matcher_.knnMatch(prevDesc_, desc, goodMatches);

        // draw matches for visualization
        drawMatches(prevGray_, prevKp_, gray, kps, goodMatches, imgOut,
                        Scalar::all(-1), Scalar::all(-1), std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

        // For now, Tracker is visualization-only; pose is computed by VisualOdometry/PoseEstimator.
        info = "matches=" + std::to_string(goodMatches.size()) + ", inliers=0";
    } else {
        // first frame: draw keypoints
        drawKeypoints(gray, kps, imgOut, Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        info = "first_frame";
    }

    // update prev
    prevGray_ = gray.clone();
    prevKp_ = kps;
    prevDesc_ = desc.clone();
    frame_id_++;
    return true;
}

} // namespace vo
} // namespace cv
