#include "opencv2/slam/pose.hpp"
#include <opencv2/calib3d.hpp>

namespace cv {
namespace vo {

    bool PoseEstimator::estimate(const std::vector<Point2f> &pts1,
                             const std::vector<Point2f> &pts2,
                             double fx, double fy, double cx, double cy,
                             Mat &R, Mat &t, Mat &mask, int &inliers)
{
    if(pts1.size() < 8 || pts2.size() < 8) { inliers = 0; return false; }
    double focal = (fx + fy) * 0.5;
    Point2d pp(cx, cy);
    // If provided principal point looks invalid (e.g. zeros or tiny values),
    // fall back to the approximate center of the matched points' bounding box.
    // This is better than leaving (0,0) which can break essential-matrix estimation.
    if((pp.x <= 2.0 || pp.y <= 2.0) && !pts1.empty()){
        float minx = std::numeric_limits<float>::max();
        float miny = std::numeric_limits<float>::max();
        float maxx = std::numeric_limits<float>::lowest();
        float maxy = std::numeric_limits<float>::lowest();
        for(size_t i=0;i<pts1.size();++i){
            const Point2f &p1 = pts1[i];
            const Point2f &p2 = pts2[i];
            minx = std::min(minx, std::min(p1.x, p2.x));
            miny = std::min(miny, std::min(p1.y, p2.y));
            maxx = std::max(maxx, std::max(p1.x, p2.x));
            maxy = std::max(maxy, std::max(p1.y, p2.y));
        }
        // center of bounding box of matched points
        pp.x = (minx + maxx) * 0.5;
        pp.y = (miny + maxy) * 0.5;
    }
    mask.release();
    Mat E = findEssentialMat(pts1, pts2, focal, pp, RANSAC, 0.999, 1.0, mask);
    if(E.empty()) { inliers = 0; return false; }
    inliers = recoverPose(E, pts1, pts2, R, t, focal, pp, mask);
    return true;
}

} // namespace vo
} // namespace cv