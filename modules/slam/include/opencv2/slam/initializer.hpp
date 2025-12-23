#pragma once
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <vector>
#include "opencv2/slam/keyframe.hpp"
#include "opencv2/slam/map.hpp"

namespace cv {
namespace vo {

class Initializer {
public:
    Initializer();

    // Attempt initialization with two frames
    // Returns true if initialization successful
    bool initialize(const std::vector<KeyPoint> &kps1,
                   const std::vector<KeyPoint> &kps2,
                   const std::vector<DMatch> &matches,
                   double fx, double fy, double cx, double cy,
                   Mat &R, Mat &t,
                   std::vector<Point3d> &points3D,
                   std::vector<bool> &isTriangulated);

    // Check if frames have sufficient parallax for initialization
    static bool checkParallax(const std::vector<KeyPoint> &kps1,
                             const std::vector<KeyPoint> &kps2,
                             const std::vector<DMatch> &matches,
                             double minMedianParallax = 15.0);

private:
    // Reconstruct from Homography
    bool reconstructH(const std::vector<Point2f> &pts1,
                     const std::vector<Point2f> &pts2,
                     const Mat &H,
                     double fx, double fy, double cx, double cy,
                     Mat &R, Mat &t,
                     std::vector<Point3d> &points3D,
                     std::vector<bool> &isTriangulated,
                     float &parallax);

    // Reconstruct from Fundamental/Essential
    bool reconstructF(const std::vector<Point2f> &pts1,
                     const std::vector<Point2f> &pts2,
                     const Mat &F,
                     double fx, double fy, double cx, double cy,
                     Mat &R, Mat &t,
                     std::vector<Point3d> &points3D,
                     std::vector<bool> &isTriangulated,
                     float &parallax);

    // Check reconstructed points quality
    int checkRT(const Mat &R, const Mat &t,
               const std::vector<Point2f> &pts1,
               const std::vector<Point2f> &pts2,
               const std::vector<Point3d> &points3D,
               std::vector<bool> &isGood,
               double fx, double fy, double cx, double cy,
               float &parallax);

    // Triangulate points
    void triangulate(const Mat &P1, const Mat &P2,
                    const std::vector<Point2f> &pts1,
                    const std::vector<Point2f> &pts2,
                    std::vector<Point3d> &points3D);

    // Decompose Homography
    void decomposeH(const Mat &H, std::vector<Mat> &Rs,
                   std::vector<Mat> &ts, std::vector<Mat> &normals);

    // Compute homography score
    float computeScore(const Mat &H21, const Mat &H12,
                      const std::vector<Point2f> &pts1,
                      const std::vector<Point2f> &pts2,
                      std::vector<bool> &inliersH,
                      float sigma = 1.0);

    // Compute fundamental score
    float computeScoreF(const Mat &F21,
                       const std::vector<Point2f> &pts1,
                       const std::vector<Point2f> &pts2,
                       std::vector<bool> &inliersF,
                       float sigma = 1.0);
};

} // namespace vo
} // namespace cv