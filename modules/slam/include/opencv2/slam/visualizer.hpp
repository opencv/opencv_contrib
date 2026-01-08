#pragma once
#include <opencv2/core.hpp>
#include <vector>
#include <string>
#include "opencv2/slam/feature.hpp"
#include "opencv2/slam/matcher.hpp"

namespace cv {
namespace vo {

class Tracker {
public:
    Tracker();
    // Process a gray image, returns true if a pose was estimated. imgOut contains visualization (matches or keypoints)
    bool processFrame(const Mat &gray, const std::string &imagePath, Mat &imgOut, Mat &R_out, Mat &t_out, std::string &info);
private:
    FeatureExtractor feat_;
    Matcher matcher_;

    Mat prevGray_, prevDesc_;
    std::vector<KeyPoint> prevKp_;
    int frame_id_;
};

class Visualizer {
public:
    Visualizer(int W=1000, int H=800, double meters_per_pixel=0.02);
    // Update trajectory (x,z coordinates)
    void addPose(double x, double z);
    void clearTrajectory();
    void setTrajectoryXZ(const std::vector<cv::Point2d> &xz);
    // Show frame window
    void showFrame(const Mat &frame);
    // Show top-down trajectory
    void showTopdown();
    // Save trajectory image to file
    bool saveTrajectory(const std::string &path);
private:
    double meters_per_pixel_;
    Size mapSize_;
    Mat map_;
    std::vector<Point2d> traj_;
    Point worldToPixel(const Point2d &p) const;
};

} // namespace vo
} // namespace cv