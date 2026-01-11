#pragma once
#include <opencv2/core.hpp>
#include <vector>
#include <string>
#include "opencv2/slam/tracker.hpp"

namespace cv {
namespace vo {

struct TrackingResult;

class CV_EXPORTS Visualizer {
public:
    Visualizer(int W=1000, int H=800, double meters_per_pixel=0.02);
    // Update trajectory (x,z coordinates)
    void addPose(double x, double z);
    void clearTrajectory();
    void setTrajectoryXZ(const std::vector<cv::Point2d> &xz);
    // Show frame window
    void showFrame(const Mat &frame);
    // Overlay tracking info text onto a frame
    void drawTrackingInfo(Mat &frame, const TrackingResult &res);
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