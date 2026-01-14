#include "opencv2/slam/visualizer.hpp"
#include "opencv2/slam/visual_odometry.hpp"
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <algorithm>

namespace cv {
namespace vo {

Visualizer::Visualizer(int W, int H, double meters_per_pixel)
    : meters_per_pixel_(meters_per_pixel), dynamic_meters_per_pixel_(meters_per_pixel), world_center_(0.0, 0.0), mapSize_(W,H)
{
    map_ = Mat::zeros(mapSize_, CV_8UC3);
}

void Visualizer::updateViewport(){
    if(traj_.empty()){
        dynamic_meters_per_pixel_ = meters_per_pixel_;
        world_center_ = Point2d(0.0, 0.0);
        return;
    }

    double minx = traj_[0].x, maxx = traj_[0].x;
    double minz = traj_[0].y, maxz = traj_[0].y;
    for(const auto &p : traj_){
        minx = std::min(minx, p.x); maxx = std::max(maxx, p.x);
        minz = std::min(minz, p.y); maxz = std::max(maxz, p.y);
    }
    double rangeX = std::max(1e-6, maxx - minx);
    double rangeZ = std::max(1e-6, maxz - minz);
    world_center_ = Point2d(0.5 * (minx + maxx), 0.5 * (minz + maxz));

    const double marginPx = 40.0;
    double availX = std::max(1.0, mapSize_.width - 2.0 * marginPx);
    double availZ = std::max(1.0, mapSize_.height - 2.0 * marginPx);
    double required = std::max(rangeX / availX, rangeZ / availZ);
    // If required is tiny (small trajectory), zoom in but cap by nominal scale to avoid extreme zoom.
    dynamic_meters_per_pixel_ = std::max(required, meters_per_pixel_ * 0.1);
}

Point Visualizer::worldToPixel(const Point2d &p) const {
    Point2d origin(mapSize_.width/2.0, mapSize_.height/2.0);
    // p.x is world x, p.y is world z (stored directly from addPose)
    int x = int(origin.x + (p.x - world_center_.x) / dynamic_meters_per_pixel_);
    int y = int(origin.y - (p.y - world_center_.y) / dynamic_meters_per_pixel_);  // -p.y because image y increases downward
    return Point(x,y);
}

void Visualizer::addPose(double x, double z){
    // worldToPixel will handle the coordinate transformation
    traj_.emplace_back(x, z);
}

void Visualizer::clearTrajectory(){
    traj_.clear();
}

void Visualizer::setTrajectoryXZ(const std::vector<cv::Point2d> &xz){
    traj_ = xz;
}

void Visualizer::showFrame(const Mat &frame){
    if(frame.empty()) return;
    // Do not draw heading overlay on video frames; only show raw frame.
    imshow("frame", frame);
}

void Visualizer::drawTrackingInfo(Mat &frame, const TrackingResult &res, int frame_id){
    if(frame.empty()) return;
    // Display frame index (1-based if provided), then status and simple stats.
    std::ostringstream ss;
    if(frame_id > 0) ss << "Frame=" << frame_id << " ";
    ss << (res.ok ? "TRACKING" : "NOT TRACKING");
    ss << " matches=" << res.numMatches;
    ss << " inliers=" << res.numInliers;

    Scalar color = res.ok ? Scalar(0, 255, 0) : Scalar(0, 0, 255);
    putText(frame, ss.str(), Point(10, 30), FONT_HERSHEY_SIMPLEX, 0.7, color, 2, LINE_AA);

    // Overlay colors: pure green = matches, pure blue = inliers (inliers override matches)
    const Scalar matchColor(0, 255, 0);   // pure green (B,G,R)
    const Scalar inlierColor(255, 0, 0);  // pure blue (B,G,R)
    const int radius = 2;
    const int thickness = 1;
    size_t N = res.matchPoints.size();
    if(N){
        for(size_t i = 0; i < N; ++i){
            bool isInlier = (i < res.inlierMask.size() && res.inlierMask[i]);
            const Scalar &c = isInlier ? inlierColor : matchColor;
            Point pt(cvRound(res.matchPoints[i].x), cvRound(res.matchPoints[i].y));
            circle(frame, pt, radius, c, thickness, LINE_AA);
        }
    }
}

void Visualizer::showTopdown(){
    updateViewport();
    map_ = Mat::zeros(mapSize_, CV_8UC3);
    for (int gx = 0; gx < mapSize_.width; gx += 50) line(map_, Point(gx,0), Point(gx,mapSize_.height), Scalar(30,30,30), 1);
    for (int gy = 0; gy < mapSize_.height; gy += 50) line(map_, Point(0,gy), Point(mapSize_.width,gy), Scalar(30,30,30), 1);
    for(size_t i=1;i<traj_.size();++i){
        Point p1 = worldToPixel(traj_[i-1]);
        Point p2 = worldToPixel(traj_[i]);
        line(map_, p1, p2, Scalar(0,255,0), 2);
    }
    if(!traj_.empty()){
        Point p = worldToPixel(traj_.back());
        // draw heading arrow on topdown map based on recent trajectory
        if(traj_.size() >= 2){
            size_t K = std::min<size_t>(5, traj_.size()-1);
            double dx = 0.0, dz = 0.0;
            for(size_t i=0;i<K;i++){
                auto a = traj_[traj_.size()-1 - i];
                auto b = traj_[traj_.size()-2 - i];
                dx += (a.x - b.x);
                dz += (a.y - b.y);
            }
            dx /= K; dz /= K;
            double norm = std::hypot(dx, dz);
            if(norm > 1e-6){
                dx /= norm; dz /= norm;
                // Keep arrow length constant in screen pixels regardless of world scaling.
                const double arrow_px = 25.0; // pixels
                const double arrow_world = arrow_px * dynamic_meters_per_pixel_;
                Point2d tail_world(traj_.back().x - dx * arrow_world, traj_.back().y - dz * arrow_world);
                Point tail_px = worldToPixel(tail_world);
                arrowedLine(map_, tail_px, p, Scalar(0,0,255), 2, LINE_AA, 0, 0.3);
            }
        }
        // label near current position
        putText(map_, "Robot", p + Point(10,-10), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255,255,255), 1);
    }
    imshow("topdown", map_);
}

bool Visualizer::saveTrajectory(const std::string &path){
    if(map_.empty()) showTopdown();
    return imwrite(path, map_);
}

} // namespace vo
} // namespace cv