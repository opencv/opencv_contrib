#include "opencv2/slam/visualizer.hpp"
#include "opencv2/slam/visual_odometry.hpp"
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

namespace cv {
namespace vo {

Visualizer::Visualizer(int W, int H, double meters_per_pixel)
    : meters_per_pixel_(meters_per_pixel), mapSize_(W,H)
{
    map_ = Mat::zeros(mapSize_, CV_8UC3);
}

Point Visualizer::worldToPixel(const Point2d &p) const {
    Point2d origin(mapSize_.width/2.0, mapSize_.height/2.0);
    int x = int(origin.x + p.x / meters_per_pixel_);
    int y = int(origin.y - p.y / meters_per_pixel_);
    return Point(x,y);
}

void Visualizer::addPose(double x, double z){
    traj_.emplace_back(x,-z);
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

void Visualizer::drawTrackingInfo(Mat &frame, const TrackingResult &res){
    if(frame.empty()) return;
    std::string info = res.ok ? "TRACKING" : "NOT TRACKING";
    info += " matches=" + std::to_string(res.numMatches);
    info += " inliers=" + std::to_string(res.numInliers);
    info += " state=" + std::to_string(res.state);
    Scalar color = res.ok ? Scalar(0, 255, 0) : Scalar(0, 0, 255);
    putText(frame, info, Point(10, 30), FONT_HERSHEY_SIMPLEX, 0.7, color, 2, LINE_AA);
}

void Visualizer::showTopdown(){
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
                // arrow length in world meters
                double arrow_m = 0.5; // 0.5 meters
                // tail is behind the current position by arrow_m, head (tip) at current position
                Point2d tail_world(traj_.back().x - dx * arrow_m, traj_.back().y - dz * arrow_m);
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